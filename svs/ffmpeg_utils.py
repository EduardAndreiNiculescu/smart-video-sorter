"""Wraps ffprobe/ffmpeg calls with an auto-fallback strategy for network drives.

Streaming reads/seeks directly off a WebDAV-mapped drive (e.g. Z:\\) are attempted
first. If ffmpeg/ffprobe times out or fails, the source file is copied once to a
local temp dir and every subsequent operation for that file reuses the local copy.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("smart_video_sorter")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm", ".mts", ".m2ts"}

PROBE_TIMEOUT = 20
OP_TIMEOUT = 120


class FFmpegError(RuntimeError):
    pass


@dataclass
class ProbeResult:
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    has_audio: bool


class VideoSource:
    """A video file that may live on a network drive.

    Call `active_path` to get whichever path (network or local temp copy) should
    currently be used. `fallback()` is invoked automatically by `run_resilient`
    the first time a network read fails; after that every call in this file's
    pipeline reuses the local copy. `cleanup()` removes the temp copy.
    """

    def __init__(self, original_path: Path, temp_dir: Path):
        self.original_path = original_path
        self.temp_dir = temp_dir
        self._local_copy: Path | None = None
        self.fell_back = False

    @property
    def active_path(self) -> Path:
        return self._local_copy if self._local_copy else self.original_path

    def fallback(self) -> None:
        if self.fell_back:
            return
        logger.warning(
            "Streaming read failed for %s; copying locally to temp for reliability",
            self.original_path.name,
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        dest = self.temp_dir / f"{uuid.uuid4().hex}_{self.original_path.name}"
        shutil.copy2(self.original_path, dest)
        self._local_copy = dest
        self.fell_back = True

    def cleanup(self) -> None:
        if self._local_copy and self._local_copy.exists():
            self._local_copy.unlink(missing_ok=True)
            self._local_copy = None


def _run(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.decode(errors="ignore")[-2000:])
    return result


def run_resilient(
    source: VideoSource,
    build_args: "callable[[Path], list[str]]",
    timeout: int = OP_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Runs an ffmpeg/ffprobe command against source.active_path, falling back
    to a local copy on timeout/failure (once per file)."""
    args = build_args(source.active_path)
    try:
        return _run(args, timeout)
    except (subprocess.TimeoutExpired, FFmpegError, OSError) as exc:
        if source.fell_back:
            raise
        logger.warning("ffmpeg op failed on %s (%s); retrying with local copy", source.original_path.name, exc)
        source.fallback()
        args = build_args(source.active_path)
        return _run(args, timeout)


def probe_video(source: VideoSource) -> ProbeResult:
    def build(path: Path) -> list[str]:
        return [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]

    result = run_resilient(source, build, timeout=PROBE_TIMEOUT)
    data = json.loads(result.stdout.decode(errors="ignore"))

    duration = float(data.get("format", {}).get("duration", 0.0))
    video_stream = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])

    if video_stream is None:
        raise FFmpegError("No video stream found")

    fps_raw = video_stream.get("avg_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except ValueError:
        fps = 0.0

    return ProbeResult(
        duration=duration,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        codec=video_stream.get("codec_name", "unknown"),
        has_audio=has_audio,
    )


def extract_audio(source: VideoSource, out_wav: Path) -> Path:
    def build(path: Path) -> list[str]:
        return [
            "ffmpeg", "-y", "-i", str(path),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav),
        ]

    run_resilient(source, build, timeout=OP_TIMEOUT)
    return out_wav


def extract_frame_grid(source: VideoSource, duration: float, out_dir: Path, count: int = 9) -> list[Path]:
    """Extracts `count` frames at evenly spaced timestamps (skipping the very
    first/last 5% to avoid black intro/outro frames)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    margin = duration * 0.05
    span = max(duration - 2 * margin, 0.1)
    frames: list[Path] = []

    for i in range(count):
        ts = margin + span * (i / max(count - 1, 1))
        out_path = out_dir / f"frame_{i:02d}.jpg"

        def build(path: Path, ts=ts, out_path=out_path) -> list[str]:
            return [
                "ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(path),
                "-frames:v", "1", "-q:v", "2", "-pix_fmt", "yuvj420p", str(out_path),
            ]

        try:
            run_resilient(source, build, timeout=60)
            if out_path.exists() and out_path.stat().st_size > 0:
                frames.append(out_path)
            else:
                logger.warning("Frame extraction produced no output at %.2fs for %s", ts, source.original_path.name)
        except (FFmpegError, subprocess.TimeoutExpired) as exc:
            logger.warning("Frame extraction failed at %.2fs for %s: %s", ts, source.original_path.name, exc)

    return frames


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS
