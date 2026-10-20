"""The main sort pass: probe -> (too short? | speech? | vision) -> place -> catalog."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from . import catalog, ffmpeg_utils
from .transcribe import WhisperEngine, mini_subject
from .vision import VisionEngine

logger = logging.getLogger("smart_video_sorter")

BROLL_FOLDERS = {
    "Office": "Office",
    "Interior": "Interior",
    "Outdoor": "Outdoor",
    "Screen_Recording": "Screen_Recording",
    "Events": "Events",
    "General": "Misc",
}


@dataclass
class SortOptions:
    source: Path
    target: Path
    action: str = "copy"
    short_clip_seconds: float = 2.0
    frames_per_video: int = 9
    event_depth: int = 1
    batch_size: int | None = None
    dry_run: bool = False
    temp_dir: Path | None = None


@dataclass
class SortStats:
    counts: dict = field(default_factory=lambda: {"SHORT_CLIP": 0, "TALKING_HEAD": 0, "B-ROLL": 0, "ERROR": 0})


def discover_videos(source: Path, target: Path) -> list[Path]:
    """All video files under source, excluding anything already inside target."""
    target_resolved = target.resolve()
    found = []
    for p in source.rglob("*"):
        if not p.is_file() or not ffmpeg_utils.is_video_file(p):
            continue
        try:
            if target_resolved in p.resolve().parents:
                continue
        except OSError:
            pass
        found.append(p)
    return sorted(found)


def sort_videos(opts: SortOptions, whisper: WhisperEngine, vision: VisionEngine, console: Console) -> SortStats:
    temp_root = opts.temp_dir or (opts.target / "_temp")
    conn = catalog.init_db(opts.target / "video_catalog.db")
    stats = SortStats()

    videos = discover_videos(opts.source, opts.target)
    pending = [v for v in videos if not catalog.is_processed(conn, v)]
    if opts.batch_size:
        pending = pending[: opts.batch_size]
    console.print(f"Found [bold]{len(videos)}[/bold] videos, [bold]{len(pending)}[/bold] to process in this run.")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  console=console, transient=True) as progress:

        def done(dest: Path, detail: str = "") -> None:
            rel = dest.parent.relative_to(opts.target) if dest.is_relative_to(opts.target) else dest.parent
            progress.console.print(f"[green]✓[/green] {video.name:<26} → [bold]{rel.as_posix()}[/bold] [dim]{detail}[/dim]")

        task = progress.add_task("Processing", total=len(pending))
        for video in pending:
            src = ffmpeg_utils.VideoSource(video, temp_root)
            event = catalog.derive_event_name(video, opts.source, opts.event_depth)
            try:
                progress.update(task, description=f"[cyan]{video.name}[/cyan] probing")
                probe = ffmpeg_utils.probe_video(src)
                base = dict(source_path=str(video), duration=probe.duration,
                            resolution=f"{probe.width}x{probe.height}", codec=probe.codec, event=event)

                # 1. Fragments too short for Whisper or vision go straight to a raw bin.
                if probe.duration < opts.short_clip_seconds:
                    dest = catalog.place_file(video, opts.target / event / "_Short_Raw", opts.action, opts.dry_run)
                    catalog.record_entry(conn, catalog.CatalogEntry(**base, status="done", category="SHORT_CLIP", dest_path=str(dest)))
                    stats.counts["SHORT_CLIP"] += 1
                    done(dest, f"{probe.duration:.1f}s, too short to analyse")
                    continue

                # 2. Speech? -> Talking head, with transcript next to the file.
                transcript = None
                if probe.has_audio:
                    progress.update(task, description=f"[cyan]{video.name}[/cyan] whisper")
                    temp_root.mkdir(parents=True, exist_ok=True)
                    wav = temp_root / f"{video.stem}_{int(time.time() * 1000)}.wav"
                    ffmpeg_utils.extract_audio(src, wav)
                    transcript = whisper.transcribe(wav)
                    wav.unlink(missing_ok=True)

                if transcript and transcript.has_speech:
                    dest = catalog.place_file(video, opts.target / event / "Talking_Heads", opts.action, opts.dry_run)
                    txt = dest.with_suffix(".txt")
                    if not opts.dry_run:
                        txt.write_text(transcript.text, encoding="utf-8")
                    catalog.record_entry(conn, catalog.CatalogEntry(
                        **base, status="done", category="TALKING_HEAD", dest_path=str(dest),
                        language=transcript.language, word_count=transcript.word_count,
                        mini_subject=mini_subject(transcript.text), transcript_path=str(txt)))
                    stats.counts["TALKING_HEAD"] += 1
                    done(dest, f"speech: {transcript.language}, {transcript.word_count} words")
                    continue

                # 3. No speech -> B-roll, classified by a frame-grid vote.
                progress.update(task, description=f"[cyan]{video.name}[/cyan] vision")
                frames_dir = temp_root / f"frames_{video.stem}_{int(time.time() * 1000)}"
                frames = ffmpeg_utils.extract_frame_grid(src, probe.duration, frames_dir, opts.frames_per_video)
                category, description = vision.classify_frames(frames) if frames else ("General", "")
                shutil.rmtree(frames_dir, ignore_errors=True)

                folder = BROLL_FOLDERS.get(category, "Misc")
                dest = catalog.place_file(video, opts.target / event / "B-Roll" / folder, opts.action, opts.dry_run)
                catalog.record_entry(conn, catalog.CatalogEntry(
                    **base, status="done", category=f"B-ROLL:{category}", dest_path=str(dest),
                    visual_description=description))
                stats.counts["B-ROLL"] += 1
                done(dest, (description[:60] + "…") if len(description) > 60 else description)

            except Exception as exc:  # one bad file must never stop a 1,000-file run
                logger.error("Failed on %s: %s", video.name, exc)
                catalog.record_entry(conn, catalog.CatalogEntry(source_path=str(video), status="error", error=str(exc)))
                stats.counts["ERROR"] += 1
            finally:
                src.cleanup()
                progress.advance(task)

    write_catalog_exports(conn, opts.target)
    conn.close()
    return stats


def write_catalog_exports(conn, target: Path) -> Path:
    """Writes catalog.csv (editor-friendly). Falls back to a timestamped name if
    the file is locked, e.g. open in Excel."""
    out = target / "catalog.csv"
    try:
        catalog.export_editor_csv(conn, out)
    except OSError as exc:
        fallback = target / f"catalog_{int(time.time())}.csv"
        logger.warning("Could not write %s (%s). Writing %s instead.", out, exc, fallback)
        catalog.export_editor_csv(conn, fallback)
        out = fallback
    return out
