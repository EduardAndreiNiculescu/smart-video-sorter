"""Cut social-media shorts out of talking heads, using the stored highlights.

By default cuts are stream copies (no re-encode), so hundreds of clips take
minutes, not hours; the start then snaps to the previous keyframe. Use
accurate=True to re-encode for frame-exact cuts. Each highlight is widened
symmetrically to a 15-60 s window, padded so words are not clipped, and
clamped to the source duration.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

from . import catalog

PAD_BEFORE = 0.5
PAD_AFTER = 0.5
MIN_SHORT = 15.0
MAX_SHORT = 60.0

_SLUG_RE = re.compile(r"[^a-zA-Z0-9ăâîșțĂÂÎȘȚ]+")


def slugify(text: str, max_len: int = 40) -> str:
    return _SLUG_RE.sub("_", text.strip()).strip("_")[:max_len] or "highlight"


def widen_window(start: float, end: float, video_duration: float,
                 min_len: float = MIN_SHORT, max_len: float = MAX_SHORT) -> tuple[float, float]:
    """Grows [start, end] around its centre to at least `min_len` seconds
    (never above `max_len`), shifting it back inside [0, video_duration]."""
    span = end - start
    goal = min(max(span, min_len), max_len)
    if goal <= span:
        return start, end
    extra = (goal - span) / 2
    new_start, new_end = start - extra, end + extra
    if new_start < 0:
        new_end, new_start = new_end - new_start, 0.0
    if video_duration and new_end > video_duration:
        new_start = max(0.0, new_start - (new_end - video_duration))
        new_end = video_duration
    return new_start, new_end


def cut_clip(source: Path, out_path: Path, start: float, end: float, duration: float, accurate: bool = False) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start, end = widen_window(start, end, duration)
    ss = max(0.0, start - PAD_BEFORE)
    to = min(duration, end + PAD_AFTER) if duration else end + PAD_AFTER
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{ss:.2f}", "-to", f"{to:.2f}", "-i", str(source),
         *(["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac"] if accurate else ["-c", "copy"]),
         "-avoid_negative_ts", "make_zero", str(out_path)],
        capture_output=True,
    )
    return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def cut_shorts(target: Path, accurate: bool = False) -> dict:
    conn = catalog.init_db(target / "video_catalog.db")
    rows = conn.execute(
        "SELECT dest_path, event, duration, highlights_json FROM videos "
        "WHERE category = 'TALKING_HEAD' AND highlights_json IS NOT NULL AND highlights_json != ''"
    ).fetchall()

    manifest, cut, skipped, errors = [], 0, 0, 0
    for dest_path, event, duration, highlights_json in rows:
        src = Path(dest_path or "")
        if not src.is_file():
            continue
        for i, h in enumerate(json.loads(highlights_json), 1):
            out = target / (event or "Unsorted") / "Shorts" / f"{src.stem}_short{i:02d}_{slugify(h.get('text', ''))}{src.suffix}"
            if out.exists():
                skipped += 1
            elif cut_clip(src, out, h["start"], h["end"], duration or 0, accurate):
                cut += 1
            else:
                errors += 1
                continue
            manifest.append([event, src.name, out.name, h.get("start_fmt", ""), h.get("end_fmt", ""),
                             h.get("score", ""), h.get("reason", ""), h.get("text", "")])
    conn.close()

    manifest_path = target / "shorts_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Event", "Source", "Short", "Start", "End", "Score", "Reason", "Quote"])
        writer.writerows(manifest)
    return {"cut": cut, "skipped": skipped, "errors": errors, "manifest": str(manifest_path)}
