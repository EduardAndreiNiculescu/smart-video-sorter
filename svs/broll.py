"""Optional pass for event footage: split B-roll into event-specific shot types.

The first pass sorts B-roll into broad settings (Office, Outdoor, Events...).
For conference/gala archives that is not enough, so this pass re-captions every
B-roll clip and buckets it as Stage, Audience, Networking, Awards or Decor.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from . import catalog
from .ffmpeg_utils import VideoSource, extract_frame_grid
from .vision import VisionEngine

logger = logging.getLogger("smart_video_sorter")

# First match wins, so the most specific shot types come first.
EVENT_SHOT_KEYWORDS = [
    ("Awards", ["award", "trophy", "diploma", "ceremony", "plaque", "prize", "certificate", "medal"]),
    ("Stage", ["stage", "podium", "speaker", "microphone", "presenter", "lectern", "screen behind"]),
    ("Audience", ["audience", "crowd", "listeners", "seated", "spectators", "rows of chairs", "applause"]),
    ("Networking", ["networking", "conversation", "cocktail", "coffee break", "handshake", "people chatting",
                    "group of people", "standing"]),
    ("Decor", ["decor", "hall", "venue", "empty", "table setup", "banner", "lighting", "room"]),
]
EVENT_SHOTS = {f"B-ROLL:{name}" for name, _ in EVENT_SHOT_KEYWORDS} | {"B-ROLL:Event_Misc"}


def shot_type(captions: list[str]) -> str:
    text = " ".join(captions).lower()
    for name, keywords in EVENT_SHOT_KEYWORDS:
        if any(kw in text for kw in keywords):
            return f"B-ROLL:{name}"
    return "B-ROLL:Event_Misc"


def reclassify_event_broll(target: Path, vision: VisionEngine, batch_limit: int | None = None,
                           reverse: bool = False, frames: int = 6) -> dict:
    """`reverse=True` walks the queue from the end, so two workers can drain it
    from both sides at once (rows are re-checked, nothing is processed twice)."""
    conn = catalog.init_db(target / "video_catalog.db")
    temp_dir = target / ("_temp_tail" if reverse else "_temp")
    rows = [r for r in conn.execute(
        "SELECT source_path, dest_path, duration, category, event FROM videos WHERE category LIKE 'B-ROLL:%'"
    ).fetchall() if r[3] not in EVENT_SHOTS]
    if reverse:
        rows.reverse()
    total = len(rows)
    if batch_limit:
        rows = rows[:batch_limit]

    processed = errors = 0
    for i, (source_path, dest_path, duration, _, event) in enumerate(rows, 1):
        current = Path(dest_path or "")
        if not current.is_file():
            continue
        frame_dir = temp_dir / f"frames_{current.stem}"
        src = VideoSource(current, temp_dir)
        try:
            captions = [c for c in (vision.caption(f) for f in extract_frame_grid(src, float(duration or 10), frame_dir, frames)) if c]
            new_cat = shot_type(captions)
            new_path = catalog.unique_destination(target / (event or "Unsorted") / "B-Roll" / new_cat.split(":", 1)[1], current.name)
            shutil.move(str(current), str(new_path))
            conn.execute("UPDATE videos SET category = ?, visual_description = ?, dest_path = ? WHERE source_path = ?",
                         (new_cat, captions[0] if captions else "", str(new_path), source_path))
            conn.commit()
            processed += 1
            logger.info("[%d/%d] %s -> %s", i, len(rows), current.name, new_cat)
        except Exception as exc:
            logger.error("Reclassify failed for %s: %s", current.name, exc)
            errors += 1
        finally:
            src.cleanup()
            shutil.rmtree(frame_dir, ignore_errors=True)

    conn.close()
    return {"pending": total, "processed": processed, "errors": errors}
