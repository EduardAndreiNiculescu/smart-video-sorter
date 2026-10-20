"""Second pass over talking heads: store scored highlight moments in the catalog."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import catalog
from .ffmpeg_utils import VideoSource, extract_audio
from .transcribe import WhisperEngine

logger = logging.getLogger("smart_video_sorter")


def find_highlights(target: Path, whisper: WhisperEngine, batch_limit: int | None = None) -> dict:
    conn = catalog.init_db(target / "video_catalog.db")
    temp_dir = target / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        "SELECT source_path, dest_path FROM videos WHERE category = 'TALKING_HEAD' "
        "AND (highlights_json IS NULL OR highlights_json = '')"
    ).fetchall()
    total = len(rows)
    if batch_limit:
        rows = rows[:batch_limit]

    processed = errors = 0
    for i, (source_path, dest_path) in enumerate(rows, 1):
        dest = Path(dest_path or "")
        if not dest.is_file():
            logger.warning("[%d/%d] missing file: %s", i, len(rows), dest)
            continue
        wav = temp_dir / f"{dest.stem}_hl.wav"
        src = VideoSource(dest, temp_dir)
        try:
            extract_audio(src, wav)
            highlights = whisper.transcribe(wav).highlights
            conn.execute("UPDATE videos SET highlights_json = ? WHERE source_path = ?",
                         (json.dumps(highlights, ensure_ascii=False), source_path))
            conn.commit()
            processed += 1
            logger.info("[%d/%d] %s: %d highlights", i, len(rows), dest.name, len(highlights))
        except Exception as exc:
            logger.error("Highlights failed for %s: %s", dest.name, exc)
            errors += 1
        finally:
            src.cleanup()
            wav.unlink(missing_ok=True)

    conn.close()
    return {"pending": total, "processed": processed, "errors": errors}
