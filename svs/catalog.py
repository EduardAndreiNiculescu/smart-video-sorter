"""SQLite catalog (resume state), event naming, file placement, and CSV export.

Every processed file gets one row keyed by its source path. A file whose row is
`status='done'` is skipped on the next run, so a sort over hundreds of GB can be
stopped and resumed at any time (or run in `--batch-size` slices).
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("smart_video_sorter")

_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*]')

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    source_path TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    category TEXT,
    dest_path TEXT,
    duration REAL,
    resolution TEXT,
    codec TEXT,
    language TEXT,
    word_count INTEGER,
    mini_subject TEXT,
    transcript_path TEXT,
    visual_description TEXT,
    event TEXT,
    highlights_json TEXT,
    error TEXT,
    processed_at TEXT
);
"""

# Columns added after the first release; init_db() migrates older catalogs in place.
_LATE_COLUMNS = (("visual_description", "TEXT"), ("event", "TEXT"), ("highlights_json", "TEXT"))


@dataclass
class CatalogEntry:
    source_path: str
    status: str  # "done" | "error"
    category: str | None = None
    dest_path: str | None = None
    duration: float | None = None
    resolution: str | None = None
    codec: str | None = None
    language: str | None = None
    word_count: int | None = None
    mini_subject: str | None = None
    transcript_path: str | None = None
    visual_description: str | None = None
    event: str | None = None
    error: str | None = None


def derive_event_name(video: Path, source_root: Path, depth: int = 1) -> str:
    """Groups a video under the folder `depth` levels below the source root.

    With depth=1 (default):
        <root>/2024 - Annual Gala/CamA/clip.MTS  -> "2024 - Annual Gala"
        <root>/clip.mp4                          -> "Unsorted"
    With depth=2, for trees like <root>/<camera operator>/<event>/...:
        <root>/Operator/2024 - Annual Gala/x.MOV -> "2024 - Annual Gala"
    """
    try:
        rel = video.resolve().relative_to(source_root.resolve()).parts[:-1]
    except ValueError:
        rel = video.parts[:-1]
    if not rel:
        return "Unsorted"
    event = rel[min(depth, len(rel)) - 1]
    event = _INVALID_FS_CHARS.sub("_", event.strip().rstrip("."))
    return event or "Unsorted"


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(SCHEMA)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(videos)")}
    for col, coltype in _LATE_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {coltype}")
    conn.commit()
    return conn


def is_processed(conn: sqlite3.Connection, source_path: Path) -> bool:
    row = conn.execute(
        "SELECT 1 FROM videos WHERE source_path = ? AND status = 'done'", (str(source_path),)
    ).fetchone()
    return row is not None


def record_entry(conn: sqlite3.Connection, entry: CatalogEntry) -> None:
    data = asdict(entry)
    data["processed_at"] = datetime.now(timezone.utc).isoformat()
    columns = ", ".join(data)
    placeholders = ", ".join("?" for _ in data)
    conn.execute(f"INSERT OR REPLACE INTO videos ({columns}) VALUES ({placeholders})", list(data.values()))
    conn.commit()


def unique_destination(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / filename
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{Path(filename).stem}_{i}{Path(filename).suffix}"
        i += 1
    return candidate


def place_file(src: Path, dest_dir: Path, action: str, dry_run: bool) -> Path:
    """action: 'move' or 'copy'. Never overwrites: name clashes get a _1, _2... suffix."""
    dest = unique_destination(dest_dir, src.name)
    if dry_run:
        logger.info("[DRY-RUN] %s %s -> %s", action, src, dest)
        return dest
    if action == "move":
        shutil.move(str(src), str(dest))
    elif action == "copy":
        shutil.copy2(str(src), str(dest))
    else:
        raise ValueError(f"Unknown action: {action}")
    return dest


def export_raw_csv(conn: sqlite3.Connection, out_csv: Path) -> None:
    """Dumps every column, oldest first."""
    cursor = conn.execute("SELECT * FROM videos ORDER BY processed_at")
    columns = [d[0] for d in cursor.description]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(cursor.fetchall())


def format_duration(seconds) -> str:
    if seconds is None:
        return ""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_category(category: str | None, error: str | None) -> str:
    if error:
        return "Error"
    if not category:
        return ""
    if category.startswith("B-ROLL:"):
        return "B-Roll: " + category.split(":", 1)[1].replace("_", " ")
    return {"TALKING_HEAD": "Talking Head", "SHORT_CLIP": "Short clip"}.get(category, category)


def export_editor_csv(conn: sqlite3.Connection, out_csv: Path) -> None:
    """Human-friendly catalog for video editors: sorted by event then category,
    durations as mm:ss, highlight counts instead of raw JSON."""
    rows = conn.execute(
        "SELECT event, category, dest_path, duration, resolution, language, word_count, "
        "visual_description, transcript_path, highlights_json, error FROM videos ORDER BY event, category"
    ).fetchall()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Event", "Category", "File", "Duration", "Resolution", "Language", "Words",
            "Visual description", "Highlights", "Video path", "Transcript path", "Error",
        ])
        for (event, category, dest, duration, res, lang, words, desc, transcript, hl_json, error) in rows:
            try:
                highlights = len(json.loads(hl_json)) if hl_json else ""
            except (json.JSONDecodeError, TypeError):
                highlights = ""
            writer.writerow([
                event or "", format_category(category, error), Path(dest).name if dest else "",
                format_duration(duration), res or "", lang or "", words or "", desc or "",
                highlights, dest or "", transcript or "", error or "",
            ])
