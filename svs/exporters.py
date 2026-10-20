"""CSV & Markdown Exporter for sorted video catalog and highlights."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("smart_video_sorter")

CSV_HEADER = [
    "Filename",
    "File_Type",
    "Category",
    "Event_Edition",
    "Date_Extracted",
    "Guests_Speakers",
    "Highlights_Count",
    "Timestamps_Summary",
    "Full_Transcript_Snippet",
    "Source_Path"
]


class CatalogExporter:
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(self, records: List[Dict[str, Any]], filename: str = "catalog.csv") -> Path:
        """Export processed catalog to CSV file."""
        csv_path = self.target_dir / filename
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for rec in records:
                writer.writerow([
                    rec.get("filename", ""),
                    rec.get("file_type", ""),
                    rec.get("category", ""),
                    rec.get("event_edition", ""),
                    rec.get("date_extracted", ""),
                    ", ".join(rec.get("guests_speakers", [])),
                    len(rec.get("highlights", [])),
                    self._format_highlights_summary(rec.get("highlights", [])),
                    rec.get("transcript_snippet", "")[:300],
                    rec.get("source_path", ""),
                ])
        logger.info(f"Catalog exported to CSV: {csv_path}")
        return csv_path

    def export_json(self, records: List[Dict[str, Any]], filename: str = "catalog.json") -> Path:
        """Export detailed raw metadata to JSON."""
        json_path = self.target_dir / filename
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return json_path

    def export_markdown(self, records: List[Dict[str, Any]], filename: str = "catalog.md") -> Path:
        """Export clean Markdown report for video editors."""
        md_path = self.target_dir / filename
        lines = ["# 🎬 Smart Video Catalog Report\n"]
        lines.append(f"**Total files analyzed:** {len(records)}\n")
        lines.append("| Filename | Category | Event | Guests | Highlights |")
        lines.append("|---|---|---|---|---|")

        for rec in records:
            highlights_str = f"{len(rec.get('highlights', []))} clips"
            guests = ", ".join(rec.get("guests_speakers", [])) or "-"
            lines.append(f"| `{rec.get('filename')}` | **{rec.get('category')}** | {rec.get('event_edition', '-')} | {guests} | {highlights_str} |")

        lines.append("\n## ⏱️ Detailed Highlight Coordinates\n")
        for rec in records:
            if rec.get("highlights"):
                lines.append(f"### 🎥 `{rec.get('filename')}` ({rec.get('category')})")
                for h in rec.get("highlights"):
                    lines.append(f"- `[{h.get('start_fmt')} - {h.get('end_fmt')}]` {h.get('text')} *(Reason: {h.get('reason')})*")
                lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return md_path

    def _format_highlights_summary(self, highlights: List[Dict[str, Any]]) -> str:
        if not highlights:
            return ""
        return " | ".join(f"[{h.get('start_fmt')}-{h.get('end_fmt')}] {h.get('text')[:40]}..." for h in highlights[:5])
