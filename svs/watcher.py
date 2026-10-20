"""File-drop job runner: watch a task.json, run the job, write response.json.

Lets another tool (a script, an n8n flow, an AI agent) hand work to the
pipeline without importing it. Supported actions:

  {"action": "catalog", "source": "...", "target": "..."}   OCR flyers + transcribe videos -> catalog.csv/.json/.md
  {"action": "highlights", "target": "..."}                 score highlights for sorted talking heads
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .exporters import CatalogExporter
from .ffmpeg_utils import VideoSource, extract_audio, is_video_file
from .highlights import find_highlights
from .ocr import OCREngine
from .transcribe import WhisperEngine

logger = logging.getLogger("smart_video_sorter")


def catalog_folder(source: Path, target: Path, whisper: WhisperEngine) -> dict:
    ocr, exporter = OCREngine(), CatalogExporter(target)
    temp = target / "_temp"
    temp.mkdir(parents=True, exist_ok=True)
    records = []
    for p in sorted(x for x in source.rglob("*") if x.is_file()):
        base = {"filename": p.name, "event_edition": "", "date_extracted": "", "source_path": str(p)}
        if ocr.is_image_file(p):
            res = ocr.process_image(p)
            records.append({**base, "file_type": "Image/Flyer", "category": "Flyer_OCR",
                            "guests_speakers": res.detected_names, "highlights": [], "transcript_snippet": res.extracted_text})
        elif is_video_file(p):
            wav, src = temp / f"{p.stem}.wav", VideoSource(p, temp)
            try:
                extract_audio(src, wav)
                t = whisper.transcribe(wav)
                records.append({**base, "file_type": "Video", "category": "Talking_Head" if t.has_speech else "B-Roll",
                                "guests_speakers": [], "highlights": t.highlights, "transcript_snippet": t.text})
            except Exception as exc:
                logger.error("Audio failed for %s: %s", p.name, exc)
                records.append({**base, "file_type": "Video", "category": "B-Roll", "guests_speakers": [],
                                "highlights": [], "transcript_snippet": ""})
            finally:
                src.cleanup()
                wav.unlink(missing_ok=True)
    return {
        "status": "completed",
        "processed_files": len(records),
        "csv_catalog": str(exporter.export_csv(records)),
        "json_catalog": str(exporter.export_json(records)),
        "md_catalog": str(exporter.export_markdown(records)),
    }


def watch(task_file: Path, response_file: Path, whisper_model: str = "small", poll_seconds: float = 1.0) -> None:
    logger.info("Watching %s ...", task_file)
    last_mtime = 0.0
    whisper = None
    while True:
        try:
            if task_file.exists() and task_file.stat().st_mtime > last_mtime:
                last_mtime = task_file.stat().st_mtime
                task = json.loads(task_file.read_text(encoding="utf-8"))
                whisper = whisper or WhisperEngine(model_size=task.get("whisper_model", whisper_model))
                if task.get("action") == "highlights":
                    result = find_highlights(Path(task["target"]), whisper, task.get("batch_limit"))
                else:
                    result = catalog_folder(Path(task["source"]), Path(task["target"]), whisper)
                response_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("Task done -> %s", response_file)
        except Exception as exc:
            logger.error("Task failed: %s", exc, exc_info=True)
            response_file.write_text(json.dumps({"status": "error", "error": str(exc)}), encoding="utf-8")
        time.sleep(poll_seconds)
