"""OCR Engine for extracting text and guest metadata from images and flyers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("smart_video_sorter")

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

@dataclass
class OCRResult:
    image_path: str
    extracted_text: str
    word_count: int
    detected_names: List[str]
    has_text: bool


class OCREngine:
    def __init__(self):
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(['ro', 'en'], gpu=True)
            except Exception as e:
                logger.warning(f"EasyOCR not available, falling back to pytesseract/basic reader: {e}")
                self._reader = "fallback"
        return self._reader

    def is_image_file(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def process_image(self, image_path: Path) -> OCRResult:
        """Extract text from an image file."""
        if not image_path.exists():
            return OCRResult(str(image_path), "", 0, [], False)

        extracted_text = ""
        reader = self._get_reader()

        if reader != "fallback" and reader is not None:
            try:
                results = reader.readtext(str(image_path), detail=0)
                extracted_text = " ".join(results)
            except Exception as e:
                logger.error(f"Error during EasyOCR processing of {image_path}: {e}")
                extracted_text = ""

        if not extracted_text:
            # Fallback using pytesseract or PIL basic metadata
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(image_path)
                extracted_text = pytesseract.image_to_string(img)
            except Exception:
                extracted_text = ""

        words = extracted_text.split()
        names = self._extract_names(extracted_text)

        return OCRResult(
            image_path=str(image_path),
            extracted_text=extracted_text.strip(),
            word_count=len(words),
            detected_names=names,
            has_text=len(words) > 0,
        )

    def _extract_names(self, text: str) -> List[str]:
        """Simple heuristic for extracting capitalized full names from text."""
        import re
        # Find pairs of capitalized words (e.g. "Ion Popescu", "Elena Dumitrescu")
        pattern = r'\b[A-ZÎĂȘȚ][a-zîășț]+\s+[A-ZÎĂȘȚ][a-zîășț]+\b'
        matches = re.findall(pattern, text)
        # Deduplicate while preserving order
        seen = set()
        unique_names = []
        for name in matches:
            if name not in seen and len(name) > 5:
                seen.add(name)
                unique_names.append(name)
        return unique_names
