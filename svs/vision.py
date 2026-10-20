"""Frame classification via a local Ollama vision model (default: moondream).

moondream (and similarly-sized small vision models) caption images well but do
not reliably follow strict "reply with only one label from this list" zero-shot
classification instructions - in testing it frequently returned an empty
response for that prompt style. Instead we ask for a short free-form caption
(which it handles well) and map keywords in that caption to our categories in
Python.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import requests

logger = logging.getLogger("smart_video_sorter")

CATEGORIES = [
    "Office",
    "Interior",
    "Outdoor",
    "Screen_Recording",
    "Events",
    "General",
]

_CAPTION_PROMPT = "Describe this image in one short sentence, mentioning the setting and any people or screens visible."

# Checked in this order; first match wins. Keep specific/unambiguous keywords
# earlier (screen/office) since generic words like "room" or "people" are easy
# to false-positive on.
_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Screen_Recording", [
        "screen recording", "computer screen", "monitor", "desktop", "user interface",
        "screenshot", "application window", "code editor", "web browser",
        "presentation slide", "powerpoint", "projector screen", "slideshow",
    ]),
    ("Events", [
        "crowd", "party", "wedding", "celebration", "audience", "stage",
        "concert", "gathering", "guests", "dancing", "ceremony", "festival",
        "conference", "networking", "panel discussion", "interview",
        "microphone", "podium", "seminar", "speaker", "roundtable",
        "business event", "award", "gala", "trophy", "applause", "clapping",
        "sitting at a table", "group of people", "people talking",
        "attendees", "business club", "formal event", "projector screen",
        "presentation", "guests seated", "at an event",
    ]),
    ("Office", [
        "office", "desk", "cubicle", "conference room", "meeting room",
        "workstation", "keyboard",
    ]),
    ("Outdoor", [
        "outdoor", "outside", "park", "tree", "sky", "street", "road",
        "garden", "forest", "mountain", "beach", "nature", "grass", "sunset",
        "field", "path", "bench",
    ]),
    ("Interior", [
        "indoor", "living room", "kitchen", "hallway", "interior", "bedroom",
        "ceiling", "furniture", "room with",
    ]),
]


class VisionEngineError(RuntimeError):
    pass


class VisionEngine:
    def __init__(self, model: str = "moondream", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def check_available(self) -> None:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise VisionEngineError(
                f"Ollama server not reachable at {self.host}. Is it running? ({exc})"
            ) from exc

        names = [m.get("name", "").split(":")[0] for m in resp.json().get("models", [])]
        if self.model not in names:
            raise VisionEngineError(
                f"Model '{self.model}' not found in Ollama. Run: ollama pull {self.model}"
            )

    def caption(self, frame_path: Path) -> str | None:
        b64 = base64.b64encode(frame_path.read_bytes()).decode()
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": _CAPTION_PROMPT, "images": [b64], "stream": False},
                timeout=120,  # first call also loads the model into VRAM
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Vision request failed for %s: %s", frame_path.name, exc)
            return None

        return resp.json().get("response", "").strip()

    @staticmethod
    def _clean_caption(caption: str, max_len: int = 220) -> str:
        """First line only; '' for captions too short to be useful or degenerate
        loops ('reserve/reserve/reserve...') that small models emit on text-heavy frames."""
        line = caption.strip().splitlines()[0].strip() if caption.strip() else ""
        words = line.replace("/", " ").split()
        if len(line) <= 15 or not any(ch.isalpha() for ch in line):
            return ""
        if len(words) >= 6 and len(set(words)) / len(words) < 0.5:
            return ""
        return line[:max_len]

    @staticmethod
    def _categorize_caption(caption: str) -> str | None:
        """Returns the matched category, or None if the caption doesn't match
        any specific keyword (garbage/blurry-frame captions must NOT count as
        votes for B-Roll_General, or they drown out real signal from the good
        frames of the same video)."""
        caption_lower = caption.lower()
        for category, keywords in _CATEGORY_KEYWORDS:
            if any(kw in caption_lower for kw in keywords):
                return category
        return None

    def classify_frames(self, frame_paths: list[Path]) -> tuple[str, str]:
        """Returns (category, description). description is the most detailed
        usable caption seen across frames (short/garbage captions from blurry
        or dark frames are filtered out); empty string if none qualified."""
        votes: dict[str, int] = {}
        captions: list[str] = []
        for fp in frame_paths:
            caption = self.caption(fp)
            if not caption:
                continue
            captions.append(caption)
            category = self._categorize_caption(caption)
            if category:
                votes[category] = votes.get(category, 0) + 1

        category = max(votes, key=votes.get) if votes else "General"

        usable = [c for c in (self._clean_caption(c) for c in captions) if c]
        description = max(usable, key=len) if usable else ""

        return category, description
