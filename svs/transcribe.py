"""Speech detection, transcription and highlight scoring via faster-whisper."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("smart_video_sorter")

MIN_LANGUAGE_CONFIDENCE = 0.6
MIN_WORD_COUNT = 15

# Words that make a sentence worth clipping. Override with --keywords <file>.
DEFAULT_KEYWORDS = frozenset({
    "business", "afaceri", "profit", "marketing", "client", "clienti", "strategie", "strategy",
    "succes", "success", "rezultat", "result", "investitie", "investment", "partener", "partner",
    "vanzari", "sales", "oportunitate", "opportunity", "viitor", "future", "proiect", "project",
    "bani", "money", "growth", "crestere",
})


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float

    @staticmethod
    def fmt(t: float) -> str:
        m, s = divmod(int(t), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class Transcript:
    has_speech: bool
    language: str
    language_probability: float
    text: str
    word_count: int
    segments: list[Segment] = field(default_factory=list)
    highlights: list[dict] = field(default_factory=list)


def score_highlights(segments: list[Segment], keywords=DEFAULT_KEYWORDS, top: int = 10) -> list[dict]:
    """Picks clip-worthy moments: questions and exclamations (+2 each) and keyword
    hits (+3 each), keeping only 3-30 s segments that score at least 2."""
    picked = []
    for seg in segments:
        lower = seg.text.lower()
        is_question, is_exclaim = seg.text.endswith("?"), seg.text.endswith("!")
        matched = sorted(kw for kw in keywords if kw in lower)
        score = 2 * is_question + 2 * is_exclaim + 3 * len(matched)
        if 3.0 <= seg.end - seg.start <= 30.0 and score >= 2:
            picked.append({
                "start": seg.start,
                "end": seg.end,
                "start_fmt": Segment.fmt(seg.start),
                "end_fmt": Segment.fmt(seg.end),
                "text": seg.text,
                "score": score,
                "reason": "Question" if is_question else ("Keywords: " + ", ".join(matched) if matched else "Exclamation"),
            })
    picked.sort(key=lambda h: h["score"], reverse=True)
    return picked[:top]


def mini_subject(text: str, max_len: int = 40) -> str:
    """Slug of the first sentence, stored in the catalog as a quick label."""
    if not text:
        return ""
    first = text
    for sep in (".", "!", "?"):
        idx = first.find(sep)
        if idx != -1:
            first = first[:idx]
            break
    return "_".join(first.strip().split())[:max_len]


class WhisperEngine:
    def __init__(self, model_size: str = "small", device: str = "auto", keywords=DEFAULT_KEYWORDS):
        from faster_whisper import WhisperModel  # heavy import, only when actually transcribing

        self.model = WhisperModel(model_size, device=device, compute_type="default")
        self.keywords = keywords

    def transcribe(self, wav_path: Path) -> Transcript:
        raw, info = self.model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
        segments = [
            Segment(round(s.start, 2), round(s.end, 2), s.text.strip(), round(s.avg_logprob, 3))
            for s in raw if s.text.strip()
        ]
        text = " ".join(s.text for s in segments)
        words = len(text.split())
        # A clip "has speech" only if Whisper is confident about the language AND
        # heard enough words: this rejects music, crowd noise and mic bumps.
        has_speech = info.language_probability >= MIN_LANGUAGE_CONFIDENCE and words >= MIN_WORD_COUNT
        return Transcript(
            has_speech=has_speech,
            language=info.language,
            language_probability=info.language_probability,
            text=text,
            word_count=words,
            segments=segments,
            highlights=score_highlights(segments, self.keywords) if has_speech else [],
        )


def load_keywords(path: Path | None):
    if path is None:
        return DEFAULT_KEYWORDS
    return frozenset(w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip())
