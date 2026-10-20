"""Turn a horizontal clip into a vertical 9:16 short with karaoke captions.

1. Whisper word timestamps (the spoken word is highlighted as it is said)
2. Face-aware reframe: Haar-cascade face detection on sampled frames, crop
   centred on the median face position (falls back to centre crop)
3. Light grade (contrast/saturation), optional whoosh SFX on the first second,
   fade-out, all in a single ffmpeg filter graph
"""

from __future__ import annotations

import subprocess
from pathlib import Path

OUT_W, OUT_H = 1080, 1920
WHITE, HIGHLIGHT = "&HFFFFFF&", "&H00FF00&"  # ASS colours are BGR: &H00FF00& is green
MAX_CHUNK_WORDS = 5


def correct_word(word: str, corrections: dict[str, str]) -> str:
    """Fixes words Whisper predictably mis-hears (brand names, places), keeping punctuation."""
    stripped = word.strip(".,!?;:")
    trailing = word[len(stripped):] if stripped else ""
    fixed = corrections.get(stripped.lower())
    return fixed + trailing if fixed else word


def transcribe_words(wav: Path, model: str, language: str | None, corrections: dict[str, str]) -> list[dict]:
    from faster_whisper import WhisperModel

    segments, _ = WhisperModel(model, device="auto", compute_type="default").transcribe(
        str(wav), word_timestamps=True, language=language)
    return [
        {"word": correct_word(w.word.strip(), corrections), "start": w.start, "end": w.end}
        for seg in segments for w in seg.words
    ]


def face_crop_x(video: Path, samples: int = 8) -> tuple[int, int, int, int]:
    """Returns (crop_x, crop_w, src_w, src_h) for a 9:16 window centred on the
    median position of the largest face found across `samples` frames."""
    import cv2

    # Haar cascades ship with OpenCV 4.x; without them we fall back to a centre crop.
    cascade = None
    if hasattr(cv2, "CascadeClassifier"):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    centres = []
    for i in range(samples if cascade is not None else 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * (i + 0.5) / samples))
        ok, frame = cap.read()
        if not ok:
            continue
        faces = cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(60, 60))
        if len(faces):
            x, _, w, _ = max(faces, key=lambda f: f[2] * f[3])  # largest face = main speaker
            centres.append(x + w / 2)
    cap.release()

    crop_w = min(src_w, round(src_h * OUT_W / OUT_H))
    centre = sorted(centres)[len(centres) // 2] if centres else src_w / 2
    crop_x = max(0, min(int(centre - crop_w / 2), src_w - crop_w))
    return crop_x, crop_w, src_w, src_h


def chunk_words(words: list[dict], max_words: int = MAX_CHUNK_WORDS) -> list[list[dict]]:
    """Groups words into on-screen phrases: break on sentence end or every `max_words`."""
    chunks, current = [], []
    for w in words:
        current.append(w)
        if w["word"].rstrip().endswith((".", "!", "?")) or len(current) >= max_words:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def ass_time(t: float) -> str:
    return f"{int(t // 3600):d}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def build_ass(words: list[dict], font: str = "Arial Black", size: int = 70) -> str:
    """One Dialogue line per word: the whole phrase is shown, the current word in green."""
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {OUT_W}\nPlayResY: {OUT_H}\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{size},{WHITE},{WHITE},&H000000&,&H00000000&,1,0,0,0,100,100,0,0,1,4,2,2,60,60,320,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for chunk in chunk_words(words):
        for i, w in enumerate(chunk):
            text = " ".join(f"{{\\c{HIGHLIGHT if j == i else WHITE}}}{c['word']}" for j, c in enumerate(chunk))
            lines.append(f"Dialogue: 0,{ass_time(w['start'])},{ass_time(w['end'])},Default,,0,0,0,,{text}")
    return header + "\n".join(lines) + "\n"


def make_vertical_short(clip: Path, out: Path, model: str = "medium", language: str | None = None,
                        sfx: Path | None = None, corrections: dict[str, str] | None = None, log=print) -> Path:
    work = out.parent / f".{out.stem}_work"
    work.mkdir(parents=True, exist_ok=True)
    wav, ass = work / "audio.wav", work / "captions.ass"

    log("1/4 extracting audio")
    subprocess.run(["ffmpeg", "-y", "-i", str(clip), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                   capture_output=True, check=True)
    log(f"2/4 word-level transcription (whisper {model})")
    words = transcribe_words(wav, model, language, corrections or {})
    log(f"    {len(words)} words")
    log("3/4 finding the speaker for the vertical crop")
    crop_x, crop_w, src_w, src_h = face_crop_x(clip)
    ass.write_text(build_ass(words), encoding="utf-8")

    duration = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True).stdout.strip())
    fade = 0.4
    fade_at = max(0.0, duration - fade)
    ass_arg = str(ass).replace("\\", "/").replace(":", "\\:")
    vf = (f"crop={crop_w}:{src_h}:{crop_x}:0,scale={OUT_W}:{OUT_H},"
          f"eq=contrast=1.15:saturation=1.35:brightness=0.02,ass='{ass_arg}',fade=t=out:st={fade_at:.2f}:d={fade}")

    log("4/4 rendering: crop + grade + captions" + (" + sfx" if sfx else "") + " + fade")
    cmd = ["ffmpeg", "-y", "-i", str(clip)]
    if sfx:
        cmd += ["-i", str(sfx), "-filter_complex",
                f"[0:v]{vf}[v];[1:a]volume=0.12,atrim=0:1.2,afade=t=out:st=0.9:d=0.3[s];"
                f"[0:a][s]amix=inputs=2:duration=first:dropout_transition=0,afade=t=out:st={fade_at:.2f}:d={fade}[a]",
                "-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-vf", vf, "-af", f"afade=t=out:st={fade_at:.2f}:d={fade}"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "192k", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:])
    return out
