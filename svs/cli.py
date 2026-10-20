"""`svs` command line: sort, highlights, shorts, caption, broll-events, watch, monitor."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    # Legacy Windows codepages (cp1252) cannot print Rich's box/spinner glyphs.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Sort raw footage into talking heads and B-roll, then cut captioned shorts. Fully local.")
console = Console()
logger = logging.getLogger("smart_video_sorter")


def _logging(target: Path | None = None) -> None:
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    rich = RichHandler(console=console, show_time=False, show_path=False)
    rich.setLevel(logging.INFO)
    logger.addHandler(rich)
    if target:
        target.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(target / "error.log", encoding="utf-8")
        fh.setLevel(logging.WARNING)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)


def _vision(host: str, model: str):
    from .vision import VisionEngine, VisionEngineError

    engine = VisionEngine(model=model, host=host)
    try:
        engine.check_available()
    except VisionEngineError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    return engine


@app.command()
def sort(
    source: Path = typer.Option(..., "--source", "-s", help="Raw footage folder (local, NAS or a mapped WebDAV drive)"),
    target: Path = typer.Option(..., "--target", "-t", help="Where the sorted tree and catalog are written"),
    action: str = typer.Option("copy", help="copy | move"),
    device: str = typer.Option("auto", help="Whisper device: auto | cpu | cuda"),
    whisper_model: str = typer.Option("small", help="faster-whisper model size"),
    keywords: Optional[Path] = typer.Option(None, help="Text file, one highlight keyword per line"),
    frames_per_video: int = typer.Option(9, help="Frames sampled per B-roll clip for the vision vote"),
    short_clip_seconds: float = typer.Option(2.0, help="Clips shorter than this go to _Short_Raw unanalysed"),
    event_depth: int = typer.Option(1, help="Folder level under --source that names the event"),
    ollama_host: str = typer.Option("http://localhost:11434"),
    vision_model: str = typer.Option("moondream", help="Any Ollama vision model"),
    batch_size: Optional[int] = typer.Option(None, help="Process at most N new files, then stop"),
    dry_run: bool = typer.Option(False, help="Classify and log, but do not copy/move anything"),
    temp_dir: Optional[Path] = typer.Option(None, help="Scratch space for local copies, audio and frames"),
):
    """Classify every video: talking head (speech), B-roll (vision vote) or too-short fragment."""
    from .sorter import SortOptions, sort_videos
    from .transcribe import WhisperEngine, load_keywords

    if action not in ("copy", "move"):
        raise typer.BadParameter("action must be copy or move")
    _logging(target)
    console.print(f"[bold]Smart Video Sorter[/bold]  {source} → {target}  ({action}{', dry-run' if dry_run else ''})")
    vision = _vision(ollama_host, vision_model)
    console.print(f"Loading Whisper [cyan]{whisper_model}[/cyan] on [cyan]{device}[/cyan]...")
    whisper = WhisperEngine(whisper_model, device, load_keywords(keywords))

    stats = sort_videos(SortOptions(source, target, action, short_clip_seconds, frames_per_video,
                                    event_depth, batch_size, dry_run, temp_dir), whisper, vision, console)
    table = Table(title="Summary")
    table.add_column("Category")
    table.add_column("Files", justify="right")
    for name, n in stats.counts.items():
        table.add_row(name, str(n))
    console.print(table)
    console.print(f"Catalog: {target / 'catalog.csv'}   Errors: {target / 'error.log'}")


@app.command()
def highlights(
    target: Path = typer.Option(..., "--target", "-t"),
    whisper_model: str = typer.Option("small"),
    device: str = typer.Option("auto"),
    keywords: Optional[Path] = typer.Option(None),
    batch_size: Optional[int] = typer.Option(None),
):
    """Score clip-worthy moments (questions, exclamations, keywords) in sorted talking heads."""
    from .highlights import find_highlights
    from .transcribe import WhisperEngine, load_keywords

    _logging(target)
    result = find_highlights(target, WhisperEngine(whisper_model, device, load_keywords(keywords)), batch_size)
    console.print_json(json.dumps(result))


@app.command()
def shorts(
    target: Path = typer.Option(..., "--target", "-t"),
    accurate: bool = typer.Option(False, help="Re-encode for frame-exact cuts (slower than stream copy)"),
):
    """Cut every stored highlight into <event>/Shorts/ (15-60 s) + shorts_manifest.csv."""
    from .shorts import cut_shorts

    console.print_json(json.dumps(cut_shorts(target, accurate)))


@app.command()
def caption(
    clip: Path = typer.Argument(..., exists=True, help="Horizontal clip, ideally one short from `svs shorts`"),
    out: Path = typer.Option(None, "--out", "-o", help="Output .mp4 (default: <clip>_vertical.mp4)"),
    whisper_model: str = typer.Option("medium"),
    language: Optional[str] = typer.Option(None, help="Force a language code, e.g. ro, en"),
    sfx: Optional[Path] = typer.Option(None, exists=True, help="Optional whoosh/impact sound for the first second"),
    fix: list[str] = typer.Option([], help="Word correction, e.g. --fix crayova=Craiova (repeatable)"),
):
    """Render a 9:16 short: face-aware crop, karaoke word captions, grade, optional SFX."""
    from .captions import make_vertical_short

    corrections = dict(f.split("=", 1) for f in fix)
    out = out or clip.with_name(f"{clip.stem}_vertical.mp4")
    make_vertical_short(clip, out, whisper_model, language, sfx, {k.lower(): v for k, v in corrections.items()},
                        log=lambda m: console.print(f"[cyan]{m}[/cyan]"))
    console.print(f"[green]Done:[/green] {out}")


@app.command("broll-events")
def broll_events(
    target: Path = typer.Option(..., "--target", "-t"),
    ollama_host: str = typer.Option("http://localhost:11434"),
    vision_model: str = typer.Option("moondream"),
    batch_size: Optional[int] = typer.Option(None),
    reverse: bool = typer.Option(False, help="Drain the queue from the end (run a second worker in parallel)"),
):
    """Re-bucket B-roll from event footage into Stage / Audience / Networking / Awards / Decor."""
    from .broll import reclassify_event_broll

    _logging(target)
    console.print_json(json.dumps(reclassify_event_broll(target, _vision(ollama_host, vision_model), batch_size, reverse)))


@app.command()
def watch(
    task_file: Path = typer.Option(Path("task.json")),
    response_file: Path = typer.Option(Path("response.json")),
    whisper_model: str = typer.Option("small"),
):
    """Run jobs dropped into task.json and write results to response.json."""
    from .watcher import watch as run_watch

    _logging()
    run_watch(task_file, response_file, whisper_model)


@app.command()
def monitor(target: Path = typer.Option(..., "--target", "-t")):
    """Live dashboard for a running job (open it in a second terminal)."""
    from .monitor import run_monitor

    try:
        run_monitor(target)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app()
