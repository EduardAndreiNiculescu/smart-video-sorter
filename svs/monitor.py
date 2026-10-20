"""Live terminal dashboard for long runs: catalog progress, shorts cut, GPU load."""

from __future__ import annotations

import sqlite3
import subprocess
import time
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table

from .broll import EVENT_SHOTS


def gpu_stats() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2).stdout.strip().splitlines()[0]
        name, util, used, total = [x.strip() for x in out.split(",")]
        return f"{name}  util {util}%  vram {used}/{total} MB"
    except Exception:
        return "no NVIDIA GPU detected"


def snapshot(target: Path) -> dict:
    db = target / "video_catalog.db"
    data = {"total": 0, "by_category": {}, "hl_done": 0, "hl_total": 0, "event_broll": 0, "broll": 0, "errors": 0}
    if db.exists():
        conn = sqlite3.connect(str(db))
        try:
            for cat, n in conn.execute("SELECT COALESCE(category, 'ERROR'), COUNT(*) FROM videos GROUP BY 1"):
                data["by_category"][cat] = n
                data["total"] += n
            data["hl_total"], data["hl_done"] = conn.execute(
                "SELECT COUNT(*), SUM(highlights_json IS NOT NULL AND highlights_json != '') "
                "FROM videos WHERE category = 'TALKING_HEAD'").fetchone()
            data["broll"] = sum(n for c, n in data["by_category"].items() if c.startswith("B-ROLL:"))
            data["event_broll"] = sum(n for c, n in data["by_category"].items() if c in EVENT_SHOTS)
            data["errors"] = conn.execute("SELECT COUNT(*) FROM videos WHERE status = 'error'").fetchone()[0]
        finally:
            conn.close()
    data["shorts"] = sum(1 for _ in target.glob("*/Shorts/*")) if target.exists() else 0
    data["hl_done"] = data["hl_done"] or 0
    return data


def render(target: Path, started: float) -> Panel:
    s = snapshot(target)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_column(justify="right")
    table.add_row("Highlights scored", ProgressBar(total=max(s["hl_total"], 1), completed=s["hl_done"], width=36),
                  f"{s['hl_done']}/{s['hl_total']}")
    table.add_row("Event B-roll typed", ProgressBar(total=max(s["broll"], 1), completed=s["event_broll"], width=36),
                  f"{s['event_broll']}/{s['broll']}")
    table.add_row("Shorts cut", "", str(s["shorts"]))
    table.add_row("Catalogued files", "", str(s["total"]))
    table.add_row("Errors", "", f"[red]{s['errors']}[/red]" if s["errors"] else "0")

    cats = Table(title="By category", show_edge=False, expand=False)
    cats.add_column("Category")
    cats.add_column("Files", justify="right")
    for cat, n in sorted(s["by_category"].items(), key=lambda kv: -kv[1]):
        cats.add_row(cat, str(n))

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - started))
    return Panel(Group(table, "", cats, "", f"[magenta]{gpu_stats()}[/magenta]"),
                 title="Smart Video Sorter — live", subtitle=f"{target}  ·  {elapsed}")


def run_monitor(target: Path, refresh: float = 1.5) -> None:
    started = time.time()
    with Live(render(target, started), refresh_per_second=4, screen=False) as live:
        while True:
            time.sleep(refresh)
            live.update(render(target, started))
