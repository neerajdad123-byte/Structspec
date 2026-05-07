from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.text import Text
    RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    RICH_AVAILABLE = False


@dataclass
class TokenEvent:
    text: str
    status: str  # accepted, rejected, bonus, verified
    tier: str = ""


@dataclass
class VizState:
    prompt_len: int = 0
    max_tokens: int = 0
    current_len: int = 0
    passes: int = 0
    accepted_draft: int = 0
    proposed: int = 0
    speed_tok_per_sec: float = 0.0
    speedup_vs_greedy: float = 0.0
    greedy_tok_per_sec: float = 0.0
    tokens_log: deque = field(default_factory=lambda: deque(maxlen=120))
    recent_events: deque = field(default_factory=lambda: deque(maxlen=8))


class RichVisualizer:
    """Rich-based live terminal dashboard for speculative decoding."""

    def __init__(self, max_tokens: int, greedy_reference_speed: Optional[float] = None):
        if not RICH_AVAILABLE:
            raise ImportError(
                "rich is required for --rich-viz. Install with: pip install rich"
            )
        self.max_tokens = max_tokens
        self.greedy_ref = greedy_reference_speed
        self.state = VizState(max_tokens=max_tokens)
        self.console = Console()
        self._start_time = time.perf_counter()
        self._live: Optional[Live] = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=15,
            screen=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)

    def update(
        self,
        current_len: int,
        passes: int,
        accepted_draft: int,
        proposed: int,
        token_events: list[TokenEvent],
    ) -> None:
        self.state.current_len = current_len
        self.state.passes = passes
        self.state.accepted_draft = accepted_draft
        self.state.proposed = proposed

        elapsed = time.perf_counter() - self._start_time
        self.state.speed_tok_per_sec = current_len / max(1e-9, elapsed)
        if self.greedy_ref and self.state.speed_tok_per_sec > 0:
            self.state.speedup_vs_greedy = self.state.speed_tok_per_sec / self.greedy_ref

        for ev in token_events:
            self.state.tokens_log.append(ev)
            self.state.recent_events.append(ev)

        if self._live:
            self._live.update(self._render())

    def _render(self):
        # Token stream — long scrollable history
        token_text = Text()
        for ev in self.state.tokens_log:
            style = {
                "accepted": "bold bright_green",
                "rejected": "bold bright_red",
                "bonus": "bold cyan",
                "verified": "dim white",
            }.get(ev.status, "white")
            piece = ev.text.replace("\n", "\\n").replace("\t", "\\t")
            token_text.append(piece, style=style)

        token_panel = Panel(
            token_text,
            title="[bold blue]Live Token Stream[/]",
            subtitle="[green]+ accepted[/green]  [red]X rejected[/red]  [cyan]* bonus[/cyan]  [white]. verified[/white]",
            border_style="blue",
        )

        # Progress bar
        progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(
                bar_width=None,
                complete_style="green",
                finished_style="bright_green",
            ),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )
        progress.add_task(
            description="Generating tokens",
            total=self.state.max_tokens,
            completed=min(self.state.current_len, self.state.max_tokens),
        )

        # Metrics panel
        metrics = Table(show_header=False, box=None, padding=(0, 1))
        metrics.add_column("Label", style="cyan", no_wrap=True)
        metrics.add_column("Value", style="bold white", no_wrap=True)

        metrics.add_row("Generated", f"{self.state.current_len} / {self.state.max_tokens}")
        metrics.add_row("Target Passes", str(self.state.passes))
        draft_acc = (
            self.state.accepted_draft / max(1, self.state.proposed)
        ) * 100
        metrics.add_row(
            "Draft Tokens",
            f"{self.state.accepted_draft} / {self.state.proposed}",
        )
        metrics.add_row("Draft Accuracy", f"{draft_acc:.1f}%")
        metrics.add_row("Throughput", f"{self.state.speed_tok_per_sec:.1f} tok/s")
        if self.greedy_ref:
            color = "green" if self.state.speedup_vs_greedy >= 1.2 else "yellow"
            metrics.add_row(
                "Speedup",
                f"[bold {color}]{self.state.speedup_vs_greedy:.2f}x[/] vs greedy",
            )

        metrics_panel = Panel(
            metrics,
            title="[bold blue]Metrics[/]",
            border_style="blue",
        )

        # Recent events panel
        events_table = Table(show_header=False, box=None, padding=(0, 1))
        events_table.add_column("Mark", width=3)
        events_table.add_column("Token", style="white")
        events_table.add_column("Tier", style="dim")

        for ev in reversed(list(self.state.recent_events)[-6:]):
            mark = {
                "accepted": "[green]+[/]",
                "rejected": "[red]X[/]",
                "bonus": "[cyan]*[/]",
                "verified": "[white].[/]",
            }.get(ev.status, "?")
            piece = ev.text.replace("\n", "\\n").replace("\t", "\\t")
            if len(piece) > 24:
                piece = piece[:24] + "…"
            tier_text = f"[dim]{ev.tier}[/]" if ev.tier else ""
            row_style = {
                "accepted": "green",
                "rejected": "red",
                "bonus": "cyan",
                "verified": "white",
            }.get(ev.status, "white")
            events_table.add_row(mark, piece, tier_text, style=row_style)

        events_panel = Panel(
            events_table,
            title="[bold blue]Recent Events[/]",
            border_style="blue",
        )

        # Combine bottom panels side-by-side
        bottom = Table(show_header=False, box=None, padding=0, expand=True)
        bottom.add_column(ratio=1)
        bottom.add_column(ratio=1)
        bottom.add_row(metrics_panel, events_panel)

        return Group(token_panel, progress, bottom)
