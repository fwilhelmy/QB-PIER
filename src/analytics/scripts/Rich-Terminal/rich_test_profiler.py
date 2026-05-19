"""
Rich Text UI for Loading, Training & Device Profiling (Dummy Simulation)

Requirements:
    pip install rich

Run:
    python training_ui.py

Layout (4 parts):
  - Top-left: Model Info
  - Top-right: Device / Threads Profiling (live, simulated)
  - Middle: Training Progress (epochs bar + indented train/val bars)
  - Bottom: Logs (auto-truncated)
"""
from __future__ import annotations

import datetime
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


# --------------------------- Config & Model Info ---------------------------
@dataclass
class UIConfig:
    max_log_lines: int = 200
    epoch_min_delay_s: float = 0.15
    epoch_max_delay_s: float = 0.30
    step_min_delay_s: float = 0.02
    step_max_delay_s: float = 0.05
    val_min_delay_s: float = 0.02
    val_max_delay_s: float = 0.05
    device_refresh_every_steps: int = 5  # how often to refresh device panel


@dataclass
class ModelInfo:
    name: str = "DummyNet-XL"
    version: str = "0.2.0"
    parameters: int = 12_345_678
    description: str = "Prototype architecture for UI demo"
    config: Dict[str, str] = field(default_factory=lambda: {
        "optimizer": "AdamW",
        "lr": "3e-4",
        "batch_size": "64",
        "epochs": "5",
        "dataset": "DummyDatasetV1",
        "seed": "1337",
        "device": "cuda:0 (simulated)",
    })


# ------------------------------ Device Profile -----------------------------
@dataclass
class Process:
    pid: int
    command: str
    cpu_percent: float
    memory: int
    start_time: datetime.datetime
    thread_count: int
    state: Literal["running", "sleeping"]

    @property
    def memory_str(self) -> str:
        if self.memory > 1e6:
            return f"{int(self.memory/1e6)}M"
        if self.memory > 1e3:
            return f"{int(self.memory/1e3)}K"
        return str(self.memory)

    @property
    def time_str(self) -> str:
        return str(datetime.datetime.now() - self.start_time)


def _gen_process(pid: int) -> Process:
    return Process(
        pid=pid,
        command=f"Process {pid}",
        cpu_percent=random.random() * 40,
        memory=random.randint(10, 200) ** 3,
        start_time=datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 500) ** 2),
        thread_count=random.randint(1, 64),
        state="running" if random.randint(0, 10) < 8 else "sleeping",
    )


def _device_table(rows: int = 10) -> Table:
    processes = sorted([_gen_process(pid) for pid in range(max(1, rows))], key=lambda p: p.cpu_percent, reverse=True)
    table = Table("PID", "Command", "CPU %", "Memory", "Time", "Threads", "State", box=box.SIMPLE)
    for p in processes:
        table.add_row(str(p.pid), p.command, f"{p.cpu_percent:.1f}", p.memory_str, p.time_str, str(p.thread_count), p.state)
    return table


# --------------------------------- The UI ----------------------------------
class ModelTrainerUI:
    def __init__(self, model_info: Optional[ModelInfo] = None, ui_config: Optional[UIConfig] = None):
        self.console = Console()
        self.model_info = model_info or ModelInfo()
        self.cfg = ui_config or UIConfig()

        self._logs: deque[str] = deque(maxlen=self.cfg.max_log_lines)

        # Progress bars
        self.epoch_progress = self._make_epoch_progress()
        self.train_progress = self._make_task_progress()
        self.val_progress = self._make_task_progress()

        # Task handles
        self._epoch_task: Optional[TaskID] = None
        self._train_task: Optional[TaskID] = None
        self._val_task: Optional[TaskID] = None

        # Layout
        self.layout = self._build_layout()

    # --------------------------- Public entrypoint --------------------------
    def simulate_training(self, epochs: int = 5, train_steps: int = 100, val_steps: int = 30) -> None:
        """Run a dummy training session with device profiling panel updating live."""
        self._epoch_task = self.epoch_progress.add_task("Epoch", total=epochs)

        with Live(self.layout, console=self.console, refresh_per_second=30, screen=True):
            self._log(f"Starting training: epochs={epochs}, train_steps={train_steps}, val_steps={val_steps}")

            for epoch in range(1, epochs + 1):
                self._start_epoch(epoch, epochs)

                # Create nested tasks and keep visible until epoch end
                self._train_task = self.train_progress.add_task("Training", total=train_steps)
                self._val_task = self.val_progress.add_task("Validation", total=val_steps)

                # ---- Train phase ----
                for step in range(1, train_steps + 1):
                    self._sleep_jitter(self.cfg.step_min_delay_s, self.cfg.step_max_delay_s)
                    self.train_progress.advance(self._train_task, 1)
                    if step % max(1, train_steps // 5) == 0 or step == train_steps:
                        self._log(f"[train] epoch={epoch} step={step}/{train_steps}")
                    if step % self.cfg.device_refresh_every_steps == 0:
                        self._refresh_device_panel()

                # ---- Validation phase ----
                for vstep in range(1, val_steps + 1):
                    self._sleep_jitter(self.cfg.val_min_delay_s, self.cfg.val_max_delay_s)
                    self.val_progress.advance(self._val_task, 1)
                    if vstep % max(1, val_steps // 3) == 0 or vstep == val_steps:
                        self._log(f"[val]   epoch={epoch} step={vstep}/{val_steps}")
                    if vstep % self.cfg.device_refresh_every_steps == 0:
                        self._refresh_device_panel()

                # Epoch complete
                self.epoch_progress.advance(self._epoch_task, 1)
                self._log(f"Epoch {epoch} completed.")

                # Clear nested bars only after epoch is finished
                if self._train_task is not None:
                    self.train_progress.remove_task(self._train_task)
                    self._train_task = None
                if self._val_task is not None:
                    self.val_progress.remove_task(self._val_task)
                    self._val_task = None

                self._sleep_jitter(self.cfg.epoch_min_delay_s, self.cfg.epoch_max_delay_s)

            self._log("Training complete.")

    # ------------------------------ Rendering ------------------------------
    def _make_epoch_progress(self) -> Progress:
        return Progress(
            TextColumn("{task.description}", justify="left"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}", justify="right"),
            TextColumn(" • "),
            TimeElapsedColumn(),
            TextColumn(" / "),
            TimeRemainingColumn(),
            expand=True,
        )

    def _make_task_progress(self) -> Progress:
        return Progress(
            TextColumn("{task.description}", justify="left"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )

    def _build_layout(self) -> Layout:
        layout = Layout(name="root")
        # 3 vertical bands: header(row with 2 columns), progress, logs
        layout.split(
            Layout(name="header", ratio=4),
            Layout(name="progress", ratio=2),
            Layout(name="logs", ratio=3),
        )
        # Header is split into model info (left) and device panel (right)
        layout["header"].split_row(
            Layout(name="model_info", ratio=1, minimum_size=40),
            Layout(name="device", ratio=3, minimum_size=60),
        )

        layout["model_info"].update(self._render_model_info())
        layout["device"].update(self._render_device_panel())
        layout["progress"].update(self._render_progress_area())
        layout["logs"].update(self._render_logs())
        return layout

    def _render_model_info(self):
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Model", f"{self.model_info.name} [dim]v{self.model_info.version}[/dim]")
        table.add_row("Parameters", f"{self.model_info.parameters:,}")
        table.add_row("Description", self.model_info.description)
        table.add_row("", Rule(style="dim"))
        for k, v in self.model_info.config.items():
            table.add_row(k, str(v))
        return Panel(table, title="Model Information", border_style="cyan", padding=(1, 1))

    def _render_device_panel(self):
        # Choose table height based on current terminal; keep it compact
        rows = max(5, min(20, self.console.size.height // 4))
        table = _device_table(rows)
        return Panel(table, title="Device / Threads Profiling (simulated)", border_style="blue", padding=(0, 1))

    def _render_progress_area(self):
        # Indent training and validation progress bars slightly and let them fill remaining width
        indented_group = Group(
            Padding(self.train_progress, (0, 0, 0, 4)),  # (top, right, bottom, left)
            Padding(self.val_progress, (0, 0, 0, 4)),
        )
        group = Group(self.epoch_progress, indented_group)
        return Panel(group, title="Training Progress", border_style="green", padding=(1, 1))

    def _render_logs(self):
        log_text = Text("\n").join(Text(line) for line in self._logs)
        return Panel(
            log_text if len(self._logs) else Text("(logs will appear here)", style="dim"),
            title="Console", border_style="magenta", padding=(1, 1),
        )

    # ------------------------------ Utilities ------------------------------
    def _log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._logs.append(f"[{ts}] {message}")
        self.layout["logs"].update(self._render_logs())

    def _refresh_device_panel(self) -> None:
        self.layout["device"].update(self._render_device_panel())

    def _start_epoch(self, epoch: int, total_epochs: int) -> None:
        self._log(f"Starting epoch {epoch}/{total_epochs}")
        self.layout["model_info"].update(self._render_model_info())
        self.layout["progress"].update(self._render_progress_area())
        self._refresh_device_panel()

    def _sleep_jitter(self, min_s: float, max_s: float) -> None:
        time.sleep(random.uniform(min_s, max_s))


# --------------------------------- Demo ------------------------------------
if __name__ == "__main__":
    info = ModelInfo()
    ui = ModelTrainerUI(model_info=info, ui_config=UIConfig(max_log_lines=300))
    ui.simulate_training(epochs=5, train_steps=100, val_steps=40)
