"""
Rich Text UI for Loading & Training Models (Dummy Simulation)

Requirements:
    pip install rich

Run:
    python training_ui.py

This script defines a single independent class `ModelTrainerUI` that
creates a terminal UI using the `rich` library with three sections:
  - Top-left: model name, configuration, and info
  - Top-right: logs window (auto-truncates old lines on overflow)
  - Bottom: progress area with one progress bar for epochs and two
            nested bars for training and validation steps that persist
            until each epoch ends.
"""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from rich.padding import Padding
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
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


@dataclass
class UIConfig:
    max_log_lines: int = 200
    epoch_min_delay_s: float = 0.2
    epoch_max_delay_s: float = 0.4
    step_min_delay_s: float = 0.05
    step_max_delay_s: float = 0.12
    val_min_delay_s: float = 0.04
    val_max_delay_s: float = 0.1


@dataclass
class ModelInfo:
    name: str = "DummyNet-XL"
    version: str = "0.1.0"
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


class ModelTrainerUI:
    def __init__(self, model_info: Optional[ModelInfo] = None, ui_config: Optional[UIConfig] = None):
        self.console = Console()
        self.model_info = model_info or ModelInfo()
        self.cfg = ui_config or UIConfig()

        self._logs: deque[str] = deque(maxlen=self.cfg.max_log_lines)

        self.epoch_progress = self._make_epoch_progress("[bold]Epochs")
        self.train_progress = self._make_task_progress("Training")
        self.val_progress = self._make_task_progress("Validation")

        self._epoch_task: Optional[TaskID] = None
        self._train_task: Optional[TaskID] = None
        self._val_task: Optional[TaskID] = None

        self.layout = self._build_layout()

    def simulate_training(self, epochs: int = 5, train_steps: int = 100, val_steps: int = 30) -> None:
        """Run a dummy training session to exercise the UI.
        Keeps train/val bars visible until the epoch ends, then clears them.
        """
        self._epoch_task = self.epoch_progress.add_task("Epoch", total=epochs)

        with Live(self.layout, console=self.console, refresh_per_second=30, screen=True):
            self._log(f"Starting training: epochs={epochs}, train_steps={train_steps}, val_steps={val_steps}")

            for epoch in range(1, epochs + 1):
                self._start_epoch(epoch, epochs)

                self._train_task = self.train_progress.add_task(f"Training", total=train_steps)
                self._val_task = self.val_progress.add_task(f"Validation", total=val_steps)

                for step in range(1, train_steps + 1):
                    self._sleep_jitter(self.cfg.step_min_delay_s, self.cfg.step_max_delay_s)
                    self.train_progress.advance(self._train_task, 1)
                    if step % max(1, train_steps // 5) == 0 or step == train_steps:
                        self._log(f"[train] epoch={epoch} step={step}/{train_steps}")

                for vstep in range(1, val_steps + 1):
                    self._sleep_jitter(self.cfg.val_min_delay_s, self.cfg.val_max_delay_s)
                    self.val_progress.advance(self._val_task, 1)
                    if vstep % max(1, val_steps // 3) == 0 or vstep == val_steps:
                        self._log(f"[val] epoch={epoch} step={vstep}/{val_steps}")

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

    def _log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._logs.append(f"[{ts}] {message}")
        self.layout["logs"].update(self._render_logs())

    def _make_epoch_progress(self, title: str) -> Progress:
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

    def _make_task_progress(self, title: str) -> Progress:
        return Progress(
            TextColumn("{task.description}", justify="left"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )

    def _build_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split(
            Layout(name="top", ratio=2),
            Layout(name="bottom", ratio=1),
        )
        layout["top"].split_row(
            Layout(name="model_info", ratio=1, minimum_size=40),
            Layout(name="logs", ratio=1, minimum_size=60),
        )
        layout["model_info"].update(self._render_model_info())
        layout["logs"].update(self._render_logs())
        layout["bottom"].update(self._render_progress_area())
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

    def _render_logs(self):
        log_text = Text("\n").join(Text(line) for line in self._logs)
        return Panel(
            log_text if len(self._logs) else Text("(logs will appear here)", style="dim"),
            title="Console", border_style="magenta", padding=(1, 1),
        )

    def _render_progress_area(self):
        # Indent training and validation progress bars slightly
        indented_group = Group(
            Padding(self.train_progress, (0, 0, 0, 4)),  # (top, right, bottom, left)
            Padding(self.val_progress, (0, 0, 0, 4)),
        )

        group = Group(
            self.epoch_progress,
            indented_group,
        )

        return Panel(group, title="Training", border_style="green", padding=(1, 1))

    def _start_epoch(self, epoch: int, total_epochs: int) -> None:
        self._log(f"Starting epoch {epoch}/{total_epochs}")
        self.layout["model_info"].update(self._render_model_info())
        self.layout["bottom"].update(self._render_progress_area())

    def _sleep_jitter(self, min_s: float, max_s: float) -> None:
        time.sleep(random.uniform(min_s, max_s))


if __name__ == "__main__":
    info = ModelInfo()
    ui = ModelTrainerUI(model_info=info, ui_config=UIConfig(max_log_lines=300))
    ui.simulate_training(epochs=5, train_steps=100, val_steps=40)