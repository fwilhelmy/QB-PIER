"""Optional Discord experiment notifications; disabled unless configured."""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

DISCORD_LOG_LEVELS = {"WARNING", "ERROR", "CRITICAL"}

class DiscordWebhook(logging.Handler):
    """Send optional experiment notifications without affecting training."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        levels: Optional[Iterable[str]] = None,
        logger_names: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.webhook_url = (
            os.environ.get("DISCORD_WEBHOOK_URL", "") if webhook_url is None else webhook_url
        ).strip()
        self.levels = {level.upper() for level in (levels or DISCORD_LOG_LEVELS)}
        self.logger_names = set(logger_names) if logger_names else None

    def emit(self, record: logging.LogRecord) -> None:
        if not self.webhook_url or record.levelname not in self.levels:
            return
        if self.logger_names and record.name not in self.logger_names:
            return
        try:
            self._send(f"{record.levelname} in {record.name}", self.format(record))
        except Exception:
            # Notifications must never interrupt an experiment.
            self.handleError(record)

    def _send(self, title: str, description: str, file_path: Optional[str] = None) -> None:
        if not self.webhook_url:
            return
        try:
            from discord_webhook import DiscordEmbed, DiscordWebhook as DiscordWebhookClient
            webhook = DiscordWebhookClient(url=self.webhook_url, username="QB-PIER")
            webhook.add_embed(DiscordEmbed(title=title, description=description))
            if file_path and os.path.isfile(file_path):
                with open(file_path, "rb") as file_data:
                    webhook.add_file(file=file_data.read(), filename=os.path.basename(file_path))
            webhook.execute()
        except Exception:
            # Optional notification integrations must not break training.
            # Never include the URL in exception messages or log output.
            pass

    def log_exp_start(self, model_name: str, seeds, log_dir: str) -> None:
        self._send(f"Starting training for `{model_name}`", f"Seeds: {seeds}\nConfiguration: {log_dir}")

    def log_exp_failed(self, model_name: str, seed: int, error: str) -> None:
        self._send(f"Training failed for `{model_name}` (seed {seed})", f"Error: ```{error}```")

    def log_exp_finished(self, model_name: str, log_dir: str, trained, expected, duration: float, figure_path: str) -> None:
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        message = f"Directory: `{log_dir}`\nTrained {trained}/{expected} seeds\nDuration: {hours}h {minutes}m"
        self._send(f"Training completed for `{model_name}`", message, figure_path)

    def log_attack_failed(self, attack_name: str, run_dir: str, error: str) -> None:
        self._send(f"Attack failed for `{attack_name}`", f"Run directory: `{run_dir}`\nError: ```{error}```")

    def log_transfer_attack_finished(self, attack_name: str, source_model: str, seed: int) -> None:
        self._send(f"Transfer attack completed for `{source_model}`", f"Attack: `{attack_name}`\nSeed: {seed}")
