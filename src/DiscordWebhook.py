"""Minimal helper to send Discord notifications during training."""

import logging
import os
from typing import Iterable, Optional

from discord_webhook import DiscordEmbed, DiscordWebhook as DiscordWebhookClient

# Replace with your Discord webhook URL.
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1440008655050244136/rtfkyNVLbrJIumRQXKUGUlUIRRjhhQ9VfCdEQXap5JC9FExJCBmJHiwp6FCrVmofLpF0"
# Set log levels forwarded to the webhook.
DISCORD_LOG_LEVELS = {"WARNING", "ERROR", "CRITICAL"}
# Optionally set to a list/set of logger names to forward. Use None to allow all.
DISCORD_LOG_LOGGERS = None


class DiscordWebhook(logging.Handler):
    """Send Discord webhook notifications and act as a logging handler."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        levels: Optional[Iterable[str]] = None,
        logger_names: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self.webhook_url = (webhook_url or DISCORD_WEBHOOK_URL).strip()
        self.levels = {level.upper() for level in (levels or DISCORD_LOG_LEVELS)}
        self.logger_names = set(logger_names) if logger_names else DISCORD_LOG_LOGGERS

    def emit(self, record: logging.LogRecord) -> None:
        if self.logger_names and record.name not in self.logger_names:
            return
        if record.levelname not in self.levels:
            return
        try:
            message = self.format(record)
            title = f"{record.levelname} in {record.name}"
            self._send(title, message)
        except Exception:
            pass

    def _send(self, title: str, description: str, file_path: Optional[str] = None) -> None:
        if not self.webhook_url:
            return

        webhook = DiscordWebhookClient(url=self.webhook_url, username="QB-PIER")
        embed = DiscordEmbed(title=title, description=description)
        webhook.add_embed(embed)

        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as file_data:
                webhook.add_file(file=file_data.read(), filename=os.path.basename(file_path))

        try:
            webhook.execute()
        except Exception:
            # The logger should never break training, so ignore webhook errors.
            pass

    def log_exp_start(self, model_name: str, seeds, log_dir: str) -> None:
        title = f"Starting training for `{model_name}`"
        message = f"Seeds: {seeds}\nConfiguration: {log_dir}"
        self._send(title, message)

    def log_exp_failed(self, model_name: str, seed: int, error: str) -> None:
        title = f"Training failed for `{model_name}` (seed {seed})"
        message = f"Error: ```{error}```"
        self._send(title, message)

    def log_exp_finished(self, model_name: str, log_dir: str, trained, expected, duration: float, figure_path: str) -> None:
        title = f"Training completed for `{model_name}`"
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        message = f"Directory: `{log_dir}`\nTrained {trained}/{expected} seeds\nDuration: {hours}h {minutes}m"
        self._send(title, message, figure_path)

    def log_attack_failed(self, attack_name: str, run_dir: str, error: str) -> None:
        title = f"Attack failed for `{attack_name}`"
        message = f"Run directory: `{run_dir}`\nError: ```{error}```"
        self._send(title, message)

    def log_transfer_attack_finished(self, attack_name: str, source_model: str, seed: int) -> None:
        title = f"Transfer attack completed for `{source_model}`"
        message = f"Attack: `{attack_name}`\nSeed: {seed}"
        self._send(title, message)
