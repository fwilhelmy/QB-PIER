"""Minimal helper to send Discord notifications during training."""

import os
from typing import Optional

from discord_webhook import DiscordEmbed, DiscordWebhook

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1440008655050244136/rtfkyNVLbrJIumRQXKUGUlUIRRjhhQ9VfCdEQXap5JC9FExJCBmJHiwp6FCrVmofLpF0"  # Replace with your Discord webhook URL.


class DiscordWebhookLogger:
    """Very small wrapper around :mod:`discord_webhook`."""

    def __init__(self, webhook_url: Optional[str] = None) -> None:
        self.webhook_url = (webhook_url or DISCORD_WEBHOOK_URL).strip()

    def _send(self, title: str, description: str, file_path: Optional[str] = None) -> None:
        if not self.webhook_url:
            return

        webhook = DiscordWebhook(url=self.webhook_url, username="QB-PIER")
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
