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

    def log_exp_start(self, model_name: str, seeds) -> None:
        message = f"Starting training for `{model_name}`. Seeds: {', '.join(map(str, seeds))}"
        self._send("Experiment started", message)

    def log_exp_failed(self, model_name: str, seed: int, error: str) -> None:
        message = f"Model `{model_name}` failed for seed `{seed}`: {error}"
        self._send("Model failed", message)

    def log_training_curves(self, model_name: str, figure_path: str) -> None:
        message = f"Training curves for `{model_name}` are attached."
        self._send("Training curves", message, figure_path)

    def log_exp_finished(self, model_name: str, trained: int, expected: int, log_dir: str, figure_path: str) -> None:
        message = (
            f"Finished experiment for `{model_name}`.\n"
            f"Logs: `{log_dir}`"
        )
        self._send("Experiment completed", message, figure_path)
