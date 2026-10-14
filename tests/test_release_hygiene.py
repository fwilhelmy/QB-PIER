"""Release guards that do not require the ML/quantum dependency stack."""
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_no_generated_artifacts_or_research_directories():
    for path in ("experiments", "src/analytics/attacks", "src/analytics/notebooks", "conf/experiment", "docs/HELPER.md"):
        assert not (ROOT / path).exists(), f"Internal research file must not be released: {path}"

def test_webhook_off_by_default(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    webhook = importlib.import_module("DiscordWebhook")
    client = webhook.DiscordWebhook()
    assert client.webhook_url == ""
    client.log_exp_start("model", [0], "demo")

def test_webhook_uses_environment(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    webhook = importlib.import_module("DiscordWebhook")
    assert webhook.DiscordWebhook().webhook_url == "https://example.invalid/webhook"
    assert webhook.DiscordWebhook(webhook_url="").webhook_url == ""


def test_release_scanner_detects_fake_secret_without_printing_it(tmp_path):
    import subprocess
    import sys

    generated_fake = "https://discord.com/api/" + "webhooks/123456789/test-token-value"
    (tmp_path / "config.py").write_text(f'url = "{generated_fake}"\n')
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_public_release.py"), str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "webhook URL" in result.stdout
    assert generated_fake not in result.stdout
