from discord_webhook import DiscordWebhook, DiscordEmbed
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1440008655050244136/rtfkyNVLbrJIumRQXKUGUlUIRRjhhQ9VfCdEQXap5JC9FExJCBmJHiwp6FCrVmofLpF0"
webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL)
with open("logs\\vis_atck_full\\convnet\\0.0\\old_sal_0.png", "rb") as f:
    webhook.add_file(file=f.read(), filename="example.jpg")
embed = DiscordEmbed(title='Testing', description="test")
webhook.add_embed(embed)
response = webhook.execute()