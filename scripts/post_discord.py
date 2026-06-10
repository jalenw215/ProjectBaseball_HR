from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from hr_predictor.config import DEFAULT_PREDICTIONS_FILE
from hr_predictor.reporting import format_daily_report


async def post_message(token: str, channel_id: int, message: str) -> None:
    import discord

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    error: Exception | None = None

    @client.event
    async def on_ready() -> None:
        nonlocal error
        try:
            channel = client.get_channel(channel_id) or await client.fetch_channel(channel_id)
            if channel is None:
                raise RuntimeError(f"Could not find Discord channel {channel_id}")
            await channel.send(message[:1900])
        except Exception as exc:
            error = exc
        finally:
            await client.close()

    await client.start(token)
    if error is not None:
        raise RuntimeError(f"Could not post to Discord channel {channel_id}: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Post HR predictions to Discord")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("DISCORD_BOT_TOKEN")
    channel_id = os.getenv("DISCORD_CHANNEL_ID")
    if not token or not channel_id:
        raise SystemExit("Set DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID in .env")
    message = format_daily_report(args.predictions, args.limit)
    asyncio.run(post_message(token, int(channel_id), message))


if __name__ == "__main__":
    main()
