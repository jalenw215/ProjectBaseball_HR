from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv


async def main() -> None:
    import discord

    load_dotenv()
    token = os.environ["DISCORD_BOT_TOKEN"]
    channel_id = int(os.environ["DISCORD_CHANNEL_ID"])

    client = discord.Client(intents=discord.Intents.default())

    @client.event
    async def on_ready() -> None:
        print(f"bot_user={client.user} id={client.user.id}")
        guilds = ", ".join(f"{guild.name} ({guild.id})" for guild in client.guilds)
        print(f"visible_guilds={guilds or 'none'}")
        print(f"target_channel_id={channel_id}")
        channel = client.get_channel(channel_id)
        print(f"cached_channel={channel} type={type(channel).__name__ if channel else None}")
        try:
            fetched = await client.fetch_channel(channel_id)
        except Exception as exc:
            print(f"fetch_channel_error={type(exc).__name__}: {exc}")
            fetched = None
        print(f"fetched_channel={fetched} type={type(fetched).__name__ if fetched else None}")
        if channel is not None:
            guild = getattr(channel, "guild", None)
            print(f"channel_name={getattr(channel, 'name', None)}")
            print(f"guild_name={getattr(guild, 'name', None)}")
        elif fetched is not None:
            guild = getattr(fetched, "guild", None)
            print(f"channel_name={getattr(fetched, 'name', None)}")
            print(f"guild_name={getattr(guild, 'name', None)}")
        await client.close()

    await client.start(token)


if __name__ == "__main__":
    asyncio.run(main())
