"""
LLMChat — A Red-DiscordBot cog that responds to @mentions using the OpenAI-compatible API.

Uses the lightweight `openai` Python package, which works with:
  - Ollama (local)     base_url = http://localhost:11434/v1,  api_key = "ollama"
  - OpenAI             base_url = (default),                  api_key = sk-...
  - OpenRouter         base_url = https://openrouter.ai/api/v1, api_key = sk-or-...
  - Any OpenAI-compatible endpoint (LM Studio, vLLM, etc.)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import discord
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import box

try:
    from openai import AsyncOpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

log = logging.getLogger("red.tylerscogs.llmchat")

DEFAULT_CONTEXT_MESSAGES = 10
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MODEL = "llama3"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant in a Discord server. "
    "Be concise, friendly, and accurate. "
    "You are replying to a Discord message, so keep responses reasonably short unless detail is needed."
)
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"  # Ollama requires any non-empty string


class LLMChat(commands.Cog):
    """
    AI chat via an OpenAI-compatible API. Responds to @mentions in enabled channels.

    Works with Ollama (local), OpenAI, OpenRouter, LM Studio, vLLM, and more.
    """

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(
            self, identifier=0x4C4C4D434841, force_registration=True
        )

        self.config.register_guild(
            enabled_channels=[],
            model=DEFAULT_MODEL,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=DEFAULT_MAX_TOKENS,
            context_messages=DEFAULT_CONTEXT_MESSAGES,
            base_url=DEFAULT_BASE_URL,
            api_key=DEFAULT_API_KEY,
        )

    async def cog_load(self):
        if not _OPENAI_AVAILABLE:
            log.error(
                "LLMChat: the `openai` package is not installed. "
                "Run this in your bot's venv and then reload the cog:\n"
                "  /data/venv/bin/pip install openai"
            )

    # ------------------------------------------------------------------
    # Listener
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Only process @mentions in enabled channels."""
        if not _OPENAI_AVAILABLE:
            return
        if message.author.bot:
            return
        if not message.guild:
            return
        if self.bot.user not in message.mentions:
            return

        enabled_channels = await self.config.guild(message.guild).enabled_channels()
        if message.channel.id not in enabled_channels:
            return

        if not message.channel.permissions_for(message.guild.me).send_messages:
            return

        await self._handle_mention(message)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _handle_mention(self, message: discord.Message):
        guild_config = await self.config.guild(message.guild).all()

        user_text = message.content
        for mention in [f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"]:
            user_text = user_text.replace(mention, "").strip()

        if not user_text:
            user_text = "(no text — the user only pinged you)"

        messages = await self._build_messages(
            message,
            user_text,
            system_prompt=guild_config["system_prompt"],
            context_limit=guild_config["context_messages"],
        )

        async with message.channel.typing():
            try:
                reply = await asyncio.wait_for(
                    self._call_openai(
                        messages=messages,
                        model=guild_config["model"],
                        max_tokens=guild_config["max_tokens"],
                        base_url=guild_config["base_url"],
                        api_key=guild_config["api_key"],
                    ),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                await message.reply(
                    "⏱️ The model took too long to respond. Try again or check your backend."
                )
                return
            except Exception as exc:
                log.exception("LLM API call failed")
                await message.reply(f"❌ Error calling the model: {exc}")
                return

        if len(reply) <= 2000:
            await message.reply(reply)
        else:
            chunks = [reply[i : i + 1990] for i in range(0, len(reply), 1990)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk)
                else:
                    await message.channel.send(chunk)

    async def _build_messages(
        self,
        trigger_message: discord.Message,
        user_text: str,
        system_prompt: str,
        context_limit: int,
    ) -> list:
        messages = [{"role": "system", "content": system_prompt}]

        if context_limit > 0:
            history = []
            try:
                async for msg in trigger_message.channel.history(
                    limit=context_limit + 1,
                    before=trigger_message,
                ):
                    history.append(msg)
            except discord.Forbidden:
                pass

            history.reverse()

            for msg in history:
                if msg.author.bot and msg.author == self.bot.user:
                    role = "assistant"
                    content = msg.content
                else:
                    role = "user"
                    content = f"{msg.author.display_name}: {msg.content}"
                if content:
                    messages.append({"role": role, "content": content})

        messages.append(
            {
                "role": "user",
                "content": f"{trigger_message.author.display_name}: {user_text}",
            }
        )

        return messages

    async def _call_openai(
        self,
        messages: list,
        model: str,
        max_tokens: int,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> str:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai package is not installed.\n"
                "Run: /data/venv/bin/pip install openai\n"
                "Then: [p]reload llmchat"
            )

        client = AsyncOpenAI(
            base_url=base_url or None,
            api_key=api_key or "none",
        )

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @commands.group(name="llmchat", invoke_without_command=True)
    @commands.guild_only()
    async def llmchat(self, ctx: commands.Context):
        """LLMChat — AI responses via Ollama / OpenAI-compatible APIs."""
        await ctx.send_help()

    @llmchat.command(name="enable")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_enable(
        self, ctx: commands.Context, channel: discord.TextChannel = None
    ):
        """Enable LLMChat in a channel (defaults to current channel)."""
        channel = channel or ctx.channel
        async with self.config.guild(ctx.guild).enabled_channels() as channels:
            if channel.id in channels:
                return await ctx.send(
                    f"✅ LLMChat is already enabled in {channel.mention}."
                )
            channels.append(channel.id)
        await ctx.send(
            f"✅ LLMChat enabled in {channel.mention}. The bot will now respond to @mentions there."
        )

    @llmchat.command(name="disable")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_disable(
        self, ctx: commands.Context, channel: discord.TextChannel = None
    ):
        """Disable LLMChat in a channel (defaults to current channel)."""
        channel = channel or ctx.channel
        async with self.config.guild(ctx.guild).enabled_channels() as channels:
            if channel.id not in channels:
                return await ctx.send(f"ℹ️ LLMChat is not enabled in {channel.mention}.")
            channels.remove(channel.id)
        await ctx.send(f"✅ LLMChat disabled in {channel.mention}.")

    @llmchat.command(name="channels")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_channels(self, ctx: commands.Context):
        """List all channels where LLMChat is enabled."""
        channel_ids = await self.config.guild(ctx.guild).enabled_channels()
        if not channel_ids:
            return await ctx.send(
                "No channels enabled. Use `llmchat enable` to add one."
            )
        mentions = []
        for cid in channel_ids:
            ch = ctx.guild.get_channel(cid)
            mentions.append(ch.mention if ch else f"<deleted channel {cid}>")
        await ctx.send(f"**Enabled channels:** {', '.join(mentions)}")

    @llmchat.command(name="model")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_model(self, ctx: commands.Context, *, model: str):
        """
        Set the model name.

        Examples:
          `[p]llmchat model llama3`               <- Ollama
          `[p]llmchat model mistral`               <- Ollama
          `[p]llmchat model gpt-4o`                <- OpenAI
          `[p]llmchat model mixtral-8x7b-instruct` <- OpenRouter
        """
        await self.config.guild(ctx.guild).model.set(model)
        await ctx.send(f"✅ Model set to `{model}`.")

    @llmchat.command(name="baseurl")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_baseurl(self, ctx: commands.Context, url: str = None):
        """
        Set the API base URL.

        Ollama (default): `[p]llmchat baseurl http://localhost:11434/v1`
        OpenAI:           `[p]llmchat baseurl https://api.openai.com/v1`
        OpenRouter:       `[p]llmchat baseurl https://openrouter.ai/api/v1`
        Clear:            `[p]llmchat baseurl clear`
        """
        if url and url.lower() == "clear":
            url = None
        await self.config.guild(ctx.guild).base_url.set(url)
        await ctx.send(
            f"✅ Base URL set to `{url}`." if url else "✅ Base URL cleared."
        )

    @llmchat.command(name="apikey")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_apikey(self, ctx: commands.Context, key: str = None):
        """
        Set the API key. Run in a private channel to avoid exposing it.

        Ollama doesn't need a real key — any value works (default: "ollama").
        To clear: `[p]llmchat apikey clear`
        """
        try:
            await ctx.message.delete()
        except (discord.Forbidden, discord.HTTPException):
            pass

        if key and key.lower() == "clear":
            key = None

        await self.config.guild(ctx.guild).api_key.set(key)
        if key:
            await ctx.send(
                "✅ API key saved. (Your message was deleted to protect the key.)"
            )
        else:
            await ctx.send("✅ API key cleared.")

    @llmchat.command(name="systemprompt")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_systemprompt(self, ctx: commands.Context, *, prompt: str):
        """Set the system prompt used for all conversations."""
        await self.config.guild(ctx.guild).system_prompt.set(prompt)
        await ctx.send(f"✅ System prompt updated:\n{box(prompt)}")

    @llmchat.command(name="maxtokens")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_maxtokens(self, ctx: commands.Context, tokens: int):
        """Set max response tokens (default: 1024, range: 64-8192)."""
        if tokens < 64 or tokens > 8192:
            return await ctx.send("Please choose a value between 64 and 8192.")
        await self.config.guild(ctx.guild).max_tokens.set(tokens)
        await ctx.send(f"✅ Max tokens set to `{tokens}`.")

    @llmchat.command(name="context")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_context(self, ctx: commands.Context, count: int):
        """Set how many recent messages to include as context (default: 10, max: 50, 0 = off)."""
        if count < 0 or count > 50:
            return await ctx.send("Please choose a value between 0 and 50.")
        await self.config.guild(ctx.guild).context_messages.set(count)
        await ctx.send(f"✅ Context window set to `{count}` messages.")

    @llmchat.command(name="settings")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_settings(self, ctx: commands.Context):
        """Show current LLMChat settings."""
        cfg = await self.config.guild(ctx.guild).all()

        channel_list = []
        for cid in cfg["enabled_channels"]:
            ch = ctx.guild.get_channel(cid)
            channel_list.append(ch.mention if ch else f"<deleted {cid}>")

        embed = discord.Embed(
            title="LLMChat Settings",
            color=await ctx.embed_color(),
            timestamp=datetime.now(timezone.utc),
        )
        embed.add_field(name="Model", value=f"`{cfg['model']}`", inline=True)
        embed.add_field(name="Max Tokens", value=str(cfg["max_tokens"]), inline=True)
        embed.add_field(
            name="Context Messages", value=str(cfg["context_messages"]), inline=True
        )
        embed.add_field(
            name="Base URL",
            value=f"`{cfg['base_url']}`" if cfg["base_url"] else "_default_",
            inline=False,
        )
        embed.add_field(
            name="API Key",
            value="✅ Set" if cfg["api_key"] else "❌ Not set",
            inline=True,
        )
        embed.add_field(
            name="Enabled Channels",
            value=", ".join(channel_list) if channel_list else "_None_",
            inline=False,
        )
        embed.add_field(
            name="System Prompt",
            value=box(
                cfg["system_prompt"][:500]
                + ("..." if len(cfg["system_prompt"]) > 500 else "")
            ),
            inline=False,
        )
        await ctx.send(embed=embed)

    @llmchat.command(name="setup")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_setup(self, ctx: commands.Context):
        """Quick-start guide."""
        guide = (
            "**LLMChat Quick Setup**\n\n"
            "**For local Ollama:**\n"
            f"```\n{ctx.prefix}llmchat model llama3\n"
            f"{ctx.prefix}llmchat baseurl http://localhost:11434/v1\n"
            f"{ctx.prefix}llmchat apikey ollama\n"
            f"{ctx.prefix}llmchat enable #your-channel\n```\n"
            "**For OpenAI:**\n"
            f"```\n{ctx.prefix}llmchat model gpt-4o\n"
            f"{ctx.prefix}llmchat baseurl https://api.openai.com/v1\n"
            f"{ctx.prefix}llmchat apikey sk-YOUR_KEY\n"
            f"{ctx.prefix}llmchat enable #your-channel\n```\n"
            "**For OpenRouter:**\n"
            f"```\n{ctx.prefix}llmchat model mixtral-8x7b-instruct\n"
            f"{ctx.prefix}llmchat baseurl https://openrouter.ai/api/v1\n"
            f"{ctx.prefix}llmchat apikey sk-or-YOUR_KEY\n"
            f"{ctx.prefix}llmchat enable #your-channel\n```\n"
            f"See all settings: `{ctx.prefix}llmchat settings`"
        )
        await ctx.send(guide)
