"""
LLMChat — A Red-DiscordBot cog that responds to @mentions using LiteLLM.

Supports any LiteLLM-compatible backend including:
  - Ollama (local)  e.g. model = "ollama/llama3"
  - OpenAI          e.g. model = "gpt-4o"
  - OpenRouter      e.g. model = "openrouter/mistralai/mixtral-8x7b-instruct"
  - Any OpenAI-compatible API via custom base_url
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import discord
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import box

log = logging.getLogger("red.tylerscogs.llmchat")

# How many recent messages to include as conversation context
DEFAULT_CONTEXT_MESSAGES = 10
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MODEL = "ollama/llama3"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant in a Discord server. "
    "Be concise, friendly, and accurate. "
    "You are replying to a Discord message, so keep responses reasonably short unless detail is needed."
)


class LLMChat(commands.Cog):
    """
    AI chat powered by LiteLLM. Responds to @mentions in enabled channels.

    Supports Ollama, OpenAI, OpenRouter, and any OpenAI-compatible API.
    """

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=0x4C4C4D434841, force_registration=True)

        # Guild-level defaults
        self.config.register_guild(
            enabled_channels=[],       # list of channel IDs where the bot will respond
            model=DEFAULT_MODEL,       # LiteLLM model string
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_tokens=DEFAULT_MAX_TOKENS,
            context_messages=DEFAULT_CONTEXT_MESSAGES,
            base_url=None,             # Optional: custom API base URL (for Ollama or custom endpoints)
            api_key=None,              # Optional: stored per-guild (use [p]set api for production)
        )

    # ------------------------------------------------------------------
    # Listener
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Fires on every message; only processes @mentions in enabled channels."""
        # Ignore bots (including ourselves)
        if message.author.bot:
            return

        # Must be in a guild
        if not message.guild:
            return

        # Bot must be mentioned
        if self.bot.user not in message.mentions:
            return

        # Check if this channel is enabled
        enabled_channels = await self.config.guild(message.guild).enabled_channels()
        if message.channel.id not in enabled_channels:
            return

        # Don't respond if the bot can't send messages here
        if not message.channel.permissions_for(message.guild.me).send_messages:
            return

        await self._handle_mention(message)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _handle_mention(self, message: discord.Message):
        """Build context and call LiteLLM, then reply."""
        guild_config = await self.config.guild(message.guild).all()

        # Strip the bot mention from the user's message
        user_text = message.content
        for mention in [f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"]:
            user_text = user_text.replace(mention, "").strip()

        if not user_text:
            user_text = "(no text — the user only pinged you)"

        # Build conversation history from recent channel messages
        messages = await self._build_messages(
            message,
            user_text,
            system_prompt=guild_config["system_prompt"],
            context_limit=guild_config["context_messages"],
        )

        async with message.channel.typing():
            try:
                reply = await asyncio.wait_for(
                    self._call_litellm(
                        messages=messages,
                        model=guild_config["model"],
                        max_tokens=guild_config["max_tokens"],
                        base_url=guild_config["base_url"],
                        api_key=guild_config["api_key"],
                    ),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                await message.reply("⏱️ The model took too long to respond. Try again or check your backend.")
                return
            except Exception as exc:
                log.exception("LiteLLM call failed")
                await message.reply(f"❌ Error calling the model: {exc}")
                return

        # Discord messages max 2000 chars; split if needed
        if len(reply) <= 2000:
            await message.reply(reply)
        else:
            # Send in chunks
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
    ) -> list[dict]:
        """Fetch recent channel history and build the messages list for the API."""
        messages = [{"role": "system", "content": system_prompt}]

        # Fetch prior messages for context (excludes the trigger message itself)
        history = []
        if context_limit > 0:
            try:
                async for msg in trigger_message.channel.history(
                    limit=context_limit + 1,
                    before=trigger_message,
                ):
                    history.append(msg)
            except discord.Forbidden:
                pass  # No history access — that's fine, we'll continue without it

            history.reverse()  # oldest first

            for msg in history:
                if msg.author.bot and msg.author == self.bot.user:
                    role = "assistant"
                    content = msg.content
                else:
                    role = "user"
                    # Label non-trigger messages with their author name for clarity
                    content = f"{msg.author.display_name}: {msg.content}"
                if content:
                    messages.append({"role": role, "content": content})

        # Finally, add the actual user message that triggered us
        messages.append({
            "role": "user",
            "content": f"{trigger_message.author.display_name}: {user_text}",
        })

        return messages

    async def _call_litellm(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> str:
        """Run a LiteLLM completion in a thread so we don't block the event loop."""
        try:
            import litellm
        except ImportError:
            raise RuntimeError(
                "litellm is not installed. Run: `pip install litellm`"
            )

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

        # litellm.completion is sync; run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: litellm.completion(**kwargs),
        )

        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @commands.group(name="llmchat", invoke_without_command=True)
    @commands.guild_only()
    async def llmchat(self, ctx: commands.Context):
        """LLMChat — AI responses powered by LiteLLM/Ollama."""
        await ctx.send_help()

    # --- Channel management ---

    @llmchat.command(name="enable")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_enable(self, ctx: commands.Context, channel: discord.TextChannel = None):
        """Enable LLMChat in a channel (defaults to current channel)."""
        channel = channel or ctx.channel
        async with self.config.guild(ctx.guild).enabled_channels() as channels:
            if channel.id in channels:
                return await ctx.send(f"✅ LLMChat is already enabled in {channel.mention}.")
            channels.append(channel.id)
        await ctx.send(f"✅ LLMChat enabled in {channel.mention}. The bot will now respond to @mentions there.")

    @llmchat.command(name="disable")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_disable(self, ctx: commands.Context, channel: discord.TextChannel = None):
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
            return await ctx.send("No channels enabled. Use `llmchat enable` to add one.")
        mentions = []
        for cid in channel_ids:
            ch = ctx.guild.get_channel(cid)
            mentions.append(ch.mention if ch else f"<deleted channel {cid}>")
        await ctx.send(f"**Enabled channels:** {', '.join(mentions)}")

    # --- Model configuration ---

    @llmchat.command(name="model")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_model(self, ctx: commands.Context, *, model: str):
        """
        Set the LiteLLM model string.

        Examples:
          `[p]llmchat model ollama/llama3`
          `[p]llmchat model ollama/mistral`
          `[p]llmchat model gpt-4o`
          `[p]llmchat model openrouter/mistralai/mixtral-8x7b-instruct`
        """
        await self.config.guild(ctx.guild).model.set(model)
        await ctx.send(f"✅ Model set to `{model}`.")

    @llmchat.command(name="baseurl")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_baseurl(self, ctx: commands.Context, url: str = None):
        """
        Set (or clear) a custom API base URL.

        For local Ollama, use: `[p]llmchat baseurl http://localhost:11434`
        To clear and use the default, run: `[p]llmchat baseurl clear`
        """
        if url and url.lower() == "clear":
            url = None
        await self.config.guild(ctx.guild).base_url.set(url)
        if url:
            await ctx.send(f"✅ Base URL set to `{url}`.")
        else:
            await ctx.send("✅ Base URL cleared (using LiteLLM default).")

    @llmchat.command(name="apikey")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_apikey(self, ctx: commands.Context, key: str = None):
        """
        Set (or clear) the API key for this guild.

        Run in a private channel or DM to avoid exposing the key.
        To clear, run: `[p]llmchat apikey clear`

        Note: For Ollama (local), no API key is needed.
        """
        try:
            await ctx.message.delete()
        except (discord.Forbidden, discord.HTTPException):
            pass

        if key and key.lower() == "clear":
            key = None

        await self.config.guild(ctx.guild).api_key.set(key)
        if key:
            await ctx.send("✅ API key saved. (Your message was deleted to protect the key.)")
        else:
            await ctx.send("✅ API key cleared.")

    # --- Prompt & token settings ---

    @llmchat.command(name="systemprompt")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_systemprompt(self, ctx: commands.Context, *, prompt: str):
        """Set the system prompt the model uses for all conversations."""
        await self.config.guild(ctx.guild).system_prompt.set(prompt)
        await ctx.send(f"✅ System prompt updated:\n{box(prompt)}")

    @llmchat.command(name="maxtokens")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_maxtokens(self, ctx: commands.Context, tokens: int):
        """Set the max tokens for model responses (default: 1024)."""
        if tokens < 64 or tokens > 8192:
            return await ctx.send("Please choose a value between 64 and 8192.")
        await self.config.guild(ctx.guild).max_tokens.set(tokens)
        await ctx.send(f"✅ Max tokens set to `{tokens}`.")

    @llmchat.command(name="context")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_context(self, ctx: commands.Context, count: int):
        """
        Set how many recent channel messages to include as context (default: 10).

        Set to 0 to disable context (only the pinged message is sent).
        """
        if count < 0 or count > 50:
            return await ctx.send("Please choose a value between 0 and 50.")
        await self.config.guild(ctx.guild).context_messages.set(count)
        await ctx.send(f"✅ Context window set to `{count}` messages.")

    # --- Info / status ---

    @llmchat.command(name="settings")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_settings(self, ctx: commands.Context):
        """Show current LLMChat settings for this server."""
        cfg = await self.config.guild(ctx.guild).all()

        channel_ids = cfg["enabled_channels"]
        channel_list = []
        for cid in channel_ids:
            ch = ctx.guild.get_channel(cid)
            channel_list.append(ch.mention if ch else f"<deleted {cid}>")

        embed = discord.Embed(
            title="LLMChat Settings",
            color=await ctx.embed_color(),
            timestamp=datetime.now(timezone.utc),
        )
        embed.add_field(name="Model", value=f"`{cfg['model']}`", inline=True)
        embed.add_field(name="Max Tokens", value=str(cfg["max_tokens"]), inline=True)
        embed.add_field(name="Context Messages", value=str(cfg["context_messages"]), inline=True)
        embed.add_field(
            name="Base URL",
            value=f"`{cfg['base_url']}`" if cfg["base_url"] else "_LiteLLM default_",
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
            value=box(cfg["system_prompt"][:500] + ("..." if len(cfg["system_prompt"]) > 500 else "")),
            inline=False,
        )
        await ctx.send(embed=embed)

    @llmchat.command(name="setup")
    @commands.admin_or_permissions(manage_guild=True)
    async def llmchat_setup(self, ctx: commands.Context):
        """Quick-start guide for LLMChat."""
        guide = (
            "**LLMChat Quick Setup**\n\n"
            "**1. Install litellm** (if not already):\n"
            "```\npip install litellm\n```\n"
            "**2. Set your model:**\n"
            f"```\n{ctx.prefix}llmchat model ollama/llama3\n```\n"
            "For Ollama, also set the base URL:\n"
            f"```\n{ctx.prefix}llmchat baseurl http://localhost:11434\n```\n"
            "For OpenAI or OpenRouter, set an API key:\n"
            f"```\n{ctx.prefix}llmchat apikey YOUR_KEY_HERE\n```\n"
            "**3. Enable in a channel:**\n"
            f"```\n{ctx.prefix}llmchat enable #channel-name\n```\n"
            "**4. Ping the bot in that channel and it'll respond!**\n\n"
            f"See all settings: `{ctx.prefix}llmchat settings`\n"
            f"See all commands: `{ctx.prefix}help llmchat`"
        )
        await ctx.send(guide)
