"""
LLMChat — A Red-DiscordBot cog that responds to @mentions using the OpenAI-compatible API.

Uses the lightweight `openai` Python package, which works with:
- Ollama (local) base_url = http://localhost:11434/v1, api_key = "ollama"
- OpenAI base_url = (default), api_key = sk-...
- OpenRouter base_url = https://openrouter.ai/api/v1, api_key = sk-or-...
- Any OpenAI-compatible endpoint (LM Studio, vLLM, etc.)

In DMs (owner only): commands set global defaults applied to all guilds.
In a guild (admin): commands set per-guild overrides. Guild settings take
priority; anything not set falls back to the global default.

Smart history flow:
  1. Classifier call decides if history would help.
  2. If yes → bot sends an embed asking the user to confirm with Yes / No buttons.
     - Yes      → fetch history, answer with it.
     - No       → answer without history.
     - 60s timeout → answer without history (buttons disabled, embed updated).
  3. If classifier says no → answer immediately, no prompt.
"""

import asyncio
import json
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
DEFAULT_API_KEY = "ollama"
HISTORY_PROMPT_TIMEOUT = 60  # seconds to wait for Yes/No before proceeding without history

# ---------------------------------------------------------------------------
# Classifier prompt — kept very short so it's cheap to run
# ---------------------------------------------------------------------------
_HISTORY_CLASSIFIER_PROMPT = """\
You decide whether chat history is needed to answer a Discord message.

Return ONLY valid JSON — no markdown fences, no extra text:
{"needs_history": true, "reason": "one short sentence"}
or
{"needs_history": false, "reason": "one short sentence"}

Return needs_history=true when the message:
- Uses pronouns or references without a clear antecedent ("that", "it", "the one you mentioned")
- Is a follow-up or continuation ("what about...", "and also...", "wait so...")
- Asks about something said earlier ("what did you say about...", "earlier you mentioned...")
- Is part of an ongoing task or multi-step question
- Is ambiguous without context

Return needs_history=false when the message:
- Is a self-contained, standalone question or command
- Introduces a new topic with no backwards reference
- Is a greeting or simple acknowledgement
- Is a factual question that makes sense without context
"""


# ---------------------------------------------------------------------------
# Discord UI — Yes / No buttons
# ---------------------------------------------------------------------------

def _make_history_embed(
    title: str,
    description: str,
    color: discord.Color,
    show_footer: bool = True,
) -> discord.Embed:
    embed = discord.Embed(title=title, description=description, color=color)
    if show_footer:
        embed.set_footer(
            text=f"No response within {HISTORY_PROMPT_TIMEOUT}s → I'll answer without history."
        )
    return embed


class HistoryPromptView(discord.ui.View):
    """
    Embed + buttons asking whether to fetch message history.

    Outcomes:
      confirmed = True   → fetch history
      confirmed = False  → skip history (user said No, or timed out)
    """

    def __init__(self, author: discord.Member):
        super().__init__(timeout=HISTORY_PROMPT_TIMEOUT)
        self.author = author
        self.confirmed: Optional[bool] = None
        self._event = asyncio.Event()

    async def _resolve(self, interaction: discord.Interaction, confirmed: bool, embed: discord.Embed):
        self.confirmed = confirmed
        for child in self.children:
            child.disabled = True
        self._event.set()
        self.stop()
        await interaction.response.edit_message(embed=embed, view=self)

    async def wait_for_response(self) -> bool:
        """Block until the user clicks or the timeout fires. Returns True → use history."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=HISTORY_PROMPT_TIMEOUT)
        except asyncio.TimeoutError:
            self.confirmed = False
            for child in self.children:
                child.disabled = True
            self.stop()
        return bool(self.confirmed)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author.id:
            await interaction.response.send_message(
                "Only the person who asked can respond to this prompt.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Yes, use history", style=discord.ButtonStyle.success, emoji="📜")
    async def yes_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        embed = _make_history_embed(
            title="✅ Got it — fetching history",
            description="Reading recent messages to give you a better answer…",
            color=discord.Color.green(),
            show_footer=False,
        )
        await self._resolve(interaction, confirmed=True, embed=embed)

    @discord.ui.button(label="No, just answer", style=discord.ButtonStyle.danger, emoji="🚫")
    async def no_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        embed = _make_history_embed(
            title="🚫 Skipping history",
            description="Answering based only on your message.",
            color=discord.Color.red(),
            show_footer=False,
        )
        await self._resolve(interaction, confirmed=False, embed=embed)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class LLMChat(commands.Cog):
    """
    AI chat via an OpenAI-compatible API. Responds to @mentions in enabled channels.

    Works with Ollama (local), OpenAI, OpenRouter, LM Studio, vLLM, and more.
    Configure globally from DMs (owner) or per-guild in a server (admin).

    When a message seems to need conversation history the bot sends an embed
    with Yes / No buttons. History is only fetched if the user confirms.
    """

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(
            self, identifier=0x4C4C4D434841, force_registration=True
        )

        # Per-guild overrides — None means "use global default"
        self.config.register_guild(
            enabled_channels=[],
            model=None,
            system_prompt=None,
            max_tokens=None,
            context_messages=None,
            base_url=None,
            api_key=None,
        )

        # Global defaults — owner-configurable from DMs
        self.config.register_global(
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
                " /data/venv/bin/pip install openai"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_effective_config(self, guild: discord.Guild) -> dict:
        """Guild settings win; anything left as None falls back to global."""
        g = await self.config.guild(guild).all()
        glob = await self.config.all()
        return {
            "enabled_channels": g["enabled_channels"],
            "model": g["model"] or glob["model"],
            "system_prompt": g["system_prompt"] or glob["system_prompt"],
            "max_tokens": (
                g["max_tokens"] if g["max_tokens"] is not None else glob["max_tokens"]
            ),
            "context_messages": (
                g["context_messages"]
                if g["context_messages"] is not None
                else glob["context_messages"]
            ),
            "base_url": g["base_url"] or glob["base_url"],
            "api_key": g["api_key"] or glob["api_key"],
        }

    async def _is_owner(self, ctx: commands.Context) -> bool:
        return await ctx.bot.is_owner(ctx.author)

    async def _check_dm_owner(self, ctx: commands.Context) -> bool:
        if ctx.guild is None and not await self._is_owner(ctx):
            await ctx.send("Only the bot owner can configure LLMChat from DMs.")
            return False
        return True

    async def _check_guild_admin(self, ctx: commands.Context) -> bool:
        if ctx.guild is not None:
            if not (
                ctx.author.guild_permissions.administrator
                or ctx.author.guild_permissions.manage_guild
                or await self._is_owner(ctx)
            ):
                await ctx.send(
                    "You need the Manage Server permission to configure LLMChat."
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Smart history classifier
    # ------------------------------------------------------------------

    async def _needs_history(
        self,
        user_text: str,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> tuple[bool, str]:
        """
        Ask the model whether message history would help answer `user_text`.

        Returns (needs_history: bool, reason: str).
        Falls back to (False, "") on any error.
        """
        if not _OPENAI_AVAILABLE:
            return False, ""

        try:
            client = AsyncOpenAI(
                base_url=base_url or None,
                api_key=api_key or "none",
            )
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _HISTORY_CLASSIFIER_PROMPT},
                        {
                            "role": "user",
                            "content": f'Discord message to classify:\n"{user_text}"',
                        },
                    ],
                    max_tokens=60,
                    temperature=0,  # deterministic
                ),
                timeout=15,
            )
            raw = response.choices[0].message.content.strip()

            # Strip accidental markdown fences some models add
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            needs = bool(data.get("needs_history", False))
            reason = data.get("reason", "")
            log.debug("History classifier: needs=%s — %s", needs, reason)
            return needs, reason

        except (asyncio.TimeoutError, json.JSONDecodeError, Exception) as exc:
            log.debug("History classifier failed (%s), skipping history", exc)
            return False, ""

    # ------------------------------------------------------------------
    # Listener
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
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
        cfg = await self._get_effective_config(message.guild)

        user_text = message.content
        for mention in [f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"]:
            user_text = user_text.replace(mention, "").strip()

        if not user_text:
            user_text = "(no text — the user only pinged you)"

        # ----------------------------------------------------------------
        # Step 1: classify — does this message likely need history?
        # ----------------------------------------------------------------
        use_history = False
        prompt_msg: Optional[discord.Message] = None

        if cfg["context_messages"] > 0:
            needs, reason = await self._needs_history(
                user_text,
                model=cfg["model"],
                base_url=cfg["base_url"],
                api_key=cfg["api_key"],
            )

            if needs:
                # ------------------------------------------------------------
                # Step 2: send the embed + buttons and wait for user response
                # ------------------------------------------------------------
                view = HistoryPromptView(author=message.author)

                embed = _make_history_embed(
                    title="📜 Should I check the conversation history?",
                    description=(
                        f"Your message seems to reference earlier context.\n"
                        f"**Reason:** {reason}\n\n"
                        f"Should I read the last **{cfg['context_messages']}** messages "
                        f"to give a better answer?"
                    ),
                    color=discord.Color.blurple(),
                    show_footer=True,
                )

                prompt_msg = await message.reply(embed=embed, view=view, mention_author=False)

                # Blocks here until button press or timeout
                use_history = await view.wait_for_response()

                # If the view timed out (no button pressed), update the embed
                if not view._event.is_set():
                    timeout_embed = _make_history_embed(
                        title="⏱️ No response — skipping history",
                        description="Answering based only on your message.",
                        color=discord.Color.greyple(),
                        show_footer=False,
                    )
                    try:
                        await prompt_msg.edit(embed=timeout_embed, view=view)
                    except discord.HTTPException:
                        pass

        # ----------------------------------------------------------------
        # Step 3: build the messages list and call the model
        # ----------------------------------------------------------------
        messages = await self._build_messages(
            message,
            user_text,
            system_prompt=cfg["system_prompt"],
            context_limit=cfg["context_messages"] if use_history else 0,
            history_skipped=not use_history,
        )

        async with message.channel.typing():
            try:
                reply = await asyncio.wait_for(
                    self._call_openai(
                        messages=messages,
                        model=cfg["model"],
                        max_tokens=cfg["max_tokens"],
                        base_url=cfg["base_url"],
                        api_key=cfg["api_key"],
                    ),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                await message.reply(
                    "⏱️ The model took too long to respond. Try again or check your backend.",
                    mention_author=False,
                )
                return
            except Exception as exc:
                log.exception("LLM API call failed")
                await message.reply(f"❌ Error calling the model: {exc}", mention_author=False)
                return

        # Always reply to the original message, not the prompt embed
        if len(reply) <= 2000:
            await message.reply(reply, mention_author=False)
        else:
            chunks = [reply[i : i + 1990] for i in range(0, len(reply), 1990)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk, mention_author=False)
                else:
                    await message.channel.send(chunk)

    async def _build_messages(
        self,
        trigger_message: discord.Message,
        user_text: str,
        system_prompt: str,
        context_limit: int,
        history_skipped: bool = False,
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

        elif history_skipped:
            # Tell the model it has no prior context so it doesn't hallucinate references
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Note: The user's message is being answered without conversation history. "
                        "Do not refer to or assume any prior context."
                    ),
                }
            )

        # Final user message — supports image attachments
        image_attachments = [
            a
            for a in trigger_message.attachments
            if a.content_type and a.content_type.startswith("image/")
        ]

        if image_attachments:
            content_parts: list = [
                {
                    "type": "text",
                    "text": f"{trigger_message.author.display_name}: {user_text}",
                }
            ]
            for attachment in image_attachments:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": attachment.url},
                    }
                )
            messages.append({"role": "user", "content": content_parts})
        else:
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
    async def llmchat(self, ctx: commands.Context):
        """
        LLMChat — AI responses via Ollama / OpenAI-compatible APIs.

        In a server (admin): sets per-guild config.
        In DMs (owner only): sets global defaults for all guilds.
        """
        await ctx.send_help()

    @llmchat.command(name="enable")
    async def llmchat_enable(
        self, ctx: commands.Context, channel: discord.TextChannel = None
    ):
        """Enable LLMChat in a channel. Must be run in a server."""
        if ctx.guild is None:
            return await ctx.send("This command must be run in a server, not DMs.")
        if not await self._check_guild_admin(ctx):
            return
        channel = channel or ctx.channel
        async with self.config.guild(ctx.guild).enabled_channels() as channels:
            if channel.id in channels:
                return await ctx.send(
                    f"✅ LLMChat is already enabled in {channel.mention}."
                )
            channels.append(channel.id)
        await ctx.send(f"✅ LLMChat enabled in {channel.mention}.")

    @llmchat.command(name="disable")
    async def llmchat_disable(
        self, ctx: commands.Context, channel: discord.TextChannel = None
    ):
        """Disable LLMChat in a channel. Must be run in a server."""
        if ctx.guild is None:
            return await ctx.send("This command must be run in a server, not DMs.")
        if not await self._check_guild_admin(ctx):
            return
        channel = channel or ctx.channel
        async with self.config.guild(ctx.guild).enabled_channels() as channels:
            if channel.id not in channels:
                return await ctx.send(f"ℹ️ LLMChat is not enabled in {channel.mention}.")
            channels.remove(channel.id)
        await ctx.send(f"✅ LLMChat disabled in {channel.mention}.")

    @llmchat.command(name="channels")
    async def llmchat_channels(self, ctx: commands.Context):
        """List enabled channels. Must be run in a server."""
        if ctx.guild is None:
            return await ctx.send("This command must be run in a server, not DMs.")
        if not await self._check_guild_admin(ctx):
            return
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
    async def llmchat_model(self, ctx: commands.Context, *, model: str):
        """
        Set the model name.

        In a server: overrides the model for this guild only.
        In DMs (owner): sets the global default for all guilds.

        Examples:
          `[p]llmchat model llava`                 <- Ollama (vision-capable)
          `[p]llmchat model llama3`                <- Ollama (text only)
          `[p]llmchat model gpt-4o`                <- OpenAI (vision + text)
          `[p]llmchat model mixtral-8x7b-instruct` <- OpenRouter
        """
        if not await self._check_dm_owner(ctx):
            return
        if not await self._check_guild_admin(ctx):
            return
        if ctx.guild:
            await self.config.guild(ctx.guild).model.set(model)
            await ctx.send(f"✅ Model set to `{model}` for this server.")
        else:
            await self.config.model.set(model)
            await ctx.send(f"✅ Global default model set to `{model}`.")

    @llmchat.command(name="baseurl")
    async def llmchat_baseurl(self, ctx: commands.Context, url: str = None):
        """
        Set the API base URL.

        In a server: overrides for this guild only.
        In DMs (owner): sets the global default.

        Ollama:     `[p]llmchat baseurl http://localhost:11434/v1`
        OpenAI:     `[p]llmchat baseurl https://api.openai.com/v1`
        OpenRouter: `[p]llmchat baseurl https://openrouter.ai/api/v1`
        Clear:      `[p]llmchat baseurl clear`
        """
        if not await self._check_dm_owner(ctx):
            return
        if not await self._check_guild_admin(ctx):
            return

        if url and url.lower() == "clear":
            url = None

        if ctx.guild:
            await self.config.guild(ctx.guild).base_url.set(url)
            scope = "for this server"
        else:
            await self.config.base_url.set(url or DEFAULT_BASE_URL)
            scope = "globally"

        await ctx.send(
            f"✅ Base URL set to `{url}` {scope}."
            if url
            else f"✅ Base URL reset to default {scope}."
        )

    @llmchat.command(name="apikey")
    async def llmchat_apikey(self, ctx: commands.Context, key: str = None):
        """
        Set the API key. Run in DMs to keep it private.

        In a server: overrides for this guild only.
        In DMs (owner): sets the global default.
        To clear: `[p]llmchat apikey clear`
        """
        if not await self._check_dm_owner(ctx):
            return
        if not await self._check_guild_admin(ctx):
            return

        try:
            await ctx.message.delete()
        except (discord.Forbidden, discord.HTTPException):
            pass

        if key and key.lower() == "clear":
            key = None

        if ctx.guild:
            await self.config.guild(ctx.guild).api_key.set(key)
            scope = "for this server"
        else:
            await self.config.api_key.set(key or DEFAULT_API_KEY)
            scope = "globally"

        await ctx.send(f"✅ API key saved {scope}." if key else f"✅ API key cleared {scope}.")

    @llmchat.command(name="systemprompt")
    async def llmchat_systemprompt(self, ctx: commands.Context, *, prompt: str):
        """
        Set the system prompt.

        In a server: overrides for this guild only.
        In DMs (owner): sets the global default.
        """
        if not await self._check_dm_owner(ctx):
            return
        if not await self._check_guild_admin(ctx):
            return

        if ctx.guild:
            await self.config.guild(ctx.guild).system_prompt.set(prompt)
            scope = "for this server"
        else:
            await self.config.system_prompt.set(prompt)
            scope = "globally"

        await ctx.send(f"✅ System prompt updated {scope}:\n{box(prompt)}")

    @llmchat.command(name="maxtokens")
    async def llmchat_maxtokens(self, ctx: commands.Context, tokens: int):
        """
        Set max response tokens (range: 64-8192).

        In a server: overrides for this guild only.
        In DMs (owner): sets the global default.
        """
        if not await self._check_dm_owner(ctx):
            return
        if not await self._check_guild_admin(ctx):
            return

        if tokens < 64 or tokens > 8192:
            return await ctx.send("Please choose a value between 64 and 8192.")

        if ctx.guild:
            await self.config.guild(ctx.guild).max_tokens.set(tokens)
            scope = "for this server"
        else:
            await self.config.max_tokens.set(tokens)
            scope = "globally"

        await ctx.send(f"✅ Max tokens set to `{tokens}` {scope}.")

    @llmchat.command(name="context")
    async def llmchat_context(self, ctx: commands.Context, count: int):
        """
        Set how many recent messages to include as context when history is needed (0-50).

        History is only fetched when the classifier decides it will help AND the
        user confirms via the embed prompt. Setting to 0 disables history entirely.

        In a server: overrides for this guild only.
        In DMs (owner): sets the global default.
        """
        if not await self._check_dm_owner(ctx):
            return
        if not await self._check_guild_admin(ctx):
            return

        if count < 0 or count > 50:
            return await ctx.send("Please choose a value between 0 and 50.")

        if ctx.guild:
            await self.config.guild(ctx.guild).context_messages.set(count)
            scope = "for this server"
        else:
            await self.config.context_messages.set(count)
            scope = "globally"

        await ctx.send(f"✅ Context window set to `{count}` messages {scope}.")

    @llmchat.command(name="settings")
    async def llmchat_settings(self, ctx: commands.Context):
        """
        Show current LLMChat settings.

        In a server: shows guild settings and effective values (with global fallback).
        In DMs (owner): shows global defaults.
        """
        if not await self._check_dm_owner(ctx):
            return
        if not await self._check_guild_admin(ctx):
            return

        glob = await self.config.all()

        embed = discord.Embed(
            title="LLMChat Settings",
            color=await ctx.embed_color(),
            timestamp=datetime.now(timezone.utc),
        )

        if ctx.guild:
            g = await self.config.guild(ctx.guild).all()
            eff = await self._get_effective_config(ctx.guild)

            channel_list = []
            for cid in g["enabled_channels"]:
                ch = ctx.guild.get_channel(cid)
                channel_list.append(ch.mention if ch else f"<deleted {cid}>")

            def fmt(guild_val, effective_val):
                if guild_val is None:
                    return f"`{effective_val}` _(global default)_"
                return f"`{guild_val}`"

            embed.description = "Showing guild overrides. Values marked _(global default)_ are inherited."
            embed.add_field(name="Model", value=fmt(g["model"], eff["model"]), inline=True)
            embed.add_field(
                name="Max Tokens", value=fmt(g["max_tokens"], eff["max_tokens"]), inline=True
            )
            embed.add_field(
                name="Context Messages (max)",
                value=fmt(g["context_messages"], eff["context_messages"]),
                inline=True,
            )
            embed.add_field(
                name="Smart History",
                value=(
                    f"✅ Classifier + **user confirmation embed**\n"
                    f"Timeout: {HISTORY_PROMPT_TIMEOUT}s → skip history"
                ),
                inline=False,
            )
            embed.add_field(
                name="Base URL", value=fmt(g["base_url"], eff["base_url"]), inline=False
            )
            embed.add_field(
                name="API Key",
                value=(
                    "✅ Set (guild)"
                    if g["api_key"]
                    else ("✅ Set (global)" if glob["api_key"] else "❌ Not set")
                ),
                inline=True,
            )
            embed.add_field(
                name="Enabled Channels",
                value=", ".join(channel_list) if channel_list else "_None_",
                inline=False,
            )
            sp = g["system_prompt"] or glob["system_prompt"]
            sp_label = "System Prompt" + ("" if g["system_prompt"] else " _(global default)_")
            embed.add_field(
                name=sp_label,
                value=box(sp[:400] + ("..." if len(sp) > 400 else "")),
                inline=False,
            )
        else:
            embed.description = "Global defaults — applied to any guild that hasn't set its own override."
            embed.add_field(name="Model", value=f"`{glob['model']}`", inline=True)
            embed.add_field(name="Max Tokens", value=str(glob["max_tokens"]), inline=True)
            embed.add_field(
                name="Context Messages (max)", value=str(glob["context_messages"]), inline=True
            )
            embed.add_field(
                name="Smart History",
                value=(
                    f"✅ Classifier + **user confirmation embed**\n"
                    f"Timeout: {HISTORY_PROMPT_TIMEOUT}s → skip history"
                ),
                inline=False,
            )
            embed.add_field(name="Base URL", value=f"`{glob['base_url']}`", inline=False)
            embed.add_field(
                name="API Key",
                value="✅ Set" if glob["api_key"] else "❌ Not set",
                inline=True,
            )
            sp = glob["system_prompt"]
            embed.add_field(
                name="System Prompt",
                value=box(sp[:400] + ("..." if len(sp) > 400 else "")),
                inline=False,
            )

        await ctx.send(embed=embed)

    @llmchat.command(name="setup")
    async def llmchat_setup(self, ctx: commands.Context):
        """Quick-start guide. Works in DMs and in servers."""
        p = ctx.prefix
        guide = (
            "**LLMChat Quick Setup**\n\n"
            "**Step 1 — install openai into the bot venv (once, on the host):**\n"
            "```\n/data/venv/bin/pip install openai --no-deps\n```\n"
            "**Step 2 — set your backend (in DMs = global default, in server = guild override):**\n"
            f"```\n{p}llmchat model llava\n"
            f"{p}llmchat baseurl http://localhost:11434/v1\n"
            f"{p}llmchat apikey ollama\n```\n"
            "**Step 3 — enable in a channel (must be in the server):**\n"
            f"```\n{p}llmchat enable #your-channel\n```\n"
            "**Smart history flow:**\n"
            "1. A classifier call checks if your message likely needs conversation history.\n"
            "2. If yes → the bot sends an embed with **📜 Yes** / **🚫 No** buttons.\n"
            "3. Confirm → history is fetched and included. Decline (or ignore for 60s) → answered without it.\n\n"
            "**Vision/image support:** use a vision-capable model such as `llava` (Ollama) or `gpt-4o` (OpenAI).\n"
            "Text-only models like `llama3` will error if an image is sent.\n\n"
            "**Other backends:**\n"
            f"OpenAI: `{p}llmchat baseurl https://api.openai.com/v1`\n"
            f"OpenRouter: `{p}llmchat baseurl https://openrouter.ai/api/v1`\n\n"
            f"See settings: `{p}llmchat settings`"
        )
        await ctx.send(guide)
