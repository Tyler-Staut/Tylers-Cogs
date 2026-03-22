# LLMChat

A [Red-DiscordBot](https://github.com/Cog-Creators/Red-DiscordBot) cog that responds to **@mentions** in configured channels using any [LiteLLM](https://github.com/BerriAI/litellm)-compatible model — including local **Ollama** models.

## Features

- 🤖 Responds **only when @mentioned** in enabled channels (no random trigger spam)
- 🦙 **Ollama** support out of the box (local, no API key needed)
- 🔀 **Any LiteLLM backend**: OpenAI, OpenRouter, Anthropic, Azure, local APIs, etc.
- 💬 Includes recent channel **conversation context** so replies feel coherent
- ⚙️ Per-server configuration: model, system prompt, base URL, API key, max tokens
- 🔒 Admin-only configuration commands

## Installation

```
[p]repo add tylers-cogs https://github.com/Tyler-Staut/Tylers-Cogs
[p]cog install tylers-cogs llmchat
[p]load llmchat
```

Also install the `litellm` Python package on your bot's host:
```
pip install litellm
```

## Quick Setup

### Local Ollama

```
[p]llmchat model ollama/llama3
[p]llmchat baseurl http://localhost:11434
[p]llmchat enable #your-channel
```

### OpenAI

```
[p]llmchat model gpt-4o
[p]llmchat apikey sk-...
[p]llmchat enable #your-channel
```

### OpenRouter

```
[p]llmchat model openrouter/mistralai/mixtral-8x7b-instruct
[p]llmchat apikey sk-or-...
[p]llmchat enable #your-channel
```

## Commands

| Command | Description |
|---|---|
| `[p]llmchat setup` | Quick-start guide |
| `[p]llmchat settings` | Show current configuration |
| `[p]llmchat enable [#channel]` | Enable in a channel |
| `[p]llmchat disable [#channel]` | Disable in a channel |
| `[p]llmchat channels` | List enabled channels |
| `[p]llmchat model <model>` | Set the LiteLLM model string |
| `[p]llmchat baseurl <url\|clear>` | Set a custom API base URL |
| `[p]llmchat apikey <key\|clear>` | Set the API key |
| `[p]llmchat systemprompt <prompt>` | Set the system prompt |
| `[p]llmchat maxtokens <n>` | Set max response tokens (64–8192) |
| `[p]llmchat context <n>` | Set how many prior messages to include (0–50) |

All configuration commands require **Admin** or **Manage Server** permissions.

## How It Works

1. When the bot is @mentioned in an enabled channel, it grabs the last N messages for context.
2. Those messages are formatted into a conversation and sent to the configured LiteLLM model.
3. The response is sent as a reply to the original message.
4. Responses longer than 2000 characters are split into multiple messages.
