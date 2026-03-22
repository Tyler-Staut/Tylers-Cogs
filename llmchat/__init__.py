from .llmchat import LLMChat


async def setup(bot):
    await bot.add_cog(LLMChat(bot))
