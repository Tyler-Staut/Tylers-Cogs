from .hybrid_analysis import hybrid_analysis


async def setup(bot):
    await bot.add_cog(hybrid_analysis(bot))
