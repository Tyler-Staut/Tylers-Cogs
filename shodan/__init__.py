from .shodan import shodan


async def setup(bot):
    await bot.add_cog(shodan(bot))
