import discord
from hybrid_analysis.hybrid_analysis import RequestType
from redbot.core import commands, Config
import pkgutil
import shodan


class shodan(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(
            self,
            identifier=1,
            force_registration=True,
        )
        self.api_token = None
        self.modules = []

    async def initialize(self):
        api_token = await self.bot.get_shared_api_tokens("shodan")
        self.api_token = api_token

    @commands.group()
    async def shodan(self, ctx):
        api_token = self.api_token
        if api_token.get("api_key") is None:
            return await ctx.send("The Shodan API key has not been set.")
        # Use the API key to access content as you normally would

    @shodan.command()
    async def modules(self, ctx):
        """
        List all available modules
        """
        embed = discord.Embed(
            title="Current Supported Functions", color=discord.Color.blue()
        )
        for module in self.modules:
            embed.add_field(name=module, value="\u200b", inline=False)

        await ctx.send(embed=embed)

    @shodan.command()
    async def search(self, ctx, *, query: str):
        """Search Shodan with a query and return results in an embed."""
        api_key = self.api_token.get("api_key")
        if not api_key:
            return await ctx.send("The Shodan API key has not been set.")

        try:
            api = shodan.Shodan(api_key)
            results = api.search(query)
            embed = discord.Embed(
                title=f"Shodan Search Results for: {query}", color=discord.Color.green()
            )
            for result in results["matches"][
                :5
            ]:  # Limit to first 5 results for brevity
                embed.add_field(
                    name=result["ip_str"],
                    value=f"**Port:** {result['port']}\n**Org:** {result.get('org', 'N/A')}\n**OS:** {result.get('os', 'N/A')}",
                    inline=False,
                )
            await ctx.send(embed=embed)
        except shodan.APIError as e:
            await ctx.send(f"Error: {e}")


def setup(bot):
    bot.add_cog(shodan(bot))
