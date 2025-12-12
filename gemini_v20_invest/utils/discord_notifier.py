import os
import asyncio
import discord
from datetime import datetime
from dotenv import load_dotenv

class DiscordNotifier:
    """
    Handles Discord notifications for V20-Invest.
    Uses a Bot to send Embeds to specific channels.
    """
    def __init__(self):
        load_dotenv()
        self.token = os.getenv('DISCORD_BOT_TOKEN')
        self.channel_ids = {
            'buy': int(os.getenv('DISCORD_CHANNEL_BUY', 0)),
            'sell': int(os.getenv('DISCORD_CHANNEL_SELL', 0)),
            'report': int(os.getenv('DISCORD_CHANNEL_REPORT', 0)),
            'logs': int(os.getenv('DISCORD_CHANNEL_LOGS', 0))
        }
        
        if not self.token:
            print("⚠️ DISCORD_BOT_TOKEN not found in .env")
            self.enabled = False
        else:
            self.enabled = True
            # Setup minimal bot client
            intents = discord.Intents.default()
            self.client = discord.Client(intents=intents)

    async def _send_embed(self, channel_key, embed):
        """Send embed to specific channel"""
        if not self.enabled: return
        
        channel_id = self.channel_ids.get(channel_key)
        if not channel_id:
            print(f"⚠️ Channel ID for '{channel_key}' not configured")
            return

        try:
            # Login once (simplified for script usage)
            await self.client.login(self.token)
            channel = await self.client.fetch_channel(channel_id)
            await channel.send(embed=embed)
            await self.client.close()
        except Exception as e:
            print(f"❌ Discord Error: {e}")

    def send_signal(self, signal):
        """
        Send Buy/Sell signal with rich Embed
        signal: dict with symbol, action, confidence, price, analysis...
        """
        is_buy = 'BUY' in signal['action']
        color = discord.Color.green() if is_buy else discord.Color.red()
        channel_key = 'buy' if is_buy else 'sell'
        
        embed = discord.Embed(
            title=f"{'🚀' if is_buy else '⚠️'} V20-INVEST: {signal['action']} {signal['symbol']}",
            description=f"**Confidence:** {signal['confidence']*100:.1f}%",
            color=color,
            timestamp=datetime.now()
        )
        
        embed.add_field(name="Price", value=f"{signal['price']:.2f}€", inline=True)
        embed.add_field(name="Target", value=f"{signal.get('target', 'N/A')}€", inline=True)
        embed.add_field(name="Stop Loss", value=f"{signal.get('stop_loss', 'N/A')}€", inline=True)
        
        # Analysis section
        analysis = signal.get('analysis', {})
        analysis_text = "\n".join([f"• {k}: {v}" for k, v in analysis.items()])
        embed.add_field(name="📊 Analysis", value=analysis_text or "No details", inline=False)
        
        # Reason
        if 'reason' in signal:
            embed.add_field(name="🤖 AI Reason", value=signal['reason'], inline=False)
            
        embed.set_footer(text="Gemini V20-Invest • AlphaZero Advisory")
        
        # Run async in sync context
        asyncio.run(self._send_embed(channel_key, embed))

    def send_report(self, title, content, fields=None):
        """Send generic report (e.g. Daily Dow Analysis)"""
        embed = discord.Embed(
            title=title,
            description=content,
            color=discord.Color.blue(),
            timestamp=datetime.now()
        )
        
        if fields:
            for name, value in fields.items():
                embed.add_field(name=name, value=value, inline=True)
                
        asyncio.run(self._send_embed('report', embed))

if __name__ == "__main__":
    # Test
    notifier = DiscordNotifier()
    notifier.send_signal({
        'symbol': 'AAPL',
        'action': 'BUY_50%',
        'confidence': 0.85,
        'price': 185.50,
        'target': 210.00,
        'stop_loss': 165.00,
        'analysis': {'RSI': '45', 'Trend': 'Bullish'},
        'reason': 'Strong volume + Breakout'
    })
