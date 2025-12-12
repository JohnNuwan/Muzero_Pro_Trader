import os
import threading
from datetime import datetime
from telegram import Bot
from telegram.error import TelegramError
from dotenv import load_dotenv
import asyncio
import warnings

# Suppress asyncio warnings from telegram/yfinance interaction
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Event loop is closed')

class TelegramNotifier:
    """
    Handles Telegram notifications for V20-Invest (Advisory Mode).
    Uses threading to avoid event loop conflicts with yfinance.
    """
    def __init__(self):
        load_dotenv()
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token or not self.chat_id:
            print("WARNING: Telegram credentials not found in .env file")
            self.enabled = False
        else:
            self.enabled = True
            print(f"[OK] V20 Telegram Notifier Initialized (Chat ID: {self.chat_id})")

    def _send_message(self, text):
        if not self.enabled: return
        
        def send_in_thread():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def send():
                bot = Bot(token=self.bot_token)
                try:
                    await bot.send_message(
                        chat_id=self.chat_id,
                        text=text,
                        parse_mode='HTML'
                    )
                finally:
                    # Proper cleanup
                    pass
            
            try:
                loop.run_until_complete(send())
            except Exception as e:
                print(f"❌ Telegram Error: {e}")
            finally:
                # Clean shutdown
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()
                except:
                    pass
        
        # Run in separate thread to avoid loop conflicts
        thread = threading.Thread(target=send_in_thread, daemon=True)
        thread.start()
        thread.join(timeout=10)  # Wait max 10s

    def send_investment_advice(self, signal, portfolio_data=None):
        """
        Send detailed investment advice in French with portfolio data.
        portfolio_data: dict with 'cash', 'current_position', 'suggested_volume'
        """
        is_buy = 'BUY' in signal['action']
        emoji = "🚀" if is_buy else "⚠️"
        
        message = f"""
{emoji} <b>V20-INVEST SIGNAL</b>

<b>Action:</b> {signal['symbol']}
<b>Recommandation:</b> {signal['action']}
<b>Confiance:</b> {signal['confidence']*100:.1f}%

<b>Prix:</b> {signal['price']:.2f}€
<b>Cible:</b> {signal.get('target', 'N/A'):.2f}€
<b>Stop Loss:</b> {signal.get('stop_loss', 'N/A'):.2f}€
"""
        
        # Add portfolio info if available
        if portfolio_data:
            cash = portfolio_data.get('cash', 0)
            current_qty = portfolio_data.get('current_position', 0)
            suggested_qty = portfolio_data.get('suggested_volume', 0)
            suggested_value = suggested_qty * signal['price'] if suggested_qty else 0
            
            message += f"""
<b>💰 VOTRE PORTFOLIO</b>
• Cash disponible: {cash:.2f}€
• Volume suggéré: {suggested_qty} action(s) (~{suggested_value:.2f}€)
• Position actuelle: {current_qty} action(s)
"""
        
        message += "\n<b>═══ ANALYSE ═══</b>\n"
        
        # Add analysis details
        if 'analysis' in signal:
            for k, v in signal['analysis'].items():
                message += f"• <b>{k}:</b> {v}\n"
        
        # Add AI Reason in French
        if 'reason' in signal:
            message += f"\n<b>🤖 Raison IA:</b>\n<i>{signal['reason']}</i>"
            
        message += f"\n\n<i>Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M')}</i>"
        
        self._send_message(message)

    def send_report(self, title, content):
        """Send generic report"""
        message = f"<b>{title}</b>\n\n{content}\n\n<i>{datetime.now().strftime('%H:%M')}</i>"
        self._send_message(message)

if __name__ == "__main__":
    # Test
    notifier = TelegramNotifier()
    notifier.send_investment_advice({
        'symbol': 'AAPL',
        'action': 'BUY_50%',
        'confidence': 0.88,
        'price': 185.50,
        'target': 210.00,
        'stop_loss': 165.00,
        'analysis': {'RSI': '45 (Neutral)', 'Trend': 'Bullish'},
        'reason': 'Strong volume breakout confirmed by sector analysis.'
    })
