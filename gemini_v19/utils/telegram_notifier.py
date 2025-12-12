import os
import asyncio
from datetime import datetime
from telegram import Bot
from telegram.error import TelegramError
import MetaTrader5 as mt5

class TelegramNotifier:
    """
    Handles all Telegram notifications for V19 AlphaZero trading system.
    """
    def __init__(self):
        # Load credentials from environment
        from dotenv import load_dotenv
        load_dotenv()
        
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token or not self.chat_id:
            print("⚠️ Telegram credentials not found in .env file")
            self.enabled = False
        else:
            self.bot = Bot(token=self.bot_token)
            self.enabled = True
            print(f"✅ Telegram Notifier Initialized (Chat ID: {self.chat_id})")
    
    def _send_message(self, text):
        """Send message with error handling"""
        if not self.enabled:
            return
            
        try:
            # Run async code in sync context
            asyncio.run(self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode='HTML'
            ))
        except TelegramError as e:
            print(f"❌ Telegram Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
    
    def send_trade_alert(self, symbol, action, volume, price, confidence, value_estimate, context=None):
        """Send instant alert when opening a position"""
        emoji = "🚀" if action == "BUY" else "🔻"
        
        message = f"""
{emoji} <b>V19 TRADE ALERT</b>

<b>Symbol:</b> {symbol}
<b>Action:</b> {action} {volume:.2f} @ {price:.5f}
<b>Confidence:</b> {confidence*100:.1f}% (MCTS)
<b>Value Est:</b> {value_estimate:+.2f}
"""
        
        if context:
            rsi = context.get('rsi', 0)
            stoch = context.get('stoch_k', 0)
            bb = context.get('bb_b', 0)
            message += f"\n<b>Context:</b>\nRSI: {int(rsi)} | Stoch: {int(stoch*100)} | BB: {bb:.1f}"
        
        message += f"\n\n<i>Time: {datetime.now().strftime('%H:%M:%S')}</i>"
        
        self._send_message(message)
    
    def send_recap(self, traders):
        """Send summary of active positions"""
        total_pnl = 0.0
        active_count = 0
        lines = []
        
        for symbol, trader in traders.items():
            positions = mt5.positions_get(symbol=symbol, magic=1919)
            
            if positions and len(positions) > 0:
                pos = positions[0]
                pos_type = "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT"
                pos_pnl = pos.profit
                total_pnl += pos_pnl
                active_count += 1
                
                emoji = "📈" if pos_pnl >= 0 else "📉"
                lines.append(f"{emoji} {symbol}: {pos_type} {pos.volume:.2f} | {pos_pnl:+.2f}€")
            else:
                lines.append(f"⚪ {symbol}: FLAT")
        
        message = f"""
📊 <b>V19 RECAP</b>

<b>Active Positions:</b> {active_count}
<b>Total PnL:</b> {total_pnl:+.2f}€

{chr(10).join(lines)}

<i>{datetime.now().strftime('%H:%M:%S')}</i>
"""
        
        self._send_message(message)
    
    def send_stats(self, hours=24):
        """Send statistics for closed trades"""
        from datetime import timedelta
        
        # Get closed deals from MT5 history
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        deals = mt5.history_deals_get(start_time, end_time)
        
        if not deals or len(deals) == 0:
            return
        
        # Filter by magic 1919 and exit deals
        v19_deals = [d for d in deals if d.magic == 1919 and d.entry == mt5.DEAL_ENTRY_OUT]
        
        if len(v19_deals) == 0:
            return
        
        # Calculate stats
        symbol_stats = {}
        total_trades = 0
        total_wins = 0
        total_pnl = 0.0
        
        for deal in v19_deals:
            symbol = deal.symbol
            profit = deal.profit
            total_pnl += profit
            total_trades += 1
            
            if profit > 0:
                total_wins += 1
            
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
            
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['pnl'] += profit
            if profit > 0:
                symbol_stats[symbol]['wins'] += 1
        
        if total_trades == 0:
            return
        
        win_rate = (total_wins / total_trades) * 100
        sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
        
        message = f"""
📈 <b>V19 STATS (Last {hours}h)</b>

<b>Closed Trades:</b> {total_trades}
<b>Win Rate:</b> {win_rate:.1f}% ({total_wins}W / {total_trades - total_wins}L)
<b>Total PnL:</b> {total_pnl:+.2f}€

"""
        
        if sorted_symbols:
            top_symbol, top_data = sorted_symbols[0]
            top_wr = (top_data['wins'] / top_data['trades']) * 100 if top_data['trades'] > 0 else 0
            message += f"🏆 <b>Top:</b> {top_symbol}: {top_data['trades']} trades | {top_wr:.0f}% | {top_data['pnl']:+.2f}€\n\n"
            
            if len(sorted_symbols) > 1:
                worst_symbol, worst_data = sorted_symbols[-1]
                worst_wr = (worst_data['wins'] / worst_data['trades']) * 100 if worst_data['trades'] > 0 else 0
                message += f"⚠️ <b>Worst:</b> {worst_symbol}: {worst_data['trades']} trades | {worst_wr:.0f}% | {worst_data['pnl']:+.2f}€\n\n"
        
        message += f"<i>{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>"
        
        self._send_message(message)
    
    def send_startup_message(self):
        """Send notification when V19 starts"""
        message = f"""
🚀 <b>V19 AlphaZero STARTED</b>

System initialized and ready to trade.

<i>{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>
"""
        self._send_message(message)
