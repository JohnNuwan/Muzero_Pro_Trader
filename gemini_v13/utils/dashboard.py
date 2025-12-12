from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
from .charts import generate_ascii_chart
from collections import deque

class Dashboard:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.logs = []
        self.price_history = {} # {symbol: deque(maxlen=60)}
        self.symbol_stats = {} # {symbol: {price, pnl, action}}
        
        # Split layout
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10)
        )
        self.layout["main"].split_row(
            Layout(name="market_overview", ratio=1),
            Layout(name="chart", ratio=2)
        )

    def get_layout(self):
        return self.layout

    def update(self, symbol, obs, action, reward, info, balance, equity):
        # 1. Update Price History
        price = obs[0]
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=60)
        self.price_history[symbol].append(price)
        
        # 2. Update Stats
        action_str = ["HOLD", "BUY", "SELL", "CLOSE"][action]
        color = "white"
        if action == 1: color = "green"
        elif action == 2: color = "red"
        elif action == 3: color = "blue"
        
        self.symbol_stats[symbol] = {
            "price": price,
            "action": f"[{color}]{action_str}[/{color}]",
            "pnl": obs[7] # Pos Profit
        }
        
        # 3. Update Logs
        if 'log' in info and info['log']:
            self.logs.append(f"[{symbol}] {info['log']}")
            if len(self.logs) > 10:
                self.logs.pop(0)

        # --- RENDER ---
        
        # Header
        self.layout["header"].update(
            Panel(f"Gemini V13 - The Sovereign | Balance: {balance:.2f} | Equity: {equity:.2f}", style="bold magenta")
        )
        
        # Market Overview Table
        table = Table(title="Market Watch")
        table.add_column("Symbol", style="cyan")
        table.add_column("Price", style="white")
        table.add_column("Action", justify="center")
        table.add_column("PnL", justify="right")
        
        for sym, stats in self.symbol_stats.items():
            table.add_row(
                sym, 
                f"{stats['price']:.5f}", 
                stats['action'], 
                f"{stats['pnl']:.2f}"
            )
            
        self.layout["market_overview"].update(Panel(table))
        
        # Chart (Last Updated Symbol)
        chart_str = generate_ascii_chart(list(self.price_history[symbol]), height=15, width=80)
        self.layout["chart"].update(Panel(Text(chart_str, style="green"), title=f"Chart: {symbol}"))
        
        # Footer (Logs)
        log_text = "\n".join(self.logs)
        self.layout["footer"].update(Panel(log_text, title="Logs", border_style="blue"))
