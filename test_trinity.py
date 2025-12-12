import pandas as pd
import numpy as np
from gemini_v14.environment.trinity_env import TrinityEnv
from rich.console import Console

console = Console()

# Create Dummy Data
data = {
    'time': pd.date_range(start='2024-01-01', periods=100, freq='H'),
    'open': np.random.rand(100) * 100,
    'high': np.random.rand(100) * 100,
    'low': np.random.rand(100) * 100,
    'close': np.linspace(100, 110, 100), # Upward trend
    'rsi': np.random.rand(100) * 100,
    'trend': np.random.choice([-1, 0, 1], 100),
    'volatility_sma': np.random.rand(100),
    'z_score': np.random.randn(100),
    'fibo_pos': np.random.rand(100)
}
df = pd.DataFrame(data)

env = TrinityEnv(df)
obs = env.reset()

console.print("[bold]Testing TrinityEnv...[/bold]")

for i in range(10):
    mode = obs['mode']
    console.print(f"Step {i}: Mode = [cyan]{mode}[/cyan]")
    
    action = 0
    if mode == "TRADER":
        # Force Buy on step 0
        if i == 0: action = 1 
        else: action = 0
    elif mode == "MANAGER":
        # Force Close on step 5
        if i == 5: action = 2
        else: action = 0
        
    obs, reward, done, truncated, info = env.step(action)
    if 'log' in info:
        console.print(info['log'])
        
console.print("[bold green]Test Complete.[/bold green]")
