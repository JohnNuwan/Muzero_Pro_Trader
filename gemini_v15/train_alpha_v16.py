import os
import sys
import time
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rich.console import Console
from rich.table import Table

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from gemini_v15.environment.commission_trinity_env import CommissionTrinityEnv
import MetaTrader5 as mt5

# Configuration
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", 
    "BTCUSD", "ETHUSD", 
    "XAUUSD", 
    "US30.cash", "GER40.cash", "US500.cash", "US100.cash"
]
TRAIN_STEPS = 100000 # Increased for better convergence
MODELS_DIR = os.path.join(current_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

console = Console()

def train_alpha_v16():
    if not mt5.initialize():
        console.print("❌ MT5 Init Failed", style="bold red")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold yellow]🚀 Alpha-V16 Training Pipeline on {device.upper()}[/bold yellow]")
    
    for symbol in SYMBOLS:
        console.print(f"\n[bold cyan]🧠 Processing {symbol}...[/bold cyan]")
        
        # 1. Determine Data Split
        # We need to know the length of data to split it.
        # We'll create a temporary env to check length.
        temp_env = CommissionTrinityEnv(symbol, lookback=2000)
        total_len = len(temp_env.m1_data)
        train_end = int(total_len * 0.8)
        
        console.print(f"Data Points: {total_len} | Train: 0-{train_end} | Val: {train_end}-{total_len}")
        
        # 2. Define Population (Archetypes)
        population = [
            {
                "name": "Sniper",
                "params": {"learning_rate": 0.0001, "gamma": 0.95, "ent_coef": 0.005},
                "desc": "Conservative, high precision"
            },
            {
                "name": "Grinder",
                "params": {"learning_rate": 0.0003, "gamma": 0.99, "ent_coef": 0.01},
                "desc": "Balanced, standard PPO"
            },
            {
                "name": "Berserker",
                "params": {"learning_rate": 0.001, "gamma": 0.90, "ent_coef": 0.05},
                "desc": "Aggressive, high risk/reward"
            }
        ]
        
        candidates = []
        
        # 3. Train Population
        for agent_cfg in population:
            name = agent_cfg["name"]
            console.print(f"  Training [bold]{name}[/bold] ({agent_cfg['desc']})...", end="")
            
            try:
                # Create Train Env
                env = DummyVecEnv([lambda: CommissionTrinityEnv(symbol, lookback=2000, start_index=0, end_index=train_end)])
                env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
                
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    verbose=0,
                    device="auto",
                    **agent_cfg["params"]
                )
                
                model.learn(total_timesteps=TRAIN_STEPS)
                candidates.append({"name": name, "model": model})
                console.print(" [green]Done[/green]")
                
            except Exception as e:
                console.print(f" [red]Failed: {e}[/red]")

        # 4. The League (Validation)
        console.print(f"  🏆 [bold]The League (Validation)[/bold]...")
        best_score = -float('inf')
        winner = None
        
        table = Table(title=f"League Results: {symbol}")
        table.add_column("Agent", style="cyan")
        table.add_column("Return", style="green")
        table.add_column("Sharpe", style="magenta")
        table.add_column("PF", style="blue")
        table.add_column("DD", style="red")
        table.add_column("Score", style="yellow")
        
        for candidate in candidates:
            name = candidate["name"]
            model = candidate["model"]
            
            # Create Validation Env
            val_env = CommissionTrinityEnv(symbol, lookback=2000, start_index=train_end, end_index=total_len - 1)
            
            obs, _ = val_env.reset()
            done = False
            
            total_reward = 0
            wins = 0
            trades = 0
            gross_profit = 0.0
            gross_loss = 0.0
            equity_curve = [val_env.initial_balance]
            returns_series = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = val_env.step(action)
                
                # Accurate Win Rate & PF Check
                if 'realized_pnl' in info and info['realized_pnl'] != 0:
                    trades += 1
                    pnl = info['realized_pnl']
                    if pnl > 0: 
                        wins += 1
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                
                # Track daily/step returns for Sharpe
                # Approximation: Step return
                step_return = (info['equity'] - equity_curve[-1]) / equity_curve[-1]
                returns_series.append(step_return)
                
                total_reward += reward
                equity_curve.append(info['equity'])
                
            # Calculate Metrics
            final_equity = equity_curve[-1]
            total_return = (final_equity - val_env.initial_balance) / val_env.initial_balance * 100
            
            # Profit Factor
            profit_factor = 0.0
            if gross_loss == 0:
                profit_factor = 10.0 if gross_profit > 0 else 0.0
            else:
                profit_factor = gross_profit / gross_loss
                
            # Sharpe Ratio (Annualized assuming 1 min steps)
            # 252 trading days * 1440 minutes = 362880 steps/year? 
            # Let's just use a relative Sharpe for comparison
            returns_np = np.array(returns_series)
            std_dev = np.std(returns_np)
            mean_ret = np.mean(returns_np)
            sharpe = 0.0
            if std_dev > 1e-9:
                sharpe = (mean_ret / std_dev) * np.sqrt(252 * 1440) # Annualized approximation
            
            # Max Drawdown
            peak = equity_curve[0]
            max_dd = 0
            for x in equity_curve:
                if x > peak: peak = x
                dd = (peak - x) / peak
                if dd > max_dd: max_dd = dd
            
            max_dd_pct = max_dd * 100
            win_rate = (wins / trades * 100) if trades > 0 else 0
            
            # Advanced Score Formula
            # We want High Sharpe, High PF, Low DD
            # Score = Sharpe + PF - (DD / 10)
            score = sharpe + profit_factor - (max_dd_pct * 0.1)
            
            # Penalize negative returns heavily
            if total_return < 0:
                score -= 5.0
            
            table.add_row(name, f"{total_return:.2f}%", f"{sharpe:.2f}", f"{profit_factor:.2f}", f"{max_dd_pct:.2f}%", f"{score:.2f}")
            
            if score > best_score:
                best_score = score
                winner = candidate
        
        console.print(table)
        
        # 5. Save Winner
        if winner:
            console.print(f"  👑 Champion: [bold green]{winner['name']}[/bold green]")
            path = os.path.join(MODELS_DIR, f"ppo_v16_{symbol}")
            winner["model"].save(path)
            console.print(f"  ✅ Saved to {path}")
        else:
            console.print("  ❌ No winner found.")

    mt5.shutdown()
    console.print("\n[bold green]🎉 Alpha-V16 Training Complete![/bold green]")

if __name__ == "__main__":
    train_alpha_v16()
