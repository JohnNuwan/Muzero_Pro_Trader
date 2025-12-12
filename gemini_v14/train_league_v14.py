import sys
import os
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from gemini_v14.environment.trinity_env import TrinityEnv
from gemini_v14.utils.data_loader import DataLoader
from gemini_v14.agents.strategies.classic_traders import TurtleAgent, MovingAverageAgent
from gemini_v14.agents.strategies.neuro_evolution import NeuroEvolutionAgent
from gemini_v14.agents.strategies.rl_agents import QLearningAgent, DoubleQLearningAgent, PolicyGradientAgent, ActorCriticAgent
from gemini_v14.agents.strategies.managers import StandardManager

console = Console()

def train_league_v14():
    console.print("[bold yellow]Gemini V14 - The Trinity - CHAMPIONS LEAGUE 🏆[/bold yellow]")
    
    # 1. Load Data
    loader = DataLoader()
    # Use a sample symbol for the league (e.g. EURUSD)
    symbol = "EURUSD" 
    console.print(f"Loading data for {symbol}...")
    df = loader.load_data(symbol, timeframe="M5")
    
    if df is None:
        console.print("[red]Failed to load data.[/red]")
        return

    # 2. Initialize Environment
    env = TrinityEnv(df)
    
    # 3. Initialize Agents
    # We create a roster of all available strategies
    agents = [
        TurtleAgent("Turtle_Classic"),
        MovingAverageAgent("MA_Crossover"),
        NeuroEvolutionAgent("NeuroEvo_1"),
        QLearningAgent("QL_Tabular"),
        DoubleQLearningAgent("Double_QL"),
        PolicyGradientAgent("PG_Reinforce"),
        ActorCriticAgent("Actor_Critic")
    ]
    
    manager = StandardManager("Std_Manager")
    
    results = []
    
    # 4. Competition Loop
    for agent in agents:
        console.print(f"Training {agent.name}...", end="")
        
        total_reward = 0
        episodes = 5 # Short league for testing
        
        start_time = time.time()
        
        for ep in range(episodes):
            obs = env.reset()
            done = False
            truncated = False
            
            while not done and not truncated:
                mode = obs['mode']
                
                if mode == "TRADER":
                    action = agent.act(obs['trader_obs'])
                    # For RL agents, we might want to learn here
                    # But for this league, we just evaluate performance mostly
                    # (RL agents learn online)
                    
                elif mode == "MANAGER":
                    action = manager.act(obs['manager_obs'])
                    
                next_obs, reward, done, truncated, info = env.step(action)
                
                # RL Learning Step (if applicable)
                if mode == "TRADER" and hasattr(agent, 'learn'):
                    # Online Learning (QL, DoubleQL, ActorCritic)
                    if isinstance(agent, (QLearningAgent, ActorCriticAgent)):
                        # We need next_state. In this env, next_obs is the next state.
                        # But we need to know if next_obs['mode'] is TRADER or MANAGER?
                        # Simplified: We assume next_obs['trader_obs'] is the next state for Trader.
                        # Reward is immediate reward (0 usually) + Manager reward later?
                        # This is tricky. For now, we pass 0 reward if not done, or actual reward if done.
                        # V14 TODO: Proper credit assignment.
                        agent.learn(obs['trader_obs'], action, reward, next_obs['trader_obs'], done)
                    
                    # Episodic Learning (PG)
                    elif isinstance(agent, PolicyGradientAgent):
                        agent.store_transition(obs['trader_obs'], action, reward)
                
                obs = next_obs
                total_reward += reward
                
            # End of Episode Learning for PG
            if isinstance(agent, PolicyGradientAgent) and not isinstance(agent, ActorCriticAgent):
                agent.learn()
                
        duration = time.time() - start_time
        avg_score = total_reward / episodes
        results.append({'name': agent.name, 'score': avg_score, 'time': duration})
        console.print(f" [green]Done[/green] ({avg_score:.2f})")

    # 5. Display Results
    results.sort(key=lambda x: x['score'], reverse=True)
    
    table = Table(title="🏆 V14 LEAGUE RESULTS 🏆")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Time", justify="right", style="yellow")
    
    for i, res in enumerate(results):
        table.add_row(str(i+1), res['name'], f"{res['score']:.2f}", f"{res['time']:.2f}s")
        
    console.print(table)
    
    # Save Top 3 Winners
    console.print("[bold]Saving Top 3 Agents...[/bold]")
    if not os.path.exists("gemini_v14/models"):
        os.makedirs("gemini_v14/models")
        
    for i in range(3):
        if i < len(results):
            winner_name = results[i]['name']
            # Find the agent object
            agent = next((a for a in agents if a.name == winner_name), None)
            if agent:
                agent.save(f"gemini_v14/models/{winner_name}.pkl")

if __name__ == "__main__":
    train_league_v14()
