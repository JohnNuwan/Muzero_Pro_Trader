import time
import random
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

from config import PROJECT_NAME, SYMBOL, TIMEFRAME, DEPOSIT
from environment.backtest_env import MT5BacktestEnv
from utils.data_loader import DataLoader
from utils.persistence import ModelManager
from agents.q_learning_agent import QLearningAgent

console = Console()

def train_agent(agent, env, episodes=500):
    """Entraîne un agent spécifique et retourne son score final"""
    # console.print(f"[yellow]Entraînement de {agent.name}...[/yellow]")
    start_time = time.time()
    
    final_balance = 0
    best_balance = 0
    
    # On utilise une barre de progression globale dans le main, donc ici on reste silencieux ou minimaliste
import random
import time
import os
import glob
import MetaTrader5 as mt5
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from config import PROJECT_NAME, SYMBOLS, TIMEFRAME
from utils.data_loader import DataLoader
from utils.persistence import ModelManager
from environment.backtest_env import BacktestEnv
from agents.q_learning_agent import QLearningAgent

console = Console()

def get_multiplier(symbol):
    """Détermine le multiplicateur PnL correct via MT5"""
    info = mt5.symbol_info(symbol)
    if info is None:
        return 100000 # Default Forex
    
    # Heuristique basée sur le type ou les digits
    if "JPY" in symbol: return 1000 # JPY pairs (2-3 digits)
    if "XAU" in symbol: return 100  # Gold (Standard 100oz)
    if "BTC" in symbol: return 1    # Crypto (1 Coin)
    if "US30" in symbol or "GER40" in symbol or "US500" in symbol or "SPX" in symbol: return 1 # Indices (often 1 point = 1 currency unit for cash CFDs)
    
    return 100000 # Standard Forex (5 digits)

def run_league():
    console.clear()
    console.print(f"[bold magenta]{PROJECT_NAME} - EVOLUTIONARY LEAGUE (MULTI-SYMBOL) 🧬🏆[/bold magenta]")
    console.print("Objectif : Entraîner des agents sur un panier d'actifs diversifié.")
    
    # 1. Connexion MT5 & Chargement des Données
    if not mt5.initialize():
        console.print(f"[red]Erreur MT5: {mt5.last_error()}[/red]")
        return

    data_cache = {}
    multipliers = {}
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Chargement des données...", total=len(SYMBOLS))
        
        for symbol in SYMBOLS:
            loader = DataLoader()
            df = loader.load_data(symbol, TIMEFRAME)
            if df is not None:
                df = loader.add_indicators(df)
                # On garde les 2000 dernières bougies pour le training pour que ce soit rapide mais suffisant
                if len(df) > 2000:
                    df = df.iloc[-2000:].reset_index(drop=True)
                data_cache[symbol] = df
                multipliers[symbol] = get_multiplier(symbol)
            progress.advance(task)
            
    mt5.shutdown()
    
    # 2. Génération des Agents (Population Initiale)
    agents = []
    
    # Archetypes
    agents.append(QLearningAgent(4, 8, learning_rate=0.1, discount_factor=0.95, epsilon_decay=0.995, name="Archetype_Balanced"))
    agents.append(QLearningAgent(4, 8, learning_rate=0.2, discount_factor=0.90, epsilon_decay=0.99, name="Archetype_Aggressive"))
    agents.append(QLearningAgent(4, 8, learning_rate=0.05, discount_factor=0.99, epsilon_decay=0.999, name="Archetype_Conservative"))
    
    # Mutants (Q-Learning)
    for i in range(5):
        lr = round(random.uniform(0.01, 0.3), 3)
        df = round(random.uniform(0.8, 0.999), 3)
        name = f"Mutant_QL_{i+1}_LR{lr}_G{df}"
        agents.append(QLearningAgent(4, 8, learning_rate=lr, discount_factor=df, epsilon_decay=0.995, name=name))

    # Mutants (Evolution Strategy)
    from agents.es_agent import ESAgent
    agents.append(ESAgent(input_size=8, action_size=4, population_size=15, sigma=0.1, learning_rate=0.03, name="Mutant_ES_Balanced"))
    agents.append(ESAgent(input_size=8, action_size=4, population_size=20, sigma=0.2, learning_rate=0.05, name="Mutant_ES_Aggressive"))
    agents.append(ESAgent(input_size=8, action_size=4, population_size=10, sigma=0.05, learning_rate=0.01, name="Mutant_ES_Conservative"))

    # 3. Tournoi (Entraînement)
    results = []
    
    with Progress() as progress:
        main_task = progress.add_task("[bold green]Entraînement de la Ligue...", total=len(agents))
        
        for agent in agents:
            start_time = time.time()
            total_reward = 0
            
            # Chaque agent joue 500 épisodes
            # À chaque épisode, on change de symbole au hasard !
            for episode in range(500):
                symbol = random.choice(SYMBOLS)
                if symbol not in data_cache: continue
                
                df = data_cache[symbol]
                mult = multipliers[symbol]
                
                env = BacktestEnv(df, deposit=10000, multiplier=mult)
                obs, _ = env.reset()
                done = False
                
                while not done:
                    action = agent.act(obs)
                    next_obs, reward, done, truncated, _ = env.step(action)
                    agent.learn(obs, action, reward, next_obs, done)
                    obs = next_obs
                    total_reward += reward
            
            # Évaluation finale (Moyenne sur tous les symboles)
            final_score = 0
            valid_symbols = 0
            for symbol in SYMBOLS:
                if symbol not in data_cache: continue
                valid_symbols += 1
                
                env = BacktestEnv(data_cache[symbol], deposit=10000, multiplier=multipliers[symbol])
                obs, _ = env.reset()
                done = False
                while not done:
                    action = agent.act(obs) # Mode exploit
                    next_obs, _, done, _, _ = env.step(action)
                    obs = next_obs
                final_score += env.balance
            
            if valid_symbols > 0:
                final_score /= valid_symbols
            
            elapsed = time.time() - start_time
            results.append({
                "name": agent.name,
                "score": final_score,
                "time": elapsed,
                "agent": agent
            })
            
            progress.update(main_task, advance=1, description=f"Training {agent.name}...")

    # 4. Classement et Sélection
    results.sort(key=lambda x: x["score"], reverse=True)
    
    table = Table(title="🏆 LEAGUE RESULTS (Multi-Asset) 🏆")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Avg Balance", justify="right", style="green")
    table.add_column("Time", justify="right")
    
    model_manager = ModelManager()
    
    # Sauvegarde des 5 meilleurs
    for i, res in enumerate(results[:5]):
        rank = i + 1
        champion_name = f"Champion_{rank}"
        res["agent"].name = champion_name
        model_manager.save_model(res["agent"], episode="final")
        
        table.add_row(str(rank), res["name"], f"{res['score']:.2f}", f"{res['time']:.1f}s")
        
    # Affichage du reste
    for i, res in enumerate(results[5:]):
        table.add_row(str(i+6), res["name"], f"{res['score']:.2f}", f"{res['time']:.1f}s")
        
    console.print(table)
    console.print("\n[bold green]✅ Les 5 meilleurs agents Multi-Asset ont été sauvegardés ![/bold green]")

if __name__ == "__main__":
    run_league()
