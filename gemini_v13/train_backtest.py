import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
import MetaTrader5 as mt5 # Juste pour l'init si besoin de télécharger les data

from config import PROJECT_NAME, VERSION, SYMBOL, TIMEFRAME, DEPOSIT
from environment.backtest_env import MT5BacktestEnv
from utils.dashboard import Dashboard
from utils.persistence import ModelManager
from utils.data_loader import DataLoader
from agents.q_learning_agent import QLearningAgent

console = Console()

def main():
    console.clear()
    console.print(Panel(f"[bold magenta]{PROJECT_NAME} - TRAINING MODE[/bold magenta]", border_style="magenta"))
    
    # 1. Chargement des Données
    loader = DataLoader()
    # On doit init MT5 juste pour télécharger les data si elles n'existent pas
    if not mt5.initialize():
        console.print("[red]MT5 requis pour le téléchargement initial des données[/red]")
    
    df = loader.load_data(SYMBOL, TIMEFRAME)
    if df is None:
        return
        
    # Ajout des indicateurs (Les Yeux)
    df = loader.add_indicators(df)
    console.print(f"[cyan]Indicateurs calculés (RSI, SMA, Trend). Prêt.[/cyan]")
        
    # 2. Init Environnement & Agent
    env = MT5BacktestEnv(df, DEPOSIT)
    
    # Agent
    agent = QLearningAgent(action_space_size=env.action_space.n, observation_space_size=env.observation_space.shape[0])
    model_manager = ModelManager()
    model_manager.load_latest_model(agent) # On reprend l'entraînement précédent
    
    # Dashboard simplifié pour le training
    dashboard = Dashboard()
    
    console.print(f"[yellow]Démarrage de l'entraînement sur {len(df)} bougies...[/yellow]")
    
    # 3. Boucle d'Entraînement (Rapide)
    episodes = 50 # Nombre de fois qu'on passe sur tout l'historique
    
    try:
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.act(obs)
                next_obs, reward, done, truncated, info = env.step(action)
                
                agent.learn(obs, action, reward, next_obs, done)
                
                obs = next_obs
                total_reward += reward
                
                # Update Dashboard moins fréquent pour la vitesse (tous les 100 steps)
                if env.current_step % 100 == 0:
                    dashboard.update(obs, action, reward, info, env.balance, env.equity)
                    print(f"Episode {ep+1}/{episodes} | Step {env.current_step}/{env.max_steps} | Balance: {env.balance:.2f}", end="\r")
            
            # Fin d'épisode
            model_manager.save_model(agent, f"train_ep{ep+1}")
            console.print(f"\n[green]Episode {ep+1} terminé. Reward Total: {total_reward:.2f} | Balance Finale: {env.balance:.2f}[/green]")
            
    except KeyboardInterrupt:
        console.print("\n[red]Entraînement interrompu[/red]")
        model_manager.save_model(agent, "interrupted")

if __name__ == "__main__":
    main()
