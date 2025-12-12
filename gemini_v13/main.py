import MetaTrader5 as mt5
import time
import sys
import os
import glob
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from config import PROJECT_NAME, VERSION, SYMBOLS, TIMEFRAME, DEPOSIT
from environment.mt5_env import MT5TradingEnv
from utils.dashboard import Dashboard
from utils.persistence import ModelManager
from agents.q_learning_agent import QLearningAgent
from agents.mixture_of_experts import MixtureOfExpertsAgent

console = Console()

def check_dependencies():
    """Vérifie et installe les dépendances manquantes"""
    try:
        import gymnasium
        import rich
    except ImportError:
        console.print("[bold red]Dépendances manquantes ![/bold red]")
        console.print("Veuillez exécuter : pip install gymnasium gym-anytrading rich")
        sys.exit(1)

def splash_screen():
    """Affiche l'écran de démarrage"""
    console.clear()
    title = Text(f"\n{PROJECT_NAME}\n", style="bold magenta", justify="center")
    subtitle = Text(f"Version {VERSION}\n", style="dim white", justify="center")
    
    console.print(Panel(title + subtitle, border_style="magenta"))
    time.sleep(2)

def main():
    splash_screen()
    
    # 1. Connexion MT5
    if not mt5.initialize():
        console.print(f"[bold red]Échec connexion MT5: {mt5.last_error()}[/bold red]")
        return
    
    console.print(f"[green]Connexion MT5 établie sur {mt5.account_info().server}[/green]")
    
    # 2. Initialisation des environnements (Multi-Symbol)
    envs = {}
    obs_dict = {}
    
    console.print(f"[blue]Initialisation de {len(SYMBOLS)} paires...[/blue]")
    for symbol in SYMBOLS:
        env = MT5TradingEnv(symbol, TIMEFRAME, DEPOSIT)
        envs[symbol] = env
        obs, info = env.reset()
        obs_dict[symbol] = obs
    
    # 3. Initialisation du Dashboard & Persistence
    dashboard = Dashboard()
    model_manager = ModelManager()
    
    # 4. Initialisation de l'Agent (Mixture of Experts)
    # On cherche les Champions
    champion_files = glob.glob(os.path.join("brain", "models", "Champion_*.pkl"))
    
    # Note: On utilise UN SEUL agent (Conseil des Sages) pour TOUTES les paires.
    # C'est une approche "Shared Weights". L'agent apprend des patterns universels.
    
    if len(champion_files) >= 1:
        console.print(f"[green]Chargement du Conseil des Sages ({len(champion_files)} Champions détectés)...[/green]")
        experts = []
        # On prend un env au hasard pour la shape, ils sont tous pareils
        sample_env = list(envs.values())[0]
        
        for model_path in champion_files:
            expert = QLearningAgent(sample_env.action_space.n, sample_env.observation_space.shape[0])
            model_manager.load_model(expert, model_path)
            experts.append(expert)
            
        agent = MixtureOfExpertsAgent(experts)
        console.print(f"[bold gold1]👑 {agent.name} est en ligne sur {len(SYMBOLS)} symboles.[/bold gold1]")
    else:
        console.print("[yellow]Aucun Champion trouvé. Chargement d'un agent standard...[/yellow]")
        sample_env = list(envs.values())[0]
        agent = QLearningAgent(sample_env.action_space.n, sample_env.observation_space.shape[0])
        model_manager.load_latest_model(agent)

    # 5. Boucle Principale (Live)
    console.print("[bold green]Démarrage du Live Trading Multi-Symbol...[/bold green]")
    
    with Live(dashboard.get_layout(), refresh_per_second=4, screen=True) as live:
        try:
            while True:
                for symbol in SYMBOLS:
                    env = envs[symbol]
                    obs = obs_dict[symbol]
                    
                    # 1. Action
                    action = agent.act(obs)
                    
                    # 2. Step
                    next_obs, reward, done, truncated, info = env.step(action)
                    
                    # 3. Learn (Désactivé en live)
                    
                    # 4. Update Dashboard
                    vote_info = getattr(agent, "last_vote_details", "")
                    if vote_info:
                        info['log'] = f"{info.get('log', '')} [dim]({vote_info})[/dim]"
                        
                    # On passe le symbole au dashboard pour qu'il sache quoi mettre à jour
                    dashboard.update(symbol, obs, action, reward, info, env.balance, env.equity)
                    
                    obs_dict[symbol] = next_obs
                    
                    if done:
                        obs_dict[symbol], info = env.reset()
                
                # Pause (Simulation temps réel)
                time.sleep(1) # 1 seconde par cycle complet
                
        except KeyboardInterrupt:
            pass
    
    console.print("\n[bold red]Arrêt manuel.[/bold red]")
    mt5.shutdown()

if __name__ == "__main__":
    check_dependencies()
    main()
