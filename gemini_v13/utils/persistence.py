import os
from datetime import datetime
from rich.console import Console

console = Console()

import pickle
import glob

class ModelManager:
    def __init__(self, base_dir="brain/models"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
    def save_model(self, agent, episode):
        filename = f"{self.base_dir}/{agent.name}_ep{episode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        try:
            if hasattr(agent, 'save'):
                # Use agent's own save method if available (e.g. ESAgent)
                agent.save(filename)
            else:
                # Default to Q-Table pickle (e.g. QLearningAgent)
                with open(filename, 'wb') as f:
                    pickle.dump(agent.q_table, f)
            # console.print(f"[cyan]Modèle sauvegardé : {filename}[/cyan]")
            return filename
        except Exception as e:
            console.print(f"[red]Erreur sauvegarde : {e}[/red]")
            return None
        
    def load_model(self, agent, filename):
        if os.path.exists(filename):
            try:
                if hasattr(agent, 'load'):
                    # Use agent's own load method
                    agent.load(filename)
                else:
                    # Default to Q-Table pickle
                    with open(filename, 'rb') as f:
                        agent.q_table = pickle.load(f)
                console.print(f"[green]Modèle chargé : {filename}[/green]")
                return True
            except Exception as e:
                console.print(f"[red]Erreur chargement : {e}[/red]")
                return False
        else:
            console.print(f"[red]Fichier introuvable : {filename}[/red]")
            return False

    def load_latest_model(self, agent):
        """Charge le dernier modèle disponible pour cet agent"""
        search_pattern = f"{self.base_dir}/{agent.name}_*.pkl"
        files = glob.glob(search_pattern)
        if not files:
            console.print(f"[yellow]Aucun modèle trouvé pour {agent.name}[/yellow]")
            return False
            
        latest_file = max(files, key=os.path.getctime)
        return self.load_model(agent, latest_file)
