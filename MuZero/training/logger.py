import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class MuZeroLogger:
    def __init__(self, results_path="MuZero/results"):
        self.results_path = results_path
        os.makedirs(results_path, exist_ok=True)
        
        # Timestamp pour cette session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(results_path, f"training_log_{self.session_id}.json")
        
        # TensorBoard Writer
        log_dir = os.path.join(results_path, "runs", self.session_id)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"📊 TensorBoard logging to: {log_dir}")
        
        # Métriques
        self.metrics = {
            "steps": [],
            "loss": [],
            "value_loss": [],
            "policy_loss": [],
            "reward_loss": [],
            "avg_reward": [],
            "games_played": 0
        }
        
        # PERSISTENCE MAGIC: Load previous history if available
        self.history_file = os.path.join(results_path, "metrics_history.json")
        self._load_history()
        
    def log_training_step(self, step, loss, value_loss, policy_loss, reward_loss):
        """Log une étape d'entraînement"""
        self.metrics["steps"].append(step)
        self.metrics["loss"].append(float(loss))
        self.metrics["value_loss"].append(float(value_loss))
        self.metrics["policy_loss"].append(float(policy_loss))
        self.metrics["reward_loss"].append(float(reward_loss))
        
        # TensorBoard
        self.writer.add_scalar("Loss/Total", float(loss), step)
        self.writer.add_scalar("Loss/Value", float(value_loss), step)
        self.writer.add_scalar("Loss/Policy", float(policy_loss), step)
        self.writer.add_scalar("Loss/Reward", float(reward_loss), step)
        
        # Sauvegarde incrémentale
        self._save_metrics()
        self._save_history() # Also save to global history file
        
        # Affichage console
        print(f"Step {step}: Loss={loss:.4f} (V:{value_loss:.4f}, P:{policy_loss:.4f}, R:{reward_loss:.4f})")
        
    def log_game(self, reward):
        """Log une partie jouée"""
        self.metrics["games_played"] += 1
        
        # Calcul reward moyen (sur les 100 dernières parties)
        if "rewards" not in self.metrics:
            self.metrics["rewards"] = []
        self.metrics["rewards"].append(float(reward))
        
        # Moyenne glissante
        recent_rewards = self.metrics["rewards"][-100:]
        avg = np.mean(recent_rewards)
        self.metrics["avg_reward"].append(avg)
        
        # TensorBoard
        self.writer.add_scalar("Reward/Game", float(reward), self.metrics["games_played"])
        self.writer.add_scalar("Reward/Average100", float(avg), self.metrics["games_played"])
        
    
    def _save_metrics(self):
        """Sauvegarde les métriques session (debug)"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def _save_history(self):
        """Sauvegarde l'historique GLOBAL pour la continuité des graphes"""
        with open(self.history_file, 'w') as f:
            json.dump(self.metrics, f, indent=2) # Overwrite global file with current cumulative state

    def _load_history(self):
        """Charge l'historique précédent pour continuer la courbe"""
        if os.path.exists(self.history_file):
            print(f"📈 Loading metrics history from {self.history_file}...")
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    
                    # Merge checks
                    if "steps" in data and len(data["steps"]) > 0:
                        last_step = data["steps"][-1]
                        print(f"   Resuming graph from Step {last_step}")
                        self.metrics = data
            except Exception as e:
                print(f"⚠️ Failed to load metrics history: {e}")
        else:
            print("🆕 No metrics history found. Starting new graph.")
    
    def plot_training(self):
        """Génère des graphiques de training"""
        if len(self.metrics["steps"]) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'MuZero Training Metrics - {self.session_id}', fontsize=16)
        
        # Loss totale
        axes[0, 0].plot(self.metrics["steps"], self.metrics["loss"], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss détaillées
        axes[0, 1].plot(self.metrics["steps"], self.metrics["value_loss"], 'r-', label='Value Loss', alpha=0.7)
        axes[0, 1].plot(self.metrics["steps"], self.metrics["policy_loss"], 'g-', label='Policy Loss', alpha=0.7)
        axes[0, 1].plot(self.metrics["steps"], self.metrics["reward_loss"], 'orange', label='Reward Loss', alpha=0.7)
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Loss Components')
        axes[0, 1].set_title('Loss Breakdown')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward moyen
        if len(self.metrics["avg_reward"]) > 0:
            axes[1, 0].plot(self.metrics["avg_reward"], 'purple', linewidth=2)
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1, 0].set_xlabel('Games Played')
            axes[1, 0].set_ylabel('Average Reward (last 100 games)')
            axes[1, 0].set_title('Reward Evolution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution des rewards
        if "rewards" in self.metrics and len(self.metrics["rewards"]) > 0:
            axes[1, 1].hist(self.metrics["rewards"], bins=50, color='teal', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='k', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('Reward')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Reward Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarde
        plot_file = os.path.join(self.results_path, f"training_plot_{self.session_id}.png")
        plt.savefig(plot_file, dpi=150)
        print(f"📊 Chart saved: {plot_file}")
        plt.close()
        
    def print_summary(self):
        """Affiche un résumé"""
        if len(self.metrics["steps"]) == 0:
            print("No training data yet.")
            return
            
        print("\n" + "="*60)
        print("📊 TRAINING SUMMARY")
        print("="*60)
        print(f"Session: {self.session_id}")
        print(f"Training Steps: {len(self.metrics['steps'])}")
        print(f"Games Played: {self.metrics['games_played']}")
        
        if len(self.metrics["loss"]) > 0:
            print(f"\nLoss Stats:")
            print(f"  Current: {self.metrics['loss'][-1]:.4f}")
            print(f"  Best: {min(self.metrics['loss']):.4f}")
            print(f"  Average: {np.mean(self.metrics['loss']):.4f}")
        
        if "rewards" in self.metrics and len(self.metrics["rewards"]) > 0:
            print(f"\nReward Stats:")
            print(f"  Latest Avg (100 games): {self.metrics['avg_reward'][-1]:.2f}")
            print(f"  Best Game: {max(self.metrics['rewards']):.2f}")
            print(f"  Worst Game: {min(self.metrics['rewards']):.2f}")
            print(f"  Overall Average: {np.mean(self.metrics['rewards']):.2f}")
        
        print("="*60 + "\n")
