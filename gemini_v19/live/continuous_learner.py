import torch
import torch.optim as optim
import os
import sys
import copy
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gemini_v19.models.alphazero_net import AlphaZeroTradingNet
from gemini_v19.models.loss import AlphaZeroLoss
from gemini_v19.live.replay_db import ReplayDatabase
from gemini_v19.utils.config import NETWORK_CONFIG, CONTINUOUS_LEARNING_CONFIG, TRAINING_CONFIG
from gemini_v19.utils.selfplay_config import SELF_PLAY_CONFIG

# New Components
from gemini_v19.training.self_play import SelfPlayEngine
from gemini_v19.training.hybrid_trainer import HybridTrainer
from gemini_v19.training.tournament import TournamentManager
from gemini_v19.training.simulated_market import SimulatedMarket

class ContinuousLearner:
    """
    Handles nightly retraining using AlphaZero/MuZero hybrid pipeline.
    """
    def __init__(self, model_path="gemini_v19/models/champions/current_champion.pth"):
        self.model_path = model_path
        self.db = ReplayDatabase()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = CONTINUOUS_LEARNING_CONFIG
        self.sp_config = SELF_PLAY_CONFIG
        
    def retrain(self):
        print(f"🌙 Starting Enhanced Nightly Retraining at {os.getcwd()}...")
        
        # 1. Load Current Model (Champion)
        current_model = AlphaZeroTradingNet(**NETWORK_CONFIG).to(self.device)
        if os.path.exists(self.model_path):
            current_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("   Loaded current champion.")
        else:
            print("   No champion found. Starting from scratch.")
            
        # 2. Self-Play Generation (Exploration)
        print("🎮 Phase 1: Self-Play Generation...")
        sp_engine = SelfPlayEngine(current_model, self.device)
        self_play_data = sp_engine.generate_games() # Uses config for n_games
        print(f"   Generated {len(self_play_data)} self-play positions")
        
        # 3. Load Real Trades (Exploitation)
        print("📊 Phase 2: Loading Real Trades...")
        real_data = self.db.load_recent(limit=self.config['lookback_trades'])
        print(f"   Loaded {len(real_data)} real trades")
        
        if len(real_data) < 100:
            print("⚠️ Not enough real data. Proceeding with mostly self-play.")
            
        # 4. Hybrid Training
        print("🧠 Phase 3: Hybrid Training...")
        # Create candidate (clone of current)
        candidate_model = copy.deepcopy(current_model)
        
        trainer = HybridTrainer(self.device)
        candidate_model = trainer.train(
            candidate_model,
            self_play_data,
            real_data,
            epochs=self.config['retrain_epochs']
        )
        
        # 5. Tournament Validation
        print("⚔️ Phase 4: Tournament Validation...")
        tournament = TournamentManager(self.device)
        win_rate, sharpe_new, sharpe_old = tournament.run_tournament(
            candidate_model,
            current_model
        )
        
        print(f"   Results: Win Rate={win_rate:.1%}, Sharpe New={sharpe_new:.2f}, Sharpe Old={sharpe_old:.2f}")
        
        # 6. Decision & Deployment
        threshold_wr = self.sp_config['win_rate_threshold']
        threshold_sharpe = self.sp_config['sharpe_improvement']
        
        # Check if Sharpe improved enough (handle negative sharpe or zero)
        sharpe_improved = False
        if sharpe_old <= 0:
            if sharpe_new > sharpe_old + 0.1: # Absolute improvement
                sharpe_improved = True
        else:
            if sharpe_new > sharpe_old * threshold_sharpe:
                sharpe_improved = True
                
        if win_rate >= threshold_wr and sharpe_improved:
            print("🏆 NEW CHAMPION! Deploying...")
            self._deploy(candidate_model)
        else:
            print("🛡️ Champion retains title.")
            
        print("✅ Nightly Cycle Complete.")
        
    def _deploy(self, model):
        """Save new champion"""
        # Backup current
        if os.path.exists(self.model_path):
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.model_path.replace(".pth", f"_backup_{timestamp}.pth")
            
            import shutil
            shutil.copy2(self.model_path, backup_path)
            print(f"   📦 Backup saved to {backup_path}")
            
        # Save new
        torch.save(model.state_dict(), self.model_path)
        print(f"   Saved to {self.model_path}")
        print(f"   ⚠️ Restart main_v19_multi.py to load the new model!")

if __name__ == "__main__":
    learner = ContinuousLearner()
    learner.retrain()
