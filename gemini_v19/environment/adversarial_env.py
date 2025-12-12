import random
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gemini_v19.environment.commission_trinity_env import CommissionTrinityEnv

class AdversarialEnv(CommissionTrinityEnv):
    """
    Adversarial Environment for AlphaZero Training.
    Injects noise, slippage, and gaps to simulate harsh market conditions.
    """
    def __init__(self, symbol="EURUSD", lookback=1000, adversary_strength=0.2):
        super().__init__(symbol, lookback)
        self.adversary_strength = adversary_strength
        self.adversary_active = True
        
    def step(self, action):
        # 1. Standard Step
        state, reward, done, truncated, info = super().step(action)
        
        # 2. Adversarial Intervention
        if self.adversary_active and random.random() < self.adversary_strength:
            event_type = random.choice(['slippage', 'gap', 'volatility', 'news'])
            
            if event_type == 'slippage':
                # Simulate bad fill
                # Reduce reward (if positive) or increase penalty (if negative)
                if reward > 0:
                    reward *= 0.8 # 20% profit loss due to slippage
                elif reward < 0:
                    reward *= 1.2 # 20% extra loss
                info['adversarial_event'] = 'slippage'
                
            elif event_type == 'gap':
                # Simulate price gap against position
                if self.position_size != 0:
                    gap_penalty = abs(self.position_size) * self.pip_size * 10 # 10 pips gap
                    # Adjust balance directly to simulate gap loss
                    self.balance -= gap_penalty
                    reward -= 1.0 # Penalty for getting caught in gap
                    info['adversarial_event'] = 'gap'
                    
            elif event_type == 'volatility':
                # Add noise to state observation
                noise = np.random.normal(0, 0.1, state.shape)
                state = state + noise
                info['adversarial_event'] = 'volatility'
                
            elif event_type == 'news':
                # Major news event - huge volatility
                if self.position_size != 0:
                    # 50/50 chance of huge win or huge loss
                    if random.random() < 0.5:
                        reward -= 5.0 # Stop loss hit with slippage
                        self.balance -= 100.0 # Big hit
                    else:
                        reward += 2.0 # Lucky spike
                info['adversarial_event'] = 'news'
                
        return state, reward, done, truncated, info
