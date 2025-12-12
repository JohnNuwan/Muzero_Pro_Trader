import numpy as np
import random
from .base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, action_space_size, observation_space_size, name="QLearningExpert", learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(name=name)
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size # Note: Q-Table needs discrete states, we might need to discretize continuous obs
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Pour simplifier au début, on utilise un dictionnaire pour la Q-Table (Sparse Q-Learning)
        # Car l'espace d'observation est continu (Prix, Z-Score...)
        self.q_table = {} 

    def _get_state_key(self, obs):
        """Discretize continuous observation to use as Q-Table key"""
        # Obs: [Price, RSI, Trend, Volatility, Z-Score, Fibo_Pos, Pos_Type, Pos_Profit]
        
        # 1. RSI Binning (0-30, 30-70, 70-100)
        rsi = obs[1]
        if rsi < 30: rsi_bin = 0 # Oversold
        elif rsi > 70: rsi_bin = 2 # Overbought
        else: rsi_bin = 1 # Neutral
        
        # 2. Trend (Direct)
        trend_bin = int(obs[2]) # -1, 0, 1
        
        # 3. Z-Score Binning (Mean Reversion)
        z_score = obs[4]
        if z_score > 2: z_bin = 2 # High (Sell signal)
        elif z_score < -2: z_bin = 0 # Low (Buy signal)
        else: z_bin = 1 # Normal
        
        # 4. Fibo Binning (Position in range)
        fibo = obs[5]
        if fibo < 0.2: fibo_bin = 0 # Low Range
        elif fibo > 0.8: fibo_bin = 2 # High Range
        else: fibo_bin = 1 # Mid Range
        
        # 5. Position (Direct)
        pos_type = int(obs[6]) # -1, 0, 1
        
        # 6. Profit Binning
        profit = obs[7]
        if profit < -10: profit_bin = -1
        elif profit > 10: profit_bin = 1
        else: profit_bin = 0
        
        return (rsi_bin, trend_bin, z_bin, fibo_bin, pos_type, profit_bin)

    def act(self, obs):
        state_key = self._get_state_key(obs)
        
        # Exploration
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        # Exploitation
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
            
        return np.argmax(self.q_table[state_key])

    def learn(self, obs, action, reward, next_obs, done):
        state_key = self._get_state_key(obs)
        next_state_key = self._get_state_key(next_obs)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space_size)
            
        # Q-Learning Formula
        # Q(s,a) = Q(s,a) + lr * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Decay Epsilon
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
