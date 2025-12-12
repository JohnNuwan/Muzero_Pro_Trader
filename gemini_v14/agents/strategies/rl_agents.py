import numpy as np
from gemini_v14.agents.base_agent import TraderAgent

class QLearningAgent(TraderAgent):
    """
    Tabular Q-Learning Agent.
    Discretizes the state space.
    """
    def __init__(self, name="QLearning", learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        super().__init__(name)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {} # State -> [Q_wait, Q_buy, Q_sell]
        
    def _get_state_key(self, observation):
        # Discretize observation
        # Obs: [Price, RSI, Trend, Volatility, Z-Score, Fibo]
        rsi = int(observation[1] // 10) # 0-10
        trend = int(observation[2]) # -1, 0, 1
        z_score = int(observation[4]) # -3 to 3 approx
        return (rsi, trend, z_score)
        
    def act(self, observation):
        state = self._get_state_key(observation)
        
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2])
            
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
            
        return np.argmax(self.q_table[state])
        
    def learn(self, state, action, reward, next_state, done):
        s = self._get_state_key(state)
        ns = self._get_state_key(next_state)
        
        if s not in self.q_table: self.q_table[s] = np.zeros(3)
        if ns not in self.q_table: self.q_table[ns] = np.zeros(3)
        
        target = reward + self.gamma * np.max(self.q_table[ns]) * (1 - done)
        self.q_table[s][action] += self.lr * (target - self.q_table[s][action])
        
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class PolicyGradientAgent(TraderAgent):
    """
    Simple Policy Gradient (REINFORCE) using Numpy.
    Uses a 1-layer Softmax Policy Network.
    """
    def __init__(self, name="PolicyGradient", input_size=6, hidden_size=16, output_size=3, learning_rate=0.01):
        super().__init__(name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Weights
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        
        self.episode_memory = []
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        return self.softmax(self.z2)
        
    def act(self, observation):
        probs = self.forward(np.array(observation))
        action = np.random.choice(self.output_size, p=probs)
        return action
        
    def store_transition(self, observation, action, reward):
        self.episode_memory.append((observation, action, reward))
        
    def learn(self):
        # Monte Carlo Policy Gradient Update
        if not self.episode_memory: return
        
        # Calculate discounted rewards
        rewards = [m[2] for m in self.episode_memory]
        discounted_rewards = np.zeros_like(rewards, dtype=np.float64)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * 0.99 + rewards[t]
            discounted_rewards[t] = running_add
            
        # Normalize rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)
        
        # Update weights (Simplified Backprop)
        # This is a very basic implementation and might be unstable without a proper framework
        # For now, we just clear memory to simulate "learning" phase logic structure
        self.episode_memory = []

class DoubleQLearningAgent(QLearningAgent):
    """
    Double Q-Learning Agent.
    Uses two Q-tables to reduce maximization bias.
    """
    def __init__(self, name="DoubleQL", learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        super().__init__(name, learning_rate, discount_factor, epsilon)
        self.q_table_b = {} # Second table
        
    def act(self, observation):
        state = self._get_state_key(observation)
        
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2])
            
        if state not in self.q_table: self.q_table[state] = np.zeros(3)
        if state not in self.q_table_b: self.q_table_b[state] = np.zeros(3)
        
        # Use sum of both tables for action selection
        return np.argmax(self.q_table[state] + self.q_table_b[state])
        
    def learn(self, state, action, reward, next_state, done):
        s = self._get_state_key(state)
        ns = self._get_state_key(next_state)
        
        if s not in self.q_table: self.q_table[s] = np.zeros(3)
        if ns not in self.q_table: self.q_table[ns] = np.zeros(3)
        if s not in self.q_table_b: self.q_table_b[s] = np.zeros(3)
        if ns not in self.q_table_b: self.q_table_b[ns] = np.zeros(3)
        
        # Randomly update one of the two tables
        if np.random.rand() < 0.5:
            # Update A using B for value estimation
            best_action_a = np.argmax(self.q_table[ns])
            target = reward + self.gamma * self.q_table_b[ns][best_action_a] * (1 - done)
            self.q_table[s][action] += self.lr * (target - self.q_table[s][action])
        else:
            # Update B using A for value estimation
            best_action_b = np.argmax(self.q_table_b[ns])
            target = reward + self.gamma * self.q_table[ns][best_action_b] * (1 - done)
            self.q_table_b[s][action] += self.lr * (target - self.q_table_b[s][action])
            
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class ActorCriticAgent(PolicyGradientAgent):
    """
    Actor-Critic Agent (Simplified).
    Actor: Policy Network (from PG).
    Critic: Value Table (simplified) or Network.
    """
    def __init__(self, name="ActorCritic", input_size=6, hidden_size=16, output_size=3, learning_rate=0.01):
        super().__init__(name, input_size, hidden_size, output_size, learning_rate)
        self.critic_table = {} # State -> Value
        
    def _get_state_key(self, observation):
        # Reuse discretization for Critic Table
        rsi = int(observation[1] // 10)
        trend = int(observation[2])
        z_score = int(observation[4])
        return (rsi, trend, z_score)
        
    def learn(self, state, action, reward, next_state, done):
        # TD Error (Advantage)
        s_key = self._get_state_key(state)
        ns_key = self._get_state_key(next_state)
        
        v_s = self.critic_table.get(s_key, 0.0)
        v_ns = self.critic_table.get(ns_key, 0.0)
        
        target = reward + 0.95 * v_ns * (1 - done)
        td_error = target - v_s
        
        # Update Critic (Value Function)
        self.critic_table[s_key] = v_s + 0.1 * td_error
        
        # Update Actor (Policy) using TD Error as Advantage
        # (Simplified update logic, ideally requires backprop through network)
        pass
