import numpy as np
import time
from .base_agent import BaseAgent

class Deep_Evolution_Strategy:
    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def update_weights(self, rewards, population):
        std_reward = np.std(rewards)
        if std_reward == 0:
            return # Avoid division by zero
        
        rewards = (rewards - np.mean(rewards)) / std_reward
        for index, w in enumerate(self.weights):
            A = np.array([p[index] for p in population])
            self.weights[index] = (
                w
                + self.learning_rate
                / (self.population_size * self.sigma)
                * np.dot(A.T, rewards).T
            )

class NeuralNetwork:
    def __init__(self, input_size, layer_size, output_size):
        self.input_size = input_size
        self.layer_size = layer_size
        self.output_size = output_size
        
        # Initialize weights
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
        ]

    def predict(self, inputs):
        # Simple Feed Forward
        # Layer 1
        feed = np.dot(inputs, self.weights[0])
        # Activation (ReLU or Tanh? Notebook used linear/none explicitly shown but usually Tanh/Relu)
        # The notebook code: feed = np.dot(inputs, self.weights[0]) + self.weights[-1] (bias?)
        # Let's stick to a standard architecture: Input -> Dense -> ReLU -> Dense -> Output
        
        feed = np.tanh(feed) # Tanh is good for normalized inputs
        decision = np.dot(feed, self.weights[1])
        
        # Softmax for action probabilities
        exp_decision = np.exp(decision - np.max(decision))
        probs = exp_decision / np.sum(exp_decision)
        
        return probs

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

class ESAgent(BaseAgent):
    def __init__(self, input_size, action_size, layer_size=128, population_size=15, sigma=0.1, learning_rate=0.03, name="ES_Agent"):
        super().__init__(name=name)
        self.model = NeuralNetwork(input_size, layer_size, action_size)
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.dummy_reward_func, # Not used directly in this architecture
            self.population_size,
            self.sigma,
            self.learning_rate
        )
        
        # Training State
        self.current_episode_reward = 0
        self.population = []
        self.rewards = []
        self.current_candidate_idx = 0
        
        # Generate initial population perturbations
        self._generate_population()
        
        # Set model to first candidate
        self._set_candidate_weights(0)

    def dummy_reward_func(self, weights):
        return 0

    def _generate_population(self):
        self.population = []
        for k in range(self.population_size):
            x = []
            for w in self.model.weights:
                x.append(np.random.randn(*w.shape))
            self.population.append(x)

    def _set_candidate_weights(self, idx):
        weights_population = self.es._get_weight_from_population(
            self.es.weights, self.population[idx]
        )
        self.model.set_weights(weights_population)

    def act(self, obs):
        # Normalize obs if needed? Assuming env gives normalized obs
        probs = self.model.predict(obs)
        action = np.argmax(probs)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        self.current_episode_reward += reward
        
        if done:
            # Record reward for current candidate
            self.rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
            # Move to next candidate
            self.current_candidate_idx += 1
            
            if self.current_candidate_idx >= self.population_size:
                # All candidates evaluated, update weights
                self.es.update_weights(np.array(self.rewards), self.population)
                
                # Reset for next generation
                self.rewards = []
                self.current_candidate_idx = 0
                self._generate_population()
                # print(f"[{self.name}] Generation Complete. Weights Updated.")
            
            # Set weights for the (possibly new) candidate
            self._set_candidate_weights(self.current_candidate_idx)

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.es.weights, f)

    def load(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            self.es.weights = pickle.load(f)
            self.model.set_weights(self.es.weights)
