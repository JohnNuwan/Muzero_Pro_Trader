import numpy as np
from gemini_v14.agents.base_agent import TraderAgent

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, X):
        # X shape: (1, input_size)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1) # Activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return np.tanh(self.z2) # Output activation (tanh for -1 to 1, or softmax for probs)

    def get_weights(self):
        return np.concatenate([self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten()])

    def set_weights(self, weights):
        # Reconstruct shapes
        w1_end = self.input_size * self.hidden_size
        b1_end = w1_end + self.hidden_size
        w2_end = b1_end + self.hidden_size * self.output_size
        
        self.W1 = weights[:w1_end].reshape(self.input_size, self.hidden_size)
        self.b1 = weights[w1_end:b1_end].reshape(1, self.hidden_size)
        self.W2 = weights[b1_end:w2_end].reshape(self.hidden_size, self.output_size)
        self.b2 = weights[w2_end:].reshape(1, self.output_size)

class NeuroEvolutionAgent(TraderAgent):
    """
    Neuro-Evolution Agent.
    Uses a simple MLP to decide actions.
    Weights are evolved via Genetic Algorithm (handled externally by the League).
    """
    def __init__(self, name="NeuroEvo", input_size=6, hidden_size=16, output_size=3):
        super().__init__(name)
        self.model = NeuralNetwork(input_size, hidden_size, output_size)
        
    def act(self, observation):
        # Observation: [Price, RSI, Trend, Volatility, Z-Score, Fibo]
        state = np.array(observation).reshape(1, -1)
        logits = self.model.forward(state)
        return np.argmax(logits[0]) # 0, 1, or 2

    def mutate(self, mutation_rate=0.1, sigma=0.1):
        weights = self.model.get_weights()
        mask = np.random.rand(len(weights)) < mutation_rate
        noise = np.random.randn(len(weights)) * sigma
        weights[mask] += noise[mask]
        self.model.set_weights(weights)
