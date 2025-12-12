import unittest
import torch
import shutil
import os
import numpy as np
from MuZero.config import MuZeroConfig
from MuZero.agents.muzero_agent import MuZeroAgent
from MuZero.training.replay_buffer import GameHistory

class MockEnv:
    def __init__(self, symbol="EURUSD", lookback=1000):
        self.observation_space_shape = (84,)
        self.action_space_n = 5
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        return np.zeros(self.observation_space_shape, dtype=np.float32), {}
        
    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(84).astype(np.float32)
        reward = np.random.randn()
        done = self.step_count >= 10  # Short episode
        info = {}
        return obs, reward, done, False, info

class TestMuZeroIntegration(unittest.TestCase):
    def setUp(self):
        self.config = MuZeroConfig()
        self.config.num_simulations = 5
        self.config.batch_size = 2
        self.config.window_size = 10
        self.config.training_steps = 1
        self.config.results_path = "MuZero/results"
        self.config.weights_path = "MuZero/weights"

    def test_agent_initialization(self):
        agent = MuZeroAgent(self.config)
        self.assertIsNotNone(agent.network)
        self.assertIsNotNone(agent.replay_buffer)

    def test_self_play(self):
        agent = MuZeroAgent(self.config)
        env = MockEnv()
        
        reward = agent.play_game(env)
        self.assertIsInstance(reward, float)
        self.assertEqual(len(agent.replay_buffer.buffer), 1)
        
        game = agent.replay_buffer.buffer[0]
        self.assertIsInstance(game, GameHistory)
        self.assertGreater(len(game.actions), 0)

    def test_training_step(self):
        agent = MuZeroAgent(self.config)
        env = MockEnv()
        
        # Fill buffer
        agent.play_game(env)
        agent.play_game(env)
        
        # Train
        optimizer = torch.optim.SGD(agent.network.parameters(), lr=0.01)
        agent.network.train()
        
        obs_batch, action_batch, reward_target, value_target, policy_target = agent.replay_buffer.sample_batch()
        
        # Just check shapes
        self.assertEqual(obs_batch.shape[0], self.config.batch_size)
        self.assertEqual(action_batch.shape[0], self.config.batch_size)
        
        # Run inference
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(agent.device)
        hidden_state, policy, value = agent.network.initial_inference(obs_tensor)
        
        self.assertEqual(hidden_state.shape, (self.config.batch_size, self.config.hidden_state_size))
        self.assertEqual(policy.shape, (self.config.batch_size, self.config.action_space_size))
        self.assertEqual(value.shape, (self.config.batch_size, 1))

if __name__ == '__main__':
    unittest.main()
