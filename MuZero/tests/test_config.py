import unittest
from MuZero.config import MuZeroConfig

class TestMuZeroConfig(unittest.TestCase):
    def setUp(self):
        self.config = MuZeroConfig()
        self.config.training_steps = 1000  # Set for predictable testing

    def test_visit_softmax_temperature_fn(self):
        # Scenario 1: trained_steps < 0.5 * training_steps
        self.assertEqual(self.config.visit_softmax_temperature_fn(0), 1.0)
        self.assertEqual(self.config.visit_softmax_temperature_fn(499), 1.0)

        # Scenario 2: 0.5 * training_steps <= trained_steps < 0.75 * training_steps
        self.assertEqual(self.config.visit_softmax_temperature_fn(500), 0.5)
        self.assertEqual(self.config.visit_softmax_temperature_fn(749), 0.5)

        # Scenario 3: trained_steps >= 0.75 * training_steps
        self.assertEqual(self.config.visit_softmax_temperature_fn(750), 0.25)
        self.assertEqual(self.config.visit_softmax_temperature_fn(1000), 0.25)
        self.assertEqual(self.config.visit_softmax_temperature_fn(2000), 0.25)

if __name__ == '__main__':
    unittest.main()
