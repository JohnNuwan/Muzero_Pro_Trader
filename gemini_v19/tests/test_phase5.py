import unittest
import numpy as np
import sys
import os
import shutil
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live.replay_db import ReplayDatabase
from live.continuous_learner import ContinuousLearner

class TestPhase5(unittest.TestCase):
    def setUp(self):
        self.test_db_path = "gemini_v19/tests/test_trades.db"
        self.db = ReplayDatabase(db_path=self.test_db_path)
        
    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        if os.path.exists("gemini_v19/tests/tmp_models"):
            shutil.rmtree("gemini_v19/tests/tmp_models")
            
    def test_database_ops(self):
        print("🧪 Testing V19 Phase 5: Database Ops")
        
        # 1. Store
        state = np.random.randn(84).astype(np.float32)
        self.db.store(
            symbol="EURUSD",
            state=state,
            action=1,
            reward=0.5,
            done=False,
            metadata="{'mcts_policy': [0.1, 0.8, 0.1, 0, 0]}"
        )
        
        # 2. Count
        count = self.db.get_count()
        self.assertEqual(count, 1)
        
        # 3. Load
        samples = self.db.load_recent(limit=10)
        self.assertEqual(len(samples), 1)
        
        s, p, v = samples[0]
        self.assertEqual(s.shape, (84,))
        self.assertEqual(len(p), 5)
        self.assertEqual(v, 0.5)
        
        print("✅ Database Store/Load OK")
        
    def test_continuous_learner(self):
        print("🧪 Testing V19 Phase 5: Continuous Learner")
        
        # 1. Populate DB with dummy data
        for _ in range(110): # Need > 100 to trigger retrain
            self.db.store(
                symbol="EURUSD",
                state=np.random.randn(84).astype(np.float32),
                action=np.random.randint(0, 5),
                reward=np.random.randn(),
                done=False
            )
            
        # 2. Init Learner
        learner = ContinuousLearner(model_path="gemini_v19/tests/tmp_models/champion.pth")
        # Inject test db
        learner.db = self.db
        
        # 3. Run Retrain
        result = learner.retrain()
        
        self.assertTrue(result, "Retraining should succeed with enough data")
        self.assertTrue(os.path.exists("gemini_v19/tests/tmp_models/champion_candidate.pth"))
        
        print("✅ Continuous Learner Retrain OK")
        
        print("\n🎉 Phase 5 Complete: Continuous Learning functional!")

if __name__ == "__main__":
    unittest.main()
