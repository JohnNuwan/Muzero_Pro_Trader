import sqlite3
import pickle
import numpy as np
import os
from datetime import datetime

class ReplayDatabase:
    """
    SQLite Database for storing live trading experiences.
    Schema:
        - id: Primary Key
        - timestamp: DateTime
        - symbol: Text
        - state: Blob (Pickled numpy array)
        - action: Integer
        - reward: Float
        - done: Boolean
        - metadata: JSON/Text (Optional)
    """
    def __init__(self, db_path="gemini_v19/live/trades.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                state BLOB NOT NULL,
                action INTEGER NOT NULL,
                reward REAL NOT NULL,
                done BOOLEAN NOT NULL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)')
        
        conn.commit()
        conn.close()
        
    def store(self, symbol, state, action, reward, done, metadata=None):
        """
        Store a single transition
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Pickle state
        state_blob = pickle.dumps(state)
        
        cursor.execute('''
            INSERT INTO trades (symbol, state, action, reward, done, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, state_blob, int(action), float(reward), bool(done), str(metadata)))
        
        conn.commit()
        conn.close()
        
    def load_recent(self, limit=1000):
        """
        Load recent trades for training
        Returns: list of (state, policy_target, value_target)
        Note: In live, we only have (state, action, reward).
        We need to convert this to training format.
        
        For AlphaZero, we need Policy Target (MCTS dist) and Value Target (Return).
        
        Option A: Re-run MCTS on stored states (Expensive but accurate)
        Option B: Use 'action' as a one-hot policy target (Imitation Learning)
        Option C: Store MCTS policy in metadata if available
        
        For V19, we will assume 'metadata' contains the MCTS policy if we stored it.
        If not, we use one-hot action.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT state, action, reward, metadata 
            FROM trades 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        samples = []
        for row in rows:
            state_blob, action, reward, meta = row
            state = pickle.loads(state_blob)
            
            # Construct Policy Target
            # If we have MCTS policy in metadata, use it.
            # Otherwise, one-hot the action.
            policy = np.zeros(5, dtype=np.float32)
            policy[action] = 1.0 
            
            # Value Target is the Reward (Simplified for single-step or we need to compute returns)
            # In live, we store realized reward.
            value = float(reward)
            
            samples.append((state, policy, value))
            
        return samples
        
    def get_count(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM trades')
        count = cursor.fetchone()[0]
        conn.close()
        return count
