"""
V20 Training Script - Simplified Version
Train champion_v20.pth using collected MT5 data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

from gemini_v20_invest.models.alphazero_net import AlphaZeroTradingNet
from gemini_v20_invest.mcts.config_v20 import *

print("="*70)
print("🏋️ V20 CHAMPION TRAINING")
print("="*70)

# Training data
DATA_DIR = Path("gemini_v20_invest/training/data")
DATASETS = {
    'DowJones': DATA_DIR / 'DowJones_D1_2y.csv',
    'CAC40': DATA_DIR / 'CAC40_D1_2y.csv',
    'DAX': DATA_DIR / 'DAX_D1_2y.csv',
    'SP500': DATA_DIR / 'SP500_D1_2y.csv',
}

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS_PER_ITERATION = 10
TRAINING_ITERATIONS = 100  # Start with 100 (quick test)

def load_datasets():
    """Load all training datasets"""
    global INPUT_FEATURES
    print("\n📂 Loading datasets...")
    data = {}
    
    for name, path in DATASETS.items():
        if path.exists():
            df = pd.read_csv(path)
            
            # Set features from first dataset
            if INPUT_FEATURES is None:
                INPUT_FEATURES = get_input_features(df)
                print(f"  📊 Using {len(INPUT_FEATURES)} features: {INPUT_FEATURES[:5]}...")
            
            data[name] = df
            print(f"  ✅ {name}: {len(df)} bars")
        else:
            print(f"  ❌ {name}: file not found")
    
    return data

def prepare_training_samples(datasets, samples_per_dataset=100):
    """
    Prepare training samples from datasets
    Simple version: random states with labels based on actual price movement
    """
    print(f"\n🎲 Preparing training samples ({samples_per_dataset} per dataset)...")
    
    X_train = []
    y_policy = []
    y_value = []
    
    for name, df in datasets.items():
        # Sample random points
        indices = np.random.choice(len(df) - 10, samples_per_dataset, replace=False)
        
        for idx in indices:
            # State = 26 indicators at time t
            state = df.iloc[idx][INPUT_FEATURES].values.astype(np.float32)
            
            # Check for NaN
            if np.isnan(state).any():
                continue
            
            # Target: actual price movement in next 5-10 days
            current_price = df.iloc[idx]['close']
            future_price = df.iloc[idx + 10]['close']
            pct_change = (future_price - current_price) / current_price
            
            # Policy target (simplified)
            if pct_change > 0.05:  # +5%
                action = 2  # BUY_100%
            elif pct_change > 0.02:  # +2%
                action = 1  # BUY_50%
            elif pct_change < -0.05:  # -5%
                action = 5  # SELL_100%
            elif pct_change < -0.02:  # -2%
                action = 4  # SELL_50%
            else:
                action = 6  # HOLD
            
            # One-hot policy
            policy = np.zeros(OUTPUT_POLICY_SIZE, dtype=np.float32)
            policy[action] = 1.0
            
            # Value target (normalized return)
            value = np.tanh(pct_change * 10)  # Squash to [-1, 1]
            
            X_train.append(state)
            y_policy.append(policy)
            y_value.append(value)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_policy = np.array(y_policy, dtype=np.float32)
    y_value = np.array(y_value, dtype=np.float32).reshape(-1, 1)
    
    print(f"  ✅ Generated {len(X_train)} training samples")
    
    return X_train, y_policy, y_value

def get_input_features(df):
    """Get available indicator columns from dataframe"""
    # Exclude base OHLCV columns
    exclude = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    features = [c for c in df.columns if c not in exclude and not c.startswith('fibo')]
    return features  # Use all available

INPUT_FEATURES = None  # Will be set dynamically

def train_champion():
    """Main training loop"""
    
    # 1. Load data
    datasets = load_datasets()
    if len(datasets) == 0:
        print("❌ No datasets found!")
        return
    
    # 2. Initialize network
    input_size = len(INPUT_FEATURES)
    print(f"\n🧠 Initializing AlphaZero Network...")
    print(f"   Input: {input_size} indicators")
    print(f"   Hidden: {HIDDEN_SIZE}")
    print(f"   Actions: {OUTPUT_POLICY_SIZE}")
    
    model = AlphaZeroTradingNet(
        input_dim=input_size,  # AlphaZeroTradingNet uses 'input_dim' not 'input_size'
        action_dim=OUTPUT_POLICY_SIZE
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training loop
    print(f"\n🏋️ Training for {TRAINING_ITERATIONS} iterations...")
    
    for iteration in range(TRAINING_ITERATIONS):
        # Generate fresh samples each iteration
        X_train, y_policy, y_value = prepare_training_samples(datasets, samples_per_dataset=50)
        
        # Convert to tensors
        X = torch.FloatTensor(X_train)
        policy_target = torch.FloatTensor(y_policy)
        value_target = torch.FloatTensor(y_value)
        
        # Train for multiple epochs
        model.train()
        total_loss = 0
        
        for epoch in range(EPOCHS_PER_ITERATION):
            # Shuffle
            indices = torch.randperm(len(X))
            
            # Mini-batches
            for i in range(0, len(X), BATCH_SIZE):
                batch_idx = indices[i:i+BATCH_SIZE]
                
                # Forward
                policy_pred, value_pred = model(X[batch_idx])
                
                # Loss
                policy_loss = nn.CrossEntropyLoss()(policy_pred, policy_target[batch_idx].argmax(dim=1))
                value_loss = nn.MSELoss()(value_pred, value_target[batch_idx])
                loss = policy_loss + value_loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Log progress
        if (iteration + 1) % 10 == 0:
            avg_loss = total_loss / (EPOCHS_PER_ITERATION * (len(X) // BATCH_SIZE))
            print(f"  Iteration {iteration+1}/{TRAINING_ITERATIONS} - Loss: {avg_loss:.4f}")
    
    # 4. Save champion
    save_path = Path("gemini_v20_invest/models/champions")
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_path = save_path / "champion_v20.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"\n✅ Training complete!")
    print(f"💾 Model saved: {model_path}")
    
    return model

if __name__ == "__main__":
    try:
        champion = train_champion()
        print("\n🎯 champion_v20.pth is ready for deployment!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
