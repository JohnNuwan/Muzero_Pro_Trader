
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import random
from multiprocessing import Pool, cpu_count
from MuZero.config import MuZeroConfig
from MuZero.agents.muzero_agent import MuZeroAgent
from MuZero.training.logger import MuZeroLogger
from MuZero.environment.commission_trinity_env import CommissionTrinityEnv
from MuZero.training.replay_buffer import GameHistory

def play_game_worker(args):
    """
    Worker function that plays one game.
    Must be at module level for multiprocessing to pickle it.
    """
    config, symbol, seed, weights_dict = args
    
    # Initialize MT5 in this worker process
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print(f"⚠️ Worker {seed}: MT5 initialization failed")
    except Exception as e:
        print(f"⚠️ Worker {seed}: MT5 error: {e}")
    
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create agent
    agent = MuZeroAgent(config)
    
    # Load weights if provided
    if weights_dict:
        agent.network.load_state_dict(weights_dict)
    
    # Create environment
    env = CommissionTrinityEnv(symbol=symbol)
    
    # Play game and return history
    game_history = agent.play_game_ray(env)
    
    # Return total reward for logging
    total_reward = sum(game_history.rewards)
    
    # Shutdown MT5 in this worker
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
    except:
        pass
    
    return game_history, total_reward

def train_parallel():
    # Initialize MT5 Connection
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            print("✅ MT5 Connected")
            account_info = mt5.account_info()
            if account_info:
                print(f"   Account: {account_info.login}, Balance: {account_info.balance}")
        else:
            print("⚠️ MT5 initialization failed. Will use synthetic data.")
    except Exception as e:
        print(f"⚠️ MT5 not available: {e}. Will use synthetic data.")
    
    config = MuZeroConfig()
    agent = MuZeroAgent(config)
    logger = MuZeroLogger(config.results_path)
    
    # Optimizer and Scaler
    optimizer = optim.SGD(agent.network.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)
    
    # Ensure directories exist
    os.makedirs(config.results_path, exist_ok=True)
    os.makedirs(config.weights_path, exist_ok=True)
    
    # Determine number of workers
    num_workers = min(4, cpu_count() - 1)  # Leave one core for main process
    print(f"🚀 Starting Parallel Training with {num_workers} workers")
    print(f"Symbols: {config.symbols}")
    print(f"Device: {agent.device}")
    if torch.cuda.is_available():
        print("🚀 Mixed Precision (AMP) Enabled")
    
    # Collect initial data
    buffer_save_path = os.path.join(config.results_path, "replay_buffer_initial.pkl")
    
    if os.path.exists(buffer_save_path):
        print(f"📦 Loading saved replay buffer from {buffer_save_path}...")
        import pickle
        with open(buffer_save_path, 'rb') as f:
            agent.replay_buffer = pickle.load(f)
        print(f"✅ Loaded {len(agent.replay_buffer.buffer)} games from saved buffer!")
    else:
        print(f"⏳ Collecting initial data ({config.batch_size} games)...")
        
        games_needed = config.batch_size
        games_collected = 0
        
        with Pool(num_workers) as pool:
            while games_collected < games_needed:
                # Prepare arguments for workers
                batch_size = min(num_workers, games_needed - games_collected)
                worker_args = [
                    (config, random.choice(config.symbols), random.randint(0, 100000), None)
                    for _ in range(batch_size)
                ]
                
                # Play games in parallel
                results = pool.map(play_game_worker, worker_args)
                
                # Add to replay buffer
                for game_history, total_reward in results:
                    agent.replay_buffer.save_game(game_history)
                    games_collected += 1
                
                print(f"Collected {games_collected}/{games_needed} games...")
        
        # Save the buffer for next time
        print(f"💾 Saving replay buffer to {buffer_save_path}...")
        import pickle
        with open(buffer_save_path, 'wb') as f:
            pickle.dump(agent.replay_buffer, f)
        print(f"✅ Buffer saved!")
    
    print(f"✅ Initial data ready!")
    print("🔥 Starting Training Loop...")
    
    # Training Loop
    for step in range(config.training_steps):
        # 1. Sample Batch
        (obs_batch, action_batch, reward_target, value_target, policy_target) = agent.replay_buffer.sample_batch()
        
        # Convert to tensor
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(agent.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(agent.device)
        reward_target = torch.tensor(reward_target, dtype=torch.float32).to(agent.device)
        value_target = torch.tensor(value_target, dtype=torch.float32).to(agent.device)
        policy_target = torch.tensor(policy_target, dtype=torch.float32).to(agent.device)
        
        # 2. Train Step
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            # Initial Inference (returns: hidden_state, policy, value)
            hidden_state, policy_logits, value = agent.network.initial_inference(obs_batch)
            
            total_loss = 0
            policy_loss_sum = 0
            value_loss_sum = 0
            reward_loss_sum = 0
            
            # Unroll
            for k in range(config.num_unroll_steps):
                # Policy Loss (War Machine: 2.0)
                policy_pred = F.softmax(policy_logits, dim=1)
                current_policy_loss = torch.sum(-policy_target[:, k] * torch.log(policy_pred + 1e-8)) / config.batch_size
                total_loss += current_policy_loss * 2.0
                policy_loss_sum += current_policy_loss.item()
                
                # Value Loss (War Machine: 0.5)
                current_value_loss = F.mse_loss(value.squeeze(-1), value_target[:, k])
                total_loss += current_value_loss * 0.5
                value_loss_sum += current_value_loss.item()
                
                # Dynamics Step
                action = action_batch[:, k]
                # One-hot encode action
                action_onehot = F.one_hot(action, num_classes=config.action_space_size).float()
                
                # Recurrent inference (returns: next_hidden_state, reward, policy, value)
                hidden_state, reward, policy_logits, value = agent.network.recurrent_inference(hidden_state, action_onehot)
                
                # Reward Loss (War Machine: 2.0) - Only for k > 0
                if k > 0:
                    current_reward_loss = F.mse_loss(reward.squeeze(-1), reward_target[:, k])
                    total_loss += current_reward_loss * 2.0
                    reward_loss_sum += current_reward_loss.item()
        
        # Backward
        scaler.scale(total_loss).backward()
        
        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 3. Log every 10 steps
        if step % 10 == 0:
            logger.log_training_step(
                step,
                total_loss.item(),
                value_loss_sum,
                policy_loss_sum,
                reward_loss_sum
            )
        
        # 4. Checkpoint
        if step % config.checkpoint_interval == 0:
            agent.save(os.path.join(config.weights_path, f"checkpoint_{step}.pth"))
            logger.plot_training()
            
    print("✅ Training Complete!")

if __name__ == "__main__":
    train_parallel()
