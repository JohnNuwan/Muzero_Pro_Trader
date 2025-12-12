
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import random
from MuZero.config_v3 import MuZeroConfigV3
from MuZero.agents.muzero_agent import MuZeroAgent
from MuZero.training.logger import MuZeroLogger
from MuZero.environment.commission_trinity_env_v3 import CommissionTrinityEnvV3

def train_muzero_v3():
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
    
    config = MuZeroConfigV3()
    agent = MuZeroAgent(config)
    logger = MuZeroLogger(config.results_path)
    
    # Optimizer and Scaler for AMP
    optimizer = optim.SGD(agent.network.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # Learning Rate Scheduler (Reduce LR every 500 steps by 0.7x for faster adaptation)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)
    
    # Ensure directories exist
    os.makedirs(config.results_path, exist_ok=True)
    os.makedirs(config.weights_path, exist_ok=True)
    
    print(f"🚀 Starting MuZero V3 Training (Pro Trader Edition)")
    print(f"\n📊 Configuration:")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Training Steps: {config.training_steps}")
    print(f"   Symbols: {len(config.symbols)}")
    print(f"   Device: {agent.device}")
    
    print(f"\n💰 Reward Structure:")
    print(f"   ✅ Quality Trade (+1%): +{config.quality_trade_bonus} pts")
    print(f"   ✅ Final Growth (+10%): +{config.final_growth_bonus} pts")
    print(f"   ✅ SLBE Activation: +{config.slbe_activation_bonus} pts")
    print(f"   ✅ Smart SPLIT (profit): +{config.split_with_profit_bonus} pts")
    print(f"   ✅ Big CLOSE (+2%): +{config.close_big_winner_bonus} pts")
    
    print(f"\n⚠️ Penalties:")
    print(f"   ❌ Time in Drawdown: -{config.drawdown_time_penalty_rate} pts/20 steps")
    print(f"   ❌ Drawdown >5%: -{config.max_drawdown_penalty} pts")
    print(f"   ❌ Loss: {config.loss_penalty_multiplier}× asymmetric penalty")
    print()
    print()
    # if torch.cuda.is_available():
    #     print("🚀 Mixed Precision (AMP) Enabled")
    
    # Create V3 environments with enhanced features
    envs = {}
    for symbol in config.symbols:
        try:
            print(f"Loading V3 environment for {symbol}...")
            envs[symbol] = CommissionTrinityEnvV3(
                symbol=symbol,
                lookback=1000,
                quality_trade_multiplier=config.quality_trade_bonus,
                enable_final_growth_bonus=False,  # DISABLED: Was causing +130 reward saturation
                final_growth_threshold=config.final_growth_threshold,
                final_growth_bonus=config.final_growth_bonus,
                drawdown_penalty_rate=config.drawdown_time_penalty_rate,
                max_drawdown_penalty=config.max_drawdown_penalty,
                loss_penalty_multiplier=config.loss_penalty_multiplier
            )
        except Exception as e:
            print(f"⚠️ Failed to load real data for {symbol}: {e}")
            print(f"🔄 Switching to Synthetic Environment for {symbol}")
            from MuZero.training.synthetic_env import SyntheticCommissionTrinityEnv
            envs[symbol] = SyntheticCommissionTrinityEnv(symbol=symbol)
    
    # Resume Training Logic
    start_step = 0
    
    # 1. Load Replay Buffer
    buffer_path = os.path.join(config.results_path, "replay_buffer_v3.pkl")
    if os.path.exists(buffer_path):
        print(f"📦 Loading V3 Replay Buffer: {buffer_path}")
        try:
            import pickle
            with open(buffer_path, 'rb') as f:
                agent.replay_buffer = pickle.load(f)
            print(f"✅ Loaded {len(agent.replay_buffer.buffer)} games from buffer")
        except Exception as e:
            print(f"⚠️ Failed to load buffer: {e}")
            
    # 2. Load Checkpoint
    # Find latest checkpoint
    checkpoints = [f for f in os.listdir(config.weights_path) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
        checkpoint_path = os.path.join(config.weights_path, latest_checkpoint)
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        
        try:
            agent.network.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
            start_step = int(latest_checkpoint.split("_")[1].split(".")[0])
            print(f"✅ Resumed at step {start_step}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            
    best_mean_reward = -float('inf')

    for step in range(start_step, config.training_steps):
        # 1. Self-Play (Collect Data)
        if len(agent.replay_buffer.buffer) < config.batch_size:
            print(f"Collecting initial data... ({len(agent.replay_buffer.buffer)}/{config.batch_size})")
            symbol = random.choice(config.symbols)
            reward = agent.play_game(envs[symbol])
            logger.log_game(reward)
            print(f"  ✅ Game #{len(agent.replay_buffer.buffer)} completed on {symbol} (Reward: {reward:.2f})")
            continue
            
        # Play one game periodically to keep buffer fresh
        if step % 5 == 0:
            symbol = random.choice(config.symbols)
            total_reward = agent.play_game(envs[symbol])
            logger.log_game(total_reward)
            
            # Save Best Model
            if len(logger.metrics["avg_reward"]) > 0:
                current_mean_reward = logger.metrics["avg_reward"][-1]
                if current_mean_reward > best_mean_reward:
                    best_mean_reward = current_mean_reward
                    torch.save(agent.network.state_dict(), os.path.join(config.weights_path, "best_model_v3.pth"))
                    print(f"🔥 V3 New Best Model Saved! Reward: {best_mean_reward:.2f}")
            
            print(f"Step {step}: Played game on {symbol}, Reward: {total_reward:.2f}")
            
        # 2. Hybrid Learning: Ingest Live Experience
        if step % 10 == 0:
            live_buffer_dir = "MuZero/results_v3/live_buffer"
            if os.path.exists(live_buffer_dir):
                live_files = [f for f in os.listdir(live_buffer_dir) if f.endswith(".pkl")]
                if live_files:
                    import pickle
                    print(f"📥 Found {len(live_files)} live games. Ingesting...")
                    for file in live_files:
                        try:
                            file_path = os.path.join(live_buffer_dir, file)
                            with open(file_path, "rb") as f:
                                live_game = pickle.load(f)
                            
                            # Add to Replay Buffer
                            agent.replay_buffer.save_game(live_game)
                            
                            # Remove file after ingestion to prevent duplicates
                            os.remove(file_path)
                            print(f"  ✅ Ingested & Removed: {file}")
                        except Exception as e:
                            print(f"  ⚠️ Failed to ingest {file}: {e}")
            
        # Save Replay Buffer periodically (every 100 steps)
        if step % 100 == 0:
            import pickle
            buffer_path = os.path.join(config.results_path, "replay_buffer_v3.pkl")
            with open(buffer_path, 'wb') as f:
                pickle.dump(agent.replay_buffer, f)
            
        # 2. Training
        agent.network.train()
        optimizer.zero_grad()
        
        # Sample batch
        obs_batch, action_batch, reward_target, value_target, policy_target = agent.replay_buffer.sample_batch()
        
        # Convert to tensors
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(agent.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(agent.device)
        reward_target = torch.tensor(reward_target, dtype=torch.float32).to(agent.device)
        value_target = torch.tensor(value_target, dtype=torch.float32).to(agent.device)
        policy_target = torch.tensor(policy_target, dtype=torch.float32).to(agent.device)
        
        # Mixed Precision Context (DISABLED for stability)
        # with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        
        # Initial Inference
        hidden_state, policy_pred, value_pred = agent.network.initial_inference(obs_batch)
        
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        reward_loss_sum = 0
        
        # Unroll
        for k in range(config.num_unroll_steps):
            # 1. Loss at step k
            # Policy Loss (VERY High Weight - Force Learning)
            # Add epsilon to prevent log(0)
            policy_pred = torch.clamp(policy_pred, min=1e-8, max=1.0)
            current_policy_loss = torch.sum(-policy_target[:, k] * torch.log(policy_pred)) / config.batch_size
            total_loss += current_policy_loss * 2.0  # Weight: 2.0 (Balanced)
            policy_loss_sum += current_policy_loss.item()
            
            # Value Loss (Lower Weight - Already learning well)
            current_value_loss = F.mse_loss(value_pred.squeeze(-1), value_target[:, k])
            total_loss += current_value_loss * 0.5  # Weight: 0.5 (Balanced)
            value_loss_sum += current_value_loss.item()
            
            # Reward Loss (MSE) - Only for k > 0 (Dynamics)
            if k > 0:
                current_reward_loss = F.mse_loss(reward_pred.squeeze(-1), reward_target[:, k])
                total_loss += current_reward_loss * 2.0  # Weight: 2.0 (increased from 1.0)
                reward_loss_sum += current_reward_loss.item()
            
            # 2. Dynamics Step
            action = action_batch[:, k]
            action_one_hot = F.one_hot(action, num_classes=config.action_space_size).float()
            
            # Recurrent Inference
            hidden_state, reward_pred, policy_pred, value_pred = agent.network.recurrent_inference(hidden_state, action_one_hot)
            
        # Backward (No Scaler)
        total_loss.backward()
        
        # Check for NaN gradients
        has_nan = False
        for param in agent.network.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan = True
                    break
        
        if has_nan:
            print(f"⚠️ NaN Gradient detected at step {step}! Skipping update.")
            optimizer.zero_grad()
        else:
            # Gradient Clipping (Prevent Explosion)
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()  # Update learning rate
        
        # Log metrics
        if not has_nan:
            logger.log_training_step(step, total_loss.item(), value_loss_sum, policy_loss_sum, reward_loss_sum)
        
        if step % 100 == 0:
            print(f"Step {step}: Loss: {total_loss.item():.4f}")
            
        if step % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.weights_path, f"checkpoint_{step}.pth")
            torch.save(agent.network.state_dict(), checkpoint_path)
            
            # Generate charts
            logger.plot_training()
            logger.print_summary()

if __name__ == "__main__":
    train_muzero_v3()
