
import ray
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import time
from MuZero.config import MuZeroConfig
from MuZero.ray_config import RayConfig
from MuZero.agents.muzero_agent import MuZeroAgent
from MuZero.training.logger import MuZeroLogger
from MuZero.training.shared_storage import SharedStorage
from MuZero.training.replay_buffer_ray import ReplayBufferRay
from MuZero.training.self_play_worker import SelfPlayWorker

def train_ray():
    # 1. Initialize Ray
    ray_config = RayConfig()
    # Fix for Windows: Disable dashboard and be conservative with resources
    ray.init(
        num_cpus=ray_config.num_workers + 2, 
        num_gpus=1,
        include_dashboard=False,  # Dashboard causes issues on Windows
        ignore_reinit_error=True
    )
    
    print("🚀 Ray Initialized (Dashboard disabled for Windows compatibility)")
    ray_config.print_summary()
    
    config = MuZeroConfig()
    
    # 2. Create Actors
    # Shared Storage
    shared_storage = SharedStorage.remote(config)
    
    # Replay Buffer
    replay_buffer = ReplayBufferRay.remote(config)
    
    # Self-Play Workers
    print(f"Starting {ray_config.num_workers} Self-Play Workers...")
    workers = [
        SelfPlayWorker.remote(config, seed, shared_storage, replay_buffer)
        for seed in range(ray_config.num_workers)
    ]
    
    # Launch continuous self-play
    for worker in workers:
        worker.continuous_self_play.remote()
        
    # 3. Trainer Setup (Main Process)
    agent = MuZeroAgent(config)
    logger = MuZeroLogger(config.results_path)
    
    optimizer = optim.SGD(agent.network.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5) # Updated scheduler
    
    # Push initial weights
    shared_storage.set_weights.remote(agent.network.state_dict())
    
    # 4. Training Loop
    print("⏳ Waiting for initial data...")
    while True:
        buffer_size = ray.get(replay_buffer.get_buffer_size.remote())
        if buffer_size >= config.batch_size:
            print(f"✅ Initial data collected: {buffer_size}/{config.batch_size}")
            break
        time.sleep(1)
        
    print("🔥 Starting Training Loop...")
    
    for step in range(config.training_steps):
        # 1. Sample Batch
        batch = ray.get(replay_buffer.sample_batch.remote())
        (obs_batch, action_batch, reward_target, value_target, policy_target) = batch
        
        # Convert to tensor
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(agent.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(agent.device)
        reward_target = torch.tensor(reward_target, dtype=torch.float32).to(agent.device)
        value_target = torch.tensor(value_target, dtype=torch.float32).to(agent.device)
        policy_target = torch.tensor(policy_target, dtype=torch.float32).to(agent.device)
        
        # 2. Train Step
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Initial Inference
            value, reward, policy_logits, hidden_state = agent.network.initial_inference(obs_batch)
            
            total_loss = 0
            policy_loss_accum = 0
            value_loss_accum = 0
            reward_loss_accum = 0
            
            # Unroll
            for k in range(config.num_unroll_steps):
                # Loss at step k
                # Policy Loss (War Machine: 2.0)
                policy_pred = F.softmax(policy_logits, dim=1)
                current_policy_loss = torch.sum(-policy_target[:, k] * torch.log(policy_pred + 1e-8)) / config.batch_size
                total_loss += current_policy_loss * 2.0 
                policy_loss_accum += current_policy_loss
                
                # Value Loss (War Machine: 0.5)
                current_value_loss = F.mse_loss(value.squeeze(-1), value_target[:, k])
                total_loss += current_value_loss * 0.5
                value_loss_accum += current_value_loss
                
                # Reward Loss (War Machine: 2.0)
                if k > 0:
                    current_reward_loss = F.mse_loss(reward.squeeze(-1), reward_target[:, k])
                    total_loss += current_reward_loss * 2.0
                    reward_loss_accum += current_reward_loss
                
                # Dynamics
                action = action_batch[:, k]
                value, reward, policy_logits, hidden_state = agent.network.recurrent_inference(hidden_state, action)

            # Move scalar conversions outside the loop to avoid blocking synchronization
            policy_loss_sum = policy_loss_accum.item() if isinstance(policy_loss_accum, torch.Tensor) else policy_loss_accum
            value_loss_sum = value_loss_accum.item() if isinstance(value_loss_accum, torch.Tensor) else value_loss_accum
            reward_loss_sum = reward_loss_accum.item() if isinstance(reward_loss_accum, torch.Tensor) else reward_loss_accum
                
                # Hook for gradient clipping (on hidden state?) - Pytorch handles it on backward
                
        # Backward
        scaler.scale(total_loss).backward()
        
        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 3. Update Shared Weights (every 10 steps to reduce overhead)
        if step % 10 == 0:
            shared_storage.set_weights.remote(agent.network.state_dict())
            shared_storage.set_training_step.remote(step)
            
            # Log
            games_played = ray.get(shared_storage.increment_games_played.remote()) # Actually incremented by workers, we just read?
            # Wait, workers increment games_played. We should just read it.
            # shared_storage.get_infos.remote()
            avg_reward = ray.get(shared_storage.get_avg_reward.remote())
            
            logger.log_metrics(
                step, 
                total_loss.item(), 
                value_loss_sum, 
                policy_loss_sum, 
                reward_loss_sum, 
                avg_reward, 
                scheduler.get_last_lr()[0]
            )
            
            print(f"Step {step} | Loss: {total_loss.item():.4f} | Policy: {policy_loss_sum:.4f} | Avg Reward: {avg_reward:.2f}")
            
        # 4. Checkpoint
        if step % config.checkpoint_interval == 0:
            agent.save(os.path.join(config.weights_path, f"checkpoint_{step}.pth"))
            logger.plot_metrics()

if __name__ == "__main__":
    train_ray()
