
import ray
import torch

class RayConfig:
    def __init__(self):
        # Hardware Specs: Ryzen 4000 (assume 6-8 cores), RTX 2060 Max-Q (6GB VRAM)
        
        # 1. Resource Allocation
        self.num_workers = 4           # Leave 2-4 cores for system/trainer
        self.num_gpus_per_worker = 0.1 # Share GPU among workers (inference)
        self.num_gpus_trainer = 0.4    # Reserve 40% GPU for training
        
        # 2. Memory Management
        self.object_store_memory = 2 * 1024 * 1024 * 1024 # 2GB for shared objects
        
        # 3. Batching (Critical for 2060 Max-Q)
        self.inference_batch_size = 32 # Group worker requests
        self.training_batch_size = 64
        
    def print_summary(self):
        print(f"🚀 Ray Config for Ryzen/RTX 2060:")
        print(f"   - Workers: {self.num_workers}")
        print(f"   - GPU per Worker: {self.num_gpus_per_worker*100}%")
        print(f"   - GPU Trainer: {self.num_gpus_trainer*100}%")
