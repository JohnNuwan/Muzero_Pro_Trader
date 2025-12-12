
import torch
import sys
import os

def check_gpu():
    print("="*60)
    print("🔍 GPU Verification for MuZero")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available. Training will run on CPU (Slow).")
        return
        
    print(f"✅ CUDA is available!")
    print(f"   Device Count: {torch.cuda.device_count()}")
    print(f"   Current Device: {torch.cuda.current_device()}")
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    
    # Memory
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(f"   Total Memory: {t / 1e9:.2f} GB")
    print(f"   Reserved Memory: {r / 1e9:.2f} GB")
    print(f"   Allocated Memory: {a / 1e9:.2f} GB")
    
    # Capabilities
    cap = torch.cuda.get_device_capability(0)
    print(f"   Compute Capability: {cap[0]}.{cap[1]}")
    
    if cap[0] >= 7:
        print("   🚀 Tensor Cores Available (Mixed Precision supported)")
    else:
        print("   ⚠️ Old GPU architecture (Mixed Precision might not be effective)")
        
    # CuDNN
    print(f"   CuDNN Version: {torch.backends.cudnn.version()}")
    if torch.backends.cudnn.is_available():
        print("   ✅ CuDNN is available")
        print("   💡 Optimization: Setting torch.backends.cudnn.benchmark = True")
    
    # Test Tensor Creation
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.matmul(x, x)
        print("   ✅ Tensor operations working on GPU")
    except Exception as e:
        print(f"   ❌ Error performing GPU operations: {e}")

if __name__ == "__main__":
    check_gpu()
