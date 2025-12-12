import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.alphazero_net import AlphaZeroTradingNet
from models.loss import AlphaZeroLoss

def test_phase1():
    print("🧪 Testing V19 Phase 1: Dual-Head Network & Loss")
    
    # 1. Init Network
    input_dim = 84
    action_dim = 5
    batch_size = 4
    
    net = AlphaZeroTradingNet(input_dim=input_dim, action_dim=action_dim)
    print("✅ Network Initialized")
    
    # 2. Forward Pass
    dummy_state = torch.randn(batch_size, input_dim)
    policy, value = net(dummy_state)
    
    print(f"   Input Shape: {dummy_state.shape}")
    print(f"   Policy Shape: {policy.shape}")
    print(f"   Value Shape: {value.shape}")
    
    assert policy.shape == (batch_size, action_dim), "Policy shape mismatch"
    assert value.shape == (batch_size, 1), "Value shape mismatch"
    assert torch.allclose(policy.sum(dim=1), torch.ones(batch_size)), "Policy softmax failed"
    
    print("✅ Forward Pass OK")
    
    # 3. Loss Calculation
    loss_fn = AlphaZeroLoss()
    
    # Dummy Targets
    target_policy = torch.softmax(torch.randn(batch_size, action_dim), dim=-1)
    target_value = torch.randn(batch_size, 1)
    
    loss, p_loss, v_loss = loss_fn(policy, value, target_policy, target_value)
    
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   Policy Loss: {p_loss.item():.4f}")
    print(f"   Value Loss: {v_loss.item():.4f}")
    
    assert not torch.isnan(loss), "Loss is NaN"
    
    print("✅ Loss Calculation OK")
    
    # 4. Backward Pass (Gradient Check)
    loss.backward()
    
    has_grad = False
    for param in net.parameters():
        if param.grad is not None:
            has_grad = True
            break
            
    assert has_grad, "No gradients computed"
    print("✅ Backward Pass OK")
    
    print("\n🎉 Phase 1 Complete: Dual-Head Network is functional!")

if __name__ == "__main__":
    test_phase1()
