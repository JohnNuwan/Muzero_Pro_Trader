import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroLoss(nn.Module):
    """
    AlphaZero Loss Function
    L = (z - v)^2 - π^T * log(p) + c * ||θ||^2
    Where:
        z: Self-play return (target value)
        v: Predicted value
        π: MCTS policy (target policy)
        p: Predicted policy
        c: L2 regularization constant (handled by optimizer weight_decay)
    """
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, policy_pred, value_pred, policy_target, value_target):
        """
        Calculate combined loss
        Args:
            policy_pred: Predicted policy probabilities (batch, action_dim)
            value_pred: Predicted value (batch, 1)
            policy_target: Target policy from MCTS (batch, action_dim)
            value_target: Target value from self-play (batch, 1)
        """
        # 1. Value Loss (MSE)
        # (z - v)^2
        value_loss = self.mse_loss(value_pred, value_target)
        
        # 2. Policy Loss (Cross Entropy)
        # - Σ π * log(p)
        # Clamp for numerical stability
        policy_pred = torch.clamp(policy_pred, min=1e-6, max=1.0 - 1e-6)
        policy_loss = -torch.mean(torch.sum(policy_target * torch.log(policy_pred), dim=1))
        
        # Total Loss
        total_loss = value_loss + policy_loss
        
        return total_loss, policy_loss, value_loss
