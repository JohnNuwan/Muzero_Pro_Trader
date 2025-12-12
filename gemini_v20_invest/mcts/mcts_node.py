import numpy as np

class MCTSNode:
    """
    Node in the Monte Carlo Search Tree
    """
    def __init__(self, state, parent=None, prior_p=0.0, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action taken to reach this node
        self.children = {}    # {action: MCTSNode}
        
        # Stats
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior_p  # P(s,a) from policy network
        
    @property
    def Q(self):
        """Mean action value: W/N"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    def is_expanded(self):
        return len(self.children) > 0
        
    def __repr__(self):
        return f"Node(Action={self.action}, Visits={self.visit_count}, Q={self.Q:.2f}, Prior={self.prior:.2f})"
