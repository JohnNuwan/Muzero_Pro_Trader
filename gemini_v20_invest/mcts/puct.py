import numpy as np

def calculate_puct(node, c_puct=1.5):
    """
    Calculate PUCT score for a node
    PUCT = Q(s,a) + U(s,a)
    U(s,a) = c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(node))
    """
    if node.parent is None:
        return 0.0
        
    # Exploitation term
    Q = node.Q
    
    # Exploration term
    P = node.prior
    N_parent = node.parent.visit_count
    N_node = node.visit_count
    
    U = c_puct * P * np.sqrt(N_parent) / (1 + N_node)
    
    return Q + U

def select_child(node, c_puct=1.5):
    """
    Select the child with the highest PUCT score
    """
    best_score = -float('inf')
    best_action = None
    best_child = None
    
    for action, child in node.children.items():
        score = calculate_puct(child, c_puct)
        
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
            
    return best_action, best_child
