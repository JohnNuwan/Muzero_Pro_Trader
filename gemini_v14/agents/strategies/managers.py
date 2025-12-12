from gemini_v14.agents.base_agent import ManagerAgent

class StandardManager(ManagerAgent):
    """
    Standard Risk Manager.
    Uses fixed rules for SL, TP, and Break-Even.
    """
    def __init__(self, name="StandardManager", sl_pct=0.01, tp_pct=0.02, be_activation=0.01, split_on_be=False):
        super().__init__(name)
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.be_activation = be_activation
        self.split_on_be = split_on_be
        self.be_active = False
        self.split_active = False
        
    def act(self, observation):
        # Obs: [PnL%, Duration, Volatility, Z-Score]
        pnl_pct = observation[0] / 100.0 # Convert back to decimal
        
        # Stop Loss
        if pnl_pct < -self.sl_pct:
            return 2 # Close 100%
            
        # Take Profit
        if pnl_pct > self.tp_pct:
            return 2 # Close 100%
            
        # Break Even Logic (+ Split if enabled)
        if pnl_pct > self.be_activation and not self.be_active:
            self.be_active = True
            if self.split_on_be and not self.split_active:
                self.split_active = True
                return 5 # Signal to Move SL to BE AND Close 50%
            return 3 # Signal to Move SL to BE
            
        # Trailing Stop Logic (Simplified)
        # If we are in good profit (> 2%), start trailing
        if pnl_pct > self.tp_pct * 0.5:
             return 4 # Signal to Trail SL
            
        # Partial Close Logic (Standard)
        if pnl_pct > self.tp_pct * 0.5 and pnl_pct < self.tp_pct * 0.6:
            if not self.split_active: # Only split once
                self.split_active = True
                return 1 # Close 50%
            
        return 0 # Hold
