import logging
import os
from datetime import datetime

class V19Logger:
    """
    Centralized logger for V19 - Writes to file with agent thinking process
    """
    def __init__(self, log_dir="gemini_v19/logs"):
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        log_file = os.path.join(log_dir, f"v19_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Configure logger
        self.logger = logging.getLogger('V19')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
        
        print(f"✅ V19 Logger initialized: {log_file}")
    
    def info(self, msg):
        self.logger.info(msg)
    
    def decision(self, symbol, action, policy, value, indicators):
        """Log agent's thinking process"""
        action_names = ["HOLD", "BUY", "SELL", "SPLIT", "CLOSE"]
        
        msg = f"""
═══ {symbol} DECISION ═══
Action: {action_names[action]} (Confidence: {policy[action]*100:.1f}%)
Value: {value:+.2f}
Policy: {' | '.join([f'{action_names[i]}:{policy[i]*100:.0f}%' for i in range(5)])}
Context: RSI={indicators.get('rsi', 0):.0f} Stoch={indicators.get('stoch_k', 0)*100:.0f} BB={indicators.get('bb_b', 0):.2f}
"""
        self.info(msg)
    
    def execution(self, symbol, action, volume, price, success):
        """Log trade execution"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.info(f"{status} | {symbol}: {action} {volume:.2f} @ {price:.5f}")
    
    def iteration(self, iteration, duration):
        """Log iteration summary"""
        self.info(f"━━━ Iteration #{iteration} completed in {duration:.2f}s ━━━\n")
