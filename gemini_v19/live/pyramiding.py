import MetaTrader5 as mt5
from gemini_v19.utils.pyramid_config import PYRAMID_CONFIG

class PyramidManager:
    """
    Manages pyramiding of winning positions.
    """
    def __init__(self, symbol, magic):
        self.symbol = symbol
        self.magic = magic
        self.config = PYRAMID_CONFIG
        
    def can_pyramid(self, main_position, mcts_signal, confidence):
        """
        Check if we can add a pyramid position.
        
        Args:
            main_position: The initial position object
            mcts_signal: 'BUY' or 'SELL' from MCTS
            confidence: MCTS confidence score (0.0 - 1.0)
            
        Returns:
            bool: True if pyramiding is allowed
        """
        if not self.config['enabled']:
            return False

        # 1. Main position must be in profit
        if main_position.profit <= 0:
            return False
        
        # 2. Signal must match direction
        if main_position.type == mt5.POSITION_TYPE_BUY:
            if mcts_signal != 'BUY':
                return False
        else:  # SHORT
            if mcts_signal != 'SELL':
                return False
        
        # 3. Check max pyramids limit
        current_pyramids = self.count_pyramids(main_position)
        if current_pyramids >= self.config['max_pyramids']:
            return False
        
        # 4. Check confidence threshold
        if confidence < self.config['min_confidence']:
            return False
        
        return True
    
    def add_pyramid(self, main_position, mcts_signal):
        """
        Add a pyramid position.
        """
        # Calculate volume (50% of main position)
        pyramid_volume = main_position.volume * self.config['pyramid_volume_ratio']
        
        # Round to valid step
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            print(f"❌ [Pyramid] Symbol info not found for {self.symbol}")
            return None
            
        step = symbol_info.volume_step
        pyramid_volume = round(pyramid_volume / step) * step
        pyramid_volume = max(symbol_info.volume_min, min(pyramid_volume, symbol_info.volume_max))
        
        # Determine order type and price
        if main_position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.symbol).ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.symbol).bid
        
        # Create unique comment to link to main position
        pyramid_count = self.count_pyramids(main_position)
        comment = f"PYRAMID_{pyramid_count+1}_{main_position.ticket}"
        
        # Send order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pyramid_volume,
            "type": order_type,
            "price": price,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ [Pyramid] Added layer {pyramid_count+1}: {pyramid_volume} lots @ {price}")
            return result.order
        else:
            print(f"❌ [Pyramid] Failed to add layer: {result.comment}")
            return None
    
    def monitor_pyramids(self):
        """
        Monitor existing pyramids and move SL to BE if profit target reached.
        """
        if not self.config['enabled']:
            return

        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return
            
        for pos in positions:
            # Check if it's a pyramid position
            if not pos.comment.startswith("PYRAMID_"):
                continue
            
            # Calculate current profit %
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                continue
                
            if pos.type == mt5.POSITION_TYPE_BUY:
                current_price = tick.bid
                profit_pct = (current_price - pos.price_open) / pos.price_open
            else:
                current_price = tick.ask
                # For short, price going down is profit
                profit_pct = (pos.price_open - current_price) / pos.price_open
            
            # Check trigger
            if profit_pct >= self.config['sl_trigger_profit']:
                self._secure_position(pos)
                
    def _secure_position(self, position):
        """
        Move SL to Break Even + Spread.
        """
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            return
            
        spread_points = symbol_info.spread
        spread_price = spread_points * symbol_info.point
        
        if position.type == mt5.POSITION_TYPE_BUY:
            be_sl = position.price_open + spread_price
            # Only move if current SL is lower (or 0)
            if position.sl == 0 or position.sl < be_sl - 0.00001:
                self._modify_sl(position, be_sl)
        else:
            be_sl = position.price_open - spread_price
            # Only move if current SL is higher (or 0)
            if position.sl == 0 or position.sl > be_sl + 0.00001:
                self._modify_sl(position, be_sl)

    def _modify_sl(self, position, sl_price):
        """
        Send SL modification request.
        """
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": sl_price,
            "tp": position.tp,
            "symbol": self.symbol # Sometimes required
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"🔒 [Pyramid] Secured position {position.ticket} (SL -> BE+Spread)")
        else:
            # print(f"❌ [Pyramid] Failed to secure position: {result.comment}")
            pass # Suppress spam on failure
    
    def count_pyramids(self, main_position):
        """
        Count active pyramids linked to a main position.
        """
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return 0
            
        count = 0
        for pos in positions:
            # Check if comment contains main position ticket
            if pos.comment.startswith("PYRAMID_") and str(main_position.ticket) in pos.comment:
                count += 1
        return count
    
    def get_main_position(self):
        """
        Get the main position (oldest one without PYRAMID_ tag).
        """
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return None
            
        # Sort by time to find oldest? Usually not needed if we filter by comment
        # But let's be safe and pick the one that is NOT a pyramid
        for pos in positions:
            if not pos.comment.startswith("PYRAMID_"):
                return pos
        
        return None
