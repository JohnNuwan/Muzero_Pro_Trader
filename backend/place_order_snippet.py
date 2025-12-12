    def get_latest_atr(self, symbol):
        """Get latest ATR value for a symbol"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
            if rates is None:
                return 0.001  # Fallback
            df = pd.DataFrame(rates)
            atr = self.calculate_atr(df['high'], df['low'], df['close'], period=14)
            return atr.iloc[-1] if len(atr) > 0 else 0.001
        except:
            return 0.001

    def place_order(self, symbol, signal, lot, comment, **kwargs):
        """
        Place an order with enhanced Telegram notifications
        
        Args:
            symbol: Trading symbol
            signal: 1 for BUY, 0 for SELL
            lot: Position size
            comment: Order comment (Entry/Pyramid)
            **kwargs: regime, confidence, z_score, adx, strategy
        """
        # Get prices
        tick = mt5.symbol_info_tick(symbol)
        if signal == 1:  # BUY
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
            action = "BUY 🟢"
        else:  # SELL
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
            action = "SELL 🔴"
        
        # Calculate SL and TP
        params = self.params.get(symbol, {})
        atr = self.get_latest_atr(symbol)
        
        sl_mult = params.get("sl_mult", 2.0)
        tp_mult = params.get("tp_mult", 3.0)
        
        point = mt5.symbol_info(symbol).point
        
        if signal == 1:  # BUY
            sl = price - (atr * sl_mult)
            tp = price + (atr * tp_mult)
        else:  # SELL
            sl = price + (atr * sl_mult)
            tp = price - (atr * tp_mult)
        
        # Place order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": round(sl, mt5.symbol_info(symbol).digits),
            "tp": round(tp, mt5.symbol_info(symbol).digits),
            "magic": 888888,
            "comment": f"Gemini V12 {comment}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Save to database
            regime = kwargs.get('regime', '')
            confidence = kwargs.get('confidence', 0)
            z_score = kwargs.get('z_score', 0)
            adx = kwargs.get('adx', 0)
            strategy_name = kwargs.get('strategy', 'Unknown')
            
            trade = Trade(
                ticket=result.order,
                symbol=symbol,
                type="BUY" if signal == 1 else "SELL",
                volume=lot,
                price_open=price,
                sl=sl,
                tp=tp,
                open_time=datetime.now(),
                strategy=strategy_name,
                status="OPEN"
            )
            self.db.add(trade)
            self.db.commit()
            
            # News status
            is_news, _ = self.check_news(symbol)
            news_status = "⚠️ HIGH IMPACT" if is_news else "✅ SAFE"
            
            # HTF Trend
            htf_trend = self.check_htf_trend(symbol)
            
            # Build beautiful Telegram message
            if comment == "Entry":
                emoji = "🚀"
                title = "INITIAL ENTRY"
            else:  # Pyramid
                emoji = "📈"
                title = "PYRAMID"
            
            message = f"""
{emoji} *{title}* {emoji}
━━━━━━━━━━━━━━━━━
📊 Symbol: `{symbol}`
📍 Action: {action}
💰 Strategy: `{strategy_name}`

📈 *Trade Details:*
├ Entry: `{price:.5f}`
├ SL: `{sl:.5f}` ({abs(price-sl)/point:.0f} pts)
├ TP: `{tp:.5f}` ({abs(tp-price)/point:.0f} pts)
└ Lot: `{lot}`

🎯 *AI Analysis:*
├ Z-Score: `{z_score:.2f}`
├ Trend: `{htf_trend}`
├ Regime: `{regime}`
├ ADX: `{adx:.0f}`
└ Conf: `{confidence*100:.1f}%`

🌍 News: {news_status}
━━━━━━━━━━━━━━━━━
🎲 Ticket: `#{result.order}`
"""
            
            self.send_telegram(message.strip())
            self.log(f"✅ Order Placed: {symbol} {action} @ {price}", "SUCCESS")
            return True
        else:
            error_msg = f"❌ Order Failed: {symbol} - {result.comment if result else 'No response'}"
            self.log(error_msg, "ERROR")
            self.send_telegram(error_msg)
            return False

