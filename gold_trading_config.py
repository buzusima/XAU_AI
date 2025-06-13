import MetaTrader5 as mt5
import numpy as np
from typing import Dict, Tuple, Optional

class GoldTradingConfig:
    """
    Gold-Specific Trading Configuration
    อาจารย์ฟินิกซ์ - Optimized for XAUUSD Trading
    """
    
    def __init__(self):
        # Gold Symbol Variations (Different Brokers)
        self.gold_symbols = [
            "XAUUSD", "XAUUSD.c", "GOLD", "GOLD.c", 
            "XAU/USD", "GC", "XAUUSD.a", "GOLDSPOT"
        ]
        
        # Gold-Specific Settings
        self.gold_point_value = 0.1  # $0.1 per point for 0.01 lot
        self.gold_pip_size = 0.01    # Gold pip size
        self.gold_min_spread = 0.15  # Minimum acceptable spread
        self.gold_max_spread = 1.0   # Maximum acceptable spread
        
        # Adjusted Risk Management for Gold
        self.base_lot_size = 0.01
        self.max_lot_size = 0.10
        self.gold_sl_pips = 15       # Tighter SL for Gold
        self.gold_tp_ratio = 2.5     # Higher TP ratio
        
        # Volatility-Based Adjustments
        self.volatility_multiplier = 1.5  # Gold is 1.5x more volatile
        self.news_filter_hours = [8, 9, 10, 13, 14, 15]  # GMT news hours
        
    def detect_gold_symbol(self, mt5_connection=None) -> Optional[str]:
        """
        Auto-detect Gold symbol for any broker
        """
        if not mt5_connection:
            if not mt5.initialize():
                return None
                
        symbols = mt5.symbols_get()
        if not symbols:
            return None
            
        # Check for Gold symbols
        for symbol_info in symbols:
            symbol_name = symbol_info.name.upper()
            
            if any(gold_sym in symbol_name for gold_sym in self.gold_symbols):
                # Verify it's actually Gold by checking specifications
                if self._verify_gold_symbol(symbol_info):
                    print(f"✅ Gold symbol detected: {symbol_info.name}")
                    return symbol_info.name
                    
        print("❌ No Gold symbol found")
        return None
        
    def _verify_gold_symbol(self, symbol_info) -> bool:
        """
        Verify if symbol is actually Gold by checking specifications
        """
        try:
            # Gold characteristics:
            # - Usually 2-3 decimal places
            # - Contract size around 100 oz
            # - Point value around 0.01-0.1
            
            digits = symbol_info.digits
            contract_size = symbol_info.trade_contract_size
            
            # Gold typically has 2-3 decimal places
            if digits < 2 or digits > 3:
                return False
                
            # Contract size check (flexible for different brokers)
            if contract_size < 50 or contract_size > 200:
                return False
                
            return True
            
        except Exception:
            return False
            
    def calculate_gold_position_size(self, account_balance: float, 
                                   risk_percent: float = 0.02,
                                   sl_pips: int = None) -> float:
        """
        Calculate optimal position size for Gold trading
        """
        if sl_pips is None:
            sl_pips = self.gold_sl_pips
            
        # Risk amount in USD
        risk_amount = account_balance * risk_percent
        
        # Gold point value calculation
        # For XAUUSD: 1 pip = $0.01 for 0.01 lot (micro lot)
        #            1 pip = $0.1 for 0.01 lot (mini lot) - depends on broker
        pip_value_per_microlot = self.gold_point_value
        
        # Calculate lot size
        lot_size = risk_amount / (sl_pips * pip_value_per_microlot * 100)
        
        # Apply Gold-specific limits
        lot_size = max(0.01, min(self.max_lot_size, lot_size))
        
        # Round to broker's lot step (usually 0.01)
        lot_size = round(lot_size, 2)
        
        return lot_size
        
    def get_gold_sl_tp_levels(self, symbol: str, order_type: int, 
                            entry_price: float) -> Tuple[float, float]:
        """
        Calculate SL/TP levels optimized for Gold volatility
        """
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.0, 0.0
                
            # Get current market data for volatility calculation
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
            if rates is not None and len(rates) > 0:
                # Calculate 24-hour ATR for dynamic SL/TP
                high_prices = [rate[2] for rate in rates]  # High prices
                low_prices = [rate[3] for rate in rates]   # Low prices
                close_prices = [rate[4] for rate in rates] # Close prices
                
                # Simple ATR calculation
                ranges = []
                for i in range(1, len(rates)):
                    true_range = max(
                        high_prices[i] - low_prices[i],
                        abs(high_prices[i] - close_prices[i-1]),
                        abs(low_prices[i] - close_prices[i-1])
                    )
                    ranges.append(true_range)
                
                atr = np.mean(ranges) if ranges else self.gold_pip_size * 20
                
                # Dynamic SL based on ATR
                sl_distance = max(atr * 0.8, self.gold_pip_size * self.gold_sl_pips)
                tp_distance = sl_distance * self.gold_tp_ratio
            else:
                # Fallback to static values
                sl_distance = self.gold_pip_size * self.gold_sl_pips
                tp_distance = sl_distance * self.gold_tp_ratio
            
            if order_type == mt5.ORDER_TYPE_BUY:
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:  # SELL
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance
                
            return stop_loss, take_profit
            
        except Exception as e:
            print(f"❌ Gold SL/TP calculation error: {str(e)}")
            return 0.0, 0.0
            
    def check_gold_trading_conditions(self, symbol: str) -> Dict:
        """
        Check if Gold trading conditions are favorable
        """
        try:
            # Get current market info
            tick = mt5.symbol_info_tick(symbol)
            symbol_info = mt5.symbol_info(symbol)
            
            if not tick or not symbol_info:
                return {"status": "ERROR", "message": "Cannot get market data"}
            
            # Calculate spread
            spread = (tick.ask - tick.bid) / self.gold_pip_size
            
            # Check trading session
            from datetime import datetime
            current_hour = datetime.now().hour
            is_news_time = current_hour in self.news_filter_hours
            
            # Market volatility check
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 20)
            volatility = 0
            if rates is not None and len(rates) > 1:
                prices = [rate[4] for rate in rates]  # Close prices
                volatility = np.std(prices) / np.mean(prices) * 100
            
            conditions = {
                "status": "OK",
                "spread": spread,
                "spread_acceptable": spread <= self.gold_max_spread,
                "volatility": volatility,
                "high_volatility": volatility > 0.5,
                "news_time": is_news_time,
                "tradeable": (spread <= self.gold_max_spread and not is_news_time),
                "current_price": tick.bid,
                "recommendation": ""
            }
            
            # Trading recommendation
            if not conditions["spread_acceptable"]:
                conditions["recommendation"] = "WAIT - Spread too wide"
            elif conditions["news_time"]:
                conditions["recommendation"] = "CAUTION - News time"
            elif conditions["high_volatility"]:
                conditions["recommendation"] = "REDUCE_SIZE - High volatility"
            else:
                conditions["recommendation"] = "TRADE - Good conditions"
                
            return conditions
            
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    
    def calculate_risk_metrics(self, symbol: str, lot_size: float, 
                             sl_pips: int, entry_price: float = None) -> Dict:
        """
        Calculate comprehensive risk metrics for Gold trade
        """
        try:
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                return {"error": "Cannot get account info"}
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {"error": f"Cannot get {symbol} info"}
            
            # Get current price if not provided
            if entry_price is None:
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    return {"error": "Cannot get current price"}
                entry_price = tick.bid
            
            # Account details
            balance = account_info.balance
            equity = account_info.equity
            margin_free = account_info.margin_free
            
            # Risk calculations
            pip_value = self._calculate_pip_value(symbol_info, lot_size)
            max_loss_usd = sl_pips * pip_value
            risk_percentage = (max_loss_usd / balance) * 100
            
            # Margin requirements
            margin_required = self._calculate_margin_required(symbol_info, lot_size, entry_price)
            margin_percentage = (margin_required / margin_free) * 100 if margin_free > 0 else 100
            
            # Position value
            position_value = lot_size * symbol_info.trade_contract_size * entry_price
            
            # Risk-Reward metrics
            tp_pips = sl_pips * self.gold_tp_ratio
            potential_profit = tp_pips * pip_value
            risk_reward_ratio = potential_profit / max_loss_usd if max_loss_usd > 0 else 0
            
            # Kelly Criterion (simplified - assumes 60% win rate)
            win_rate = 0.60
            avg_win = potential_profit
            avg_loss = max_loss_usd
            kelly_percentage = ((win_rate * avg_win) - ((1 - win_rate) * avg_loss)) / avg_win if avg_win > 0 else 0
            kelly_percentage = max(0, min(kelly_percentage, 0.25)) * 100  # Cap at 25%
            
            # Drawdown estimation
            estimated_drawdown = max_loss_usd * 3  # Assume 3 consecutive losses
            drawdown_percentage = (estimated_drawdown / balance) * 100
            
            # Risk assessment
            risk_level = self._assess_risk_level(risk_percentage, margin_percentage, drawdown_percentage)
            
            return {
                "account_balance": balance,
                "account_equity": equity,
                "margin_free": margin_free,
                "lot_size": lot_size,
                "entry_price": entry_price,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "pip_value": pip_value,
                "max_loss_usd": max_loss_usd,
                "potential_profit": potential_profit,
                "risk_percentage": risk_percentage,
                "margin_required": margin_required,
                "margin_percentage": margin_percentage,
                "position_value": position_value,
                "risk_reward_ratio": risk_reward_ratio,
                "kelly_percentage": kelly_percentage,
                "estimated_drawdown": estimated_drawdown,
                "drawdown_percentage": drawdown_percentage,
                "risk_level": risk_level,
                "recommendations": self._get_risk_recommendations(risk_level, risk_percentage, margin_percentage)
            }
            
        except Exception as e:
            return {"error": f"Risk calculation error: {str(e)}"}
    
    def _calculate_pip_value(self, symbol_info, lot_size: float) -> float:
        """Calculate pip value for Gold"""
        try:
            # For Gold: 1 pip = 0.01, contract size usually 100 oz
            contract_size = symbol_info.trade_contract_size
            
            # Gold pip value = (pip size * contract size * lot size)
            pip_value = self.gold_pip_size * contract_size * lot_size
            
            return pip_value
            
        except Exception:
            # Fallback calculation
            return 0.01 * 100 * lot_size  # Standard Gold pip value
    
    def _calculate_margin_required(self, symbol_info, lot_size: float, price: float) -> float:
        """Calculate margin required for Gold position"""
        try:
            # Margin = (Contract Size * Lot Size * Price) / Leverage
            contract_size = symbol_info.trade_contract_size
            
            # Get account leverage
            account_info = mt5.account_info()
            leverage = account_info.leverage if account_info else 100
            
            margin = (contract_size * lot_size * price) / leverage
            
            return margin
            
        except Exception:
            # Fallback: assume 1:100 leverage
            return (100 * lot_size * price) / 100
    
    def _assess_risk_level(self, risk_pct: float, margin_pct: float, drawdown_pct: float) -> str:
        """Assess overall risk level"""
        
        high_risk_conditions = [
            risk_pct > 3.0,      # Risk > 3% per trade
            margin_pct > 50.0,   # Margin > 50%
            drawdown_pct > 15.0  # Potential drawdown > 15%
        ]
        
        medium_risk_conditions = [
            risk_pct > 1.5,      # Risk > 1.5% per trade
            margin_pct > 25.0,   # Margin > 25%
            drawdown_pct > 8.0   # Potential drawdown > 8%
        ]
        
        if any(high_risk_conditions):
            return "HIGH"
        elif any(medium_risk_conditions):
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_risk_recommendations(self, risk_level: str, risk_pct: float, margin_pct: float) -> list:
        """Get risk management recommendations"""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.append("⚠️ HIGH RISK: Consider reducing position size")
            if risk_pct > 3.0:
                recommended_size = 0.01 * (2.0 / risk_pct)
                recommendations.append(f"📉 Reduce lot size to ~{recommended_size:.2f} for 2% risk")
            if margin_pct > 50.0:
                recommendations.append("💰 High margin usage - monitor free margin closely")
                
        elif risk_level == "MEDIUM":
            recommendations.append("⚡ MEDIUM RISK: Acceptable but monitor closely")
            if risk_pct > 2.0:
                recommendations.append("📊 Consider reducing size for better risk management")
                
        else:
            recommendations.append("✅ LOW RISK: Good risk management")
            recommendations.append("👍 Position size is within safe parameters")
        
        # General recommendations
        recommendations.append(f"📈 Risk per trade: {risk_pct:.2f}% (Target: <2%)")
        recommendations.append(f"💸 Margin usage: {margin_pct:.1f}% (Target: <30%)")
        
        return recommendations
    
    def optimize_position_size(self, symbol: str, account_balance: float, 
                             target_risk_pct: float = 2.0, sl_pips: int = None) -> Dict:
        """
        Optimize position size based on target risk percentage
        """
        if sl_pips is None:
            sl_pips = self.gold_sl_pips
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {"error": f"Cannot get {symbol} info"}
            
            # Calculate optimal lot size for target risk
            target_risk_usd = account_balance * (target_risk_pct / 100)
            
            # Estimate pip value for 0.01 lot
            base_pip_value = self._calculate_pip_value(symbol_info, 0.01)
            
            # Calculate required lot size
            optimal_lot = target_risk_usd / (sl_pips * base_pip_value * 100)
            
            # Apply constraints
            min_lot = symbol_info.volume_min
            max_lot = min(symbol_info.volume_max, self.max_lot_size)
            lot_step = symbol_info.volume_step
            
            # Round to lot step
            optimal_lot = max(min_lot, optimal_lot)
            optimal_lot = min(max_lot, optimal_lot)
            if lot_step > 0:
                optimal_lot = round(optimal_lot / lot_step) * lot_step
            
            # Calculate actual risk with optimized size
            actual_risk_metrics = self.calculate_risk_metrics(symbol, optimal_lot, sl_pips)
            
            return {
                "target_risk_percentage": target_risk_pct,
                "target_risk_usd": target_risk_usd,
                "optimal_lot_size": optimal_lot,
                "min_lot_allowed": min_lot,
                "max_lot_allowed": max_lot,
                "lot_step": lot_step,
                "risk_metrics": actual_risk_metrics
            }
            
        except Exception as e:
            return {"error": f"Position optimization error: {str(e)}"}