# STEP 1: สร้างไฟล์ใหม่ชื่อ "advanced_features.py"
# วางไฟล์นี้ในโฟลเดอร์เดียวกับ mt5_forex_connector.py

"""
Advanced Trading Features - Simplified Integration - FIXED VERSION
===============================================
เพิ่มความสามารถขั้นสูงในระบบที่มีอยู่แล้ว
FIXED: Duplicate returns, enum handling, JSON serialization
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

# CRITICAL FIX: Add clean_data_for_json function locally
def clean_data_for_json(data):
    """Clean data for JSON serialization - FIXED VERSION"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            cleaned[key] = clean_data_for_json(value)
        return cleaned
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(clean_data_for_json(item) for item in data)
    elif isinstance(data, (np.integer, np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64, np.float32)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, pd.Series):
        return data.tolist()
    elif isinstance(data, pd.DataFrame):
        return data.to_dict('records')
    elif isinstance(data, Enum):
        return data.value
    elif hasattr(data, '__dict__'):
        return str(data)
    elif pd.isna(data):
        return None
    elif data in [np.inf, -np.inf]:
        return None
    else:
        return data

# ============================================================================
# 1. MARKET REGIME DETECTOR (เวอร์ชันง่าย)
# ============================================================================

class MarketRegime(Enum):
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH" 
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

class SimpleMarketRegimeDetector:
    """Market Regime Detector เวอร์ชันง่าย"""
    
    def detect_regime(self, df_h4: pd.DataFrame, df_h1: pd.DataFrame) -> Dict:
        """ตรวจจับ Market Regime"""
        try:
            # คำนวณ Trend Strength
            close_h4 = df_h4['close']
            ema_20 = close_h4.ewm(span=20).mean()
            ema_50 = close_h4.ewm(span=50).mean()
            
            # Trend Direction
            current_price = close_h4.iloc[-1]
            trend_up = current_price > ema_20.iloc[-1] > ema_50.iloc[-1]
            trend_down = current_price < ema_20.iloc[-1] < ema_50.iloc[-1]
            
            # Volatility
            atr = self.calculate_atr(df_h4)
            atr_percentile = self.get_atr_percentile(df_h4, atr)
            
            # กำหนด Regime
            if trend_up and atr_percentile > 50:
                regime = MarketRegime.TRENDING_BULLISH
                confidence = 0.8
            elif trend_down and atr_percentile > 50:
                regime = MarketRegime.TRENDING_BEARISH
                confidence = 0.8
            elif atr_percentile > 70:
                regime = MarketRegime.HIGH_VOLATILITY
                confidence = 0.7
            elif atr_percentile < 30:
                regime = MarketRegime.LOW_VOLATILITY
                confidence = 0.7
            else:
                regime = MarketRegime.RANGING
                confidence = 0.6
            
            return {
                'regime': regime,
                'confidence': confidence,
                'trend_strength': self.calculate_trend_strength(df_h4),
                'volatility_percentile': atr_percentile,
                'current_atr': atr
            }
            
        except Exception as e:
            print(f"Regime detection error: {str(e)}")
            return {
                'regime': MarketRegime.RANGING,
                'confidence': 0.5,
                'trend_strength': 0.0,
                'volatility_percentile': 50.0,
                'current_atr': 0.001
            }
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """คำนวณ ATR"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.001
        except Exception as e:
            print(f"ATR calculation error: {str(e)}")
            return 0.001
    
    def get_atr_percentile(self, df: pd.DataFrame, current_atr: float) -> float:
        """คำนวณ ATR Percentile"""
        try:
            atr_series = []
            for i in range(14, len(df)):
                window_df = df.iloc[i-14:i]
                atr_val = self.calculate_atr(window_df)
                atr_series.append(atr_val)
            
            if len(atr_series) == 0:
                return 50.0
                
            atr_series = pd.Series(atr_series)
            percentile = (atr_series <= current_atr).mean() * 100
            return percentile
        except Exception as e:
            print(f"ATR percentile error: {str(e)}")
            return 50.0
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """คำนวณ Trend Strength"""
        try:
            close = df['close']
            ema_10 = close.ewm(span=10).mean()
            ema_20 = close.ewm(span=20).mean()
            ema_50 = close.ewm(span=50).mean()
            
            current_price = close.iloc[-1]
            
            # Bullish conditions
            bullish_score = 0
            if current_price > ema_10.iloc[-1]: bullish_score += 1
            if ema_10.iloc[-1] > ema_20.iloc[-1]: bullish_score += 1
            if ema_20.iloc[-1] > ema_50.iloc[-1]: bullish_score += 1
            
            # Bearish conditions  
            bearish_score = 0
            if current_price < ema_10.iloc[-1]: bearish_score += 1
            if ema_10.iloc[-1] < ema_20.iloc[-1]: bearish_score += 1
            if ema_20.iloc[-1] < ema_50.iloc[-1]: bearish_score += 1
            
            return max(bullish_score, bearish_score) / 3.0
        except Exception as e:
            print(f"Trend strength error: {str(e)}")
            return 0.0

# ============================================================================
# 2. ADVANCED SIGNAL SCORER
# ============================================================================

class AdvancedSignalScorer:
    """Advanced Signal Scoring System"""
    
    def __init__(self):
        # น้ำหนักของแต่ละปัจจัย
        self.weights = {
            'trend_alignment': 0.3,
            'momentum': 0.25, 
            'volatility': 0.2,
            'volume': 0.15,
            'regime_fit': 0.1
        }
        
        # คูณค่าตาม Market Regime
        self.regime_multipliers = {
            MarketRegime.TRENDING_BULLISH: 1.2,
            MarketRegime.TRENDING_BEARISH: 1.2,
            MarketRegime.HIGH_VOLATILITY: 0.8,
            MarketRegime.LOW_VOLATILITY: 0.6,
            MarketRegime.RANGING: 0.7
        }
    
    def calculate_enhanced_score(self, signal_data: Dict, regime_data: Dict, 
                               timeframe_analysis: Dict) -> Dict:
        """คำนวณ Enhanced Signal Score"""
        try:
            # 1. Trend Alignment Score
            trend_score = self.calculate_trend_score(timeframe_analysis)
            
            # 2. Momentum Score
            momentum_score = self.calculate_momentum_score(signal_data, timeframe_analysis)
            
            # 3. Volatility Score
            volatility_score = self.calculate_volatility_score(regime_data)
            
            # 4. Volume Score
            volume_score = self.calculate_volume_score(signal_data)
            
            # 5. Regime Fit Score
            regime_score = self.calculate_regime_fit_score(signal_data, regime_data)
            
            # คำนวณ Composite Score
            composite_score = (
                trend_score * self.weights['trend_alignment'] +
                momentum_score * self.weights['momentum'] +
                volatility_score * self.weights['volatility'] +
                volume_score * self.weights['volume'] +
                regime_score * self.weights['regime_fit']
            )
            
            # ปรับตาม Market Regime
            regime_multiplier = self.regime_multipliers.get(regime_data['regime'], 1.0)
            final_score = composite_score * regime_multiplier * regime_data['confidence']
            
            # แปลงเป็น Signal Strength (0-10)
            enhanced_strength = min(10, max(0, final_score * 10))
            
            # กำหนด Quality
            if enhanced_strength >= 8:
                quality = "EXCELLENT"
            elif enhanced_strength >= 6:
                quality = "GOOD"
            elif enhanced_strength >= 4:
                quality = "FAIR"
            else:
                quality = "POOR"
            
            return {
                'enhanced_strength': round(enhanced_strength, 2),
                'enhanced_quality': quality,
                'composite_score': round(composite_score, 3),
                'regime_adjusted_score': round(final_score, 3),
                'feature_scores': {
                    'trend_alignment': round(trend_score, 3),
                    'momentum': round(momentum_score, 3),
                    'volatility': round(volatility_score, 3),
                    'volume': round(volume_score, 3),
                    'regime_fit': round(regime_score, 3)
                },
                'regime_multiplier': regime_multiplier,
                'regime': regime_data['regime'].value if hasattr(regime_data['regime'], 'value') else str(regime_data['regime']),
                'confidence': regime_data['confidence']
            }
            
        except Exception as e:
            print(f"Enhanced scoring error: {str(e)}")
            return {
                'enhanced_strength': signal_data.get('strength', 0),
                'enhanced_quality': signal_data.get('entry_quality', 'POOR'),
                'error': str(e)
            }
    
    def calculate_trend_score(self, timeframe_analysis: Dict) -> float:
        """คำนวณ Trend Alignment Score"""
        if not timeframe_analysis:
            return 0.5
        
        trend_votes = 0
        total_timeframes = 0
        
        for tf_name, tf_data in timeframe_analysis.items():
            total_timeframes += 1
            trend_bias = tf_data.get('trend_bias', 'NEUTRAL')
            
            if trend_bias == 'BULLISH':
                trend_votes += 1
            elif trend_bias == 'BEARISH':
                trend_votes += 1  # Count as aligned (same direction)
        
        if total_timeframes == 0:
            return 0.5
            
        alignment_score = trend_votes / total_timeframes
        return alignment_score
    
    def calculate_momentum_score(self, signal_data: Dict, timeframe_analysis: Dict) -> float:
        """คำนวณ Momentum Score"""
        signal = signal_data.get('signal', 'NONE')
        
        # Base momentum from signal strength
        base_momentum = signal_data.get('strength', 0) / 10
        
        # Timeframe momentum alignment
        strong_signals = 0
        total_signals = 0
        
        for tf_name, tf_data in timeframe_analysis.items():
            tf_signal = tf_data.get('signal', 'NONE')
            total_signals += 1
            
            if tf_signal in ['STRONG_BUY', 'STRONG_SELL', 'BUY', 'SELL']:
                strong_signals += 1
        
        tf_momentum = strong_signals / total_signals if total_signals > 0 else 0
        
        # Combine scores
        combined_momentum = (base_momentum + tf_momentum) / 2
        return combined_momentum
    
    def calculate_volatility_score(self, regime_data: Dict) -> float:
        """คำนวณ Volatility Score"""
        volatility_percentile = regime_data.get('volatility_percentile', 50)
        
        # Optimal volatility range (30-70 percentile)
        if 30 <= volatility_percentile <= 70:
            return 1.0
        elif 20 <= volatility_percentile <= 80:
            return 0.8
        elif 10 <= volatility_percentile <= 90:
            return 0.6
        else:
            return 0.3
    
    def calculate_volume_score(self, signal_data: Dict) -> float:
        """คำนวณ Volume Score"""
        volume_ratio = signal_data.get('volumeRatio', 1.0)
        
        if volume_ratio >= 1.5:
            return 1.0
        elif volume_ratio >= 1.2:
            return 0.8
        elif volume_ratio >= 0.8:
            return 0.6
        else:
            return 0.3
    
    def calculate_regime_fit_score(self, signal_data: Dict, regime_data: Dict) -> float:
        """คำนวณ Regime Fit Score"""
        signal = signal_data.get('signal', 'NONE')
        regime = regime_data['regime']
        
        # ตรวจสอบว่า Signal เข้ากับ Regime หรือไม่
        if signal in ['BUY', 'STRONG_BUY'] and regime == MarketRegime.TRENDING_BULLISH:
            return 1.0
        elif signal in ['SELL', 'STRONG_SELL'] and regime == MarketRegime.TRENDING_BEARISH:
            return 1.0
        elif signal == 'NONE' and regime == MarketRegime.RANGING:
            return 0.8
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return 0.6  # High volatility = higher risk
        else:
            return 0.4

# ============================================================================
# 3. DYNAMIC POSITION SIZER
# ============================================================================

class DynamicPositionSizer:
    """Dynamic Position Sizing System"""
    
    def calculate_enhanced_position_size(self, account_balance: float,
                                    base_risk_percent: float,
                                    signal_data: Dict,
                                    enhanced_score: Dict,
                                    entry_price: float,
                                    stop_loss: float,
                                    symbol: str) -> Dict:
        """Calculate Position Size with ZERO DIVISION PROTECTION - COMPLETELY FIXED"""
        try:
            # CRITICAL FIX 1: Comprehensive input validation
            if account_balance <= 0:
                return self.get_error_result("Invalid account balance", account_balance, base_risk_percent)
            
            if base_risk_percent <= 0 or base_risk_percent > 100:
                return self.get_error_result("Invalid base risk percent", account_balance, base_risk_percent)
            
            if entry_price <= 0:
                return self.get_error_result("Invalid entry price", account_balance, base_risk_percent)
            
            if stop_loss <= 0:
                return self.get_error_result("Invalid stop loss", account_balance, base_risk_percent)
            
            # CRITICAL FIX 2: Check for meaningful risk distance
            points_at_risk = abs(entry_price - stop_loss)
            min_risk_threshold = entry_price * 0.0001  # 0.01% minimum risk distance
            
            if points_at_risk < min_risk_threshold:
                return self.get_error_result(f"Stop loss too close to entry: {points_at_risk:.6f}", account_balance, base_risk_percent)
            
            # Base risk amount calculation with protection
            try:
                base_risk_amount = account_balance * (base_risk_percent / 100)
                if base_risk_amount <= 0:
                    return self.get_error_result("Calculated base risk amount is zero", account_balance, base_risk_percent)
            except Exception as risk_calc_error:
                return self.get_error_result(f"Base risk calculation error: {str(risk_calc_error)}", account_balance, base_risk_percent)
            
            # CRITICAL FIX 3: Enhanced multipliers with protection
            try:
                enhanced_strength = enhanced_score.get('enhanced_strength', 0)
                if not isinstance(enhanced_strength, (int, float)) or not np.isfinite(enhanced_strength):
                    enhanced_strength = 0
                
                # Safe multiplier calculation
                signal_strength_multiplier = max(0.5, min(2.0, 0.5 + (max(0, enhanced_strength) / 20)))
                
                confidence = enhanced_score.get('confidence', 0.8)
                if not isinstance(confidence, (int, float)) or not np.isfinite(confidence):
                    confidence = 0.8
                confidence_multiplier = max(0.1, min(1.0, confidence))
                
            except Exception as mult_error:
                print(f"Multiplier calculation error: {str(mult_error)}")
                signal_strength_multiplier = 1.0
                confidence_multiplier = 0.8
            
            # CRITICAL FIX 4: Regime-based multiplier with protection
            try:
                regime_name = enhanced_score.get('regime', 'RANGING')
                if not isinstance(regime_name, str):
                    regime_name = 'RANGING'
                    
                regime_multipliers = {
                    'TRENDING_BULLISH': 1.2,
                    'TRENDING_BEARISH': 1.2,
                    'HIGH_VOLATILITY': 0.7,
                    'LOW_VOLATILITY': 0.9,
                    'RANGING': 0.8
                }
                regime_multiplier = regime_multipliers.get(regime_name, 1.0)
                
            except Exception as regime_error:
                print(f"Regime multiplier error: {str(regime_error)}")
                regime_multiplier = 1.0
            
            # CRITICAL FIX 5: Quality-based multiplier with protection
            try:
                enhanced_quality = enhanced_score.get('enhanced_quality', 'POOR')
                if not isinstance(enhanced_quality, str):
                    enhanced_quality = 'POOR'
                    
                quality_multipliers = {
                    'EXCELLENT': 1.3,
                    'GOOD': 1.1,
                    'FAIR': 0.9,
                    'POOR': 0.6
                }
                quality_multiplier = quality_multipliers.get(enhanced_quality, 1.0)
                
            except Exception as quality_error:
                print(f"Quality multiplier error: {str(quality_error)}")
                quality_multiplier = 1.0
            
            # CRITICAL FIX 6: Final risk amount calculation with protection
            try:
                adjusted_risk_amount = (base_risk_amount * 
                                    signal_strength_multiplier * 
                                    confidence_multiplier *
                                    regime_multiplier * 
                                    quality_multiplier)
                
                if not np.isfinite(adjusted_risk_amount) or adjusted_risk_amount <= 0:
                    adjusted_risk_amount = base_risk_amount
                    
                # Limit risk amount (max 3% of account)
                max_risk_amount = account_balance * 0.03
                adjusted_risk_amount = min(adjusted_risk_amount, max_risk_amount)
                
            except Exception as adj_risk_error:
                print(f"Adjusted risk calculation error: {str(adj_risk_error)}")
                adjusted_risk_amount = base_risk_amount
            
            # CRITICAL FIX 7: Symbol-specific calculations with zero protection
            try:
                if 'XAU' in symbol:
                    # Gold: 1 pip = $0.10, 1 lot = 100 oz
                    pip_size = 0.1
                    money_per_pip = 1.0
                    pips_at_risk = points_at_risk / pip_size
                    
                    if pips_at_risk <= 0 or money_per_pip <= 0:
                        raise ValueError("Invalid pips or money per pip for Gold")
                        
                    lot_size = adjusted_risk_amount / (pips_at_risk * money_per_pip)
                    
                elif 'JPY' in symbol:
                    # JPY pairs: 1 pip = 0.01
                    pip_size = 0.01
                    
                    if entry_price <= 0:
                        raise ValueError("Invalid entry price for JPY pair")
                        
                    money_per_pip = 10.0 / entry_price
                    pips_at_risk = points_at_risk / pip_size
                    
                    if pips_at_risk <= 0 or money_per_pip <= 0:
                        raise ValueError("Invalid calculation parameters for JPY pair")
                        
                    lot_size = adjusted_risk_amount / (pips_at_risk * money_per_pip)
                    
                else:
                    # Standard Forex: 1 pip = 0.0001
                    pip_size = 0.0001
                    money_per_pip = 10.0
                    pips_at_risk = points_at_risk / pip_size
                    
                    if pips_at_risk <= 0:
                        raise ValueError("Invalid pips at risk for standard forex")
                        
                    lot_size = adjusted_risk_amount / (pips_at_risk * money_per_pip)
                
                # CRITICAL FIX 8: Validate lot size calculation
                if not np.isfinite(lot_size) or lot_size <= 0:
                    raise ValueError(f"Invalid lot size calculated: {lot_size}")
                
                # Limit lot size
                lot_size = max(0.01, min(2.0, lot_size))
                lot_size = round(lot_size, 2)
                
                # CRITICAL FIX 9: Calculate actual risk with protection
                actual_risk = pips_at_risk * money_per_pip * lot_size
                if not np.isfinite(actual_risk) or actual_risk < 0:
                    actual_risk = 0
                    
                actual_risk_percent = (actual_risk / account_balance) * 100 if account_balance > 0 else 0
                
                return {
                    'lot_size': float(lot_size),
                    'base_risk_amount': round(float(base_risk_amount), 2),
                    'adjusted_risk_amount': round(float(adjusted_risk_amount), 2),
                    'actual_risk_amount': round(float(actual_risk), 2),
                    'actual_risk_percent': round(float(actual_risk_percent), 3),
                    'multipliers': {
                        'signal_strength': round(float(signal_strength_multiplier), 3),
                        'confidence': round(float(confidence_multiplier), 3),
                        'regime': round(float(regime_multiplier), 3),
                        'quality': round(float(quality_multiplier), 3)
                    },
                    'points_at_risk': round(float(points_at_risk), 5),
                    'pip_size': float(pip_size),
                    'money_per_pip': round(float(money_per_pip), 2),
                    'calculation_status': 'SUCCESS',
                    'zero_division_protected': True
                }
                
            except Exception as symbol_calc_error:
                print(f"Symbol-specific calculation error for {symbol}: {str(symbol_calc_error)}")
                return self.get_error_result(f"Symbol calculation error: {str(symbol_calc_error)}", account_balance, base_risk_percent)
                
        except Exception as e:
            print(f"Enhanced position sizing error: {str(e)}")
            return self.get_error_result(f"General error: {str(e)}", account_balance, base_risk_percent)

    def get_error_result(self, error_message: str, account_balance: float, base_risk_percent: float) -> Dict:
        """Return safe error result for position sizing"""
        try:
            safe_base_risk = max(0, account_balance * (base_risk_percent / 100)) if account_balance > 0 else 0
        except:
            safe_base_risk = 0
            
        return {
            'lot_size': 0.01,
            'base_risk_amount': safe_base_risk,
            'adjusted_risk_amount': safe_base_risk,
            'actual_risk_amount': 0,
            'actual_risk_percent': 0,
            'error': error_message,
            'calculation_status': 'ERROR',
            'zero_division_protected': True,
            'safe_fallback_used': True
        }

    def calculate_enhanced_score(self, signal_data: Dict, regime_data: Dict, 
                            timeframe_analysis: Dict) -> Dict:
        """Calculate Enhanced Signal Score with ZERO DIVISION PROTECTION - COMPLETELY FIXED"""
        try:
            # CRITICAL FIX 1: Input validation
            if not isinstance(signal_data, dict):
                signal_data = {}
            if not isinstance(regime_data, dict):
                regime_data = {'regime': 'RANGING', 'confidence': 0.5}
            if not isinstance(timeframe_analysis, dict):
                timeframe_analysis = {}
            
            # CRITICAL FIX 2: Calculate individual scores with protection
            trend_score = self.calculate_trend_score_safe(timeframe_analysis)
            momentum_score = self.calculate_momentum_score_safe(signal_data, timeframe_analysis)
            volatility_score = self.calculate_volatility_score_safe(regime_data)
            volume_score = self.calculate_volume_score_safe(signal_data)
            regime_score = self.calculate_regime_fit_score_safe(signal_data, regime_data)
            
            # CRITICAL FIX 3: Composite score calculation with protection
            try:
                weights = self.weights
                composite_score = (
                    trend_score * weights.get('trend_alignment', 0.3) +
                    momentum_score * weights.get('momentum', 0.25) +
                    volatility_score * weights.get('volatility', 0.2) +
                    volume_score * weights.get('volume', 0.15) +
                    regime_score * weights.get('regime_fit', 0.1)
                )
                
                if not np.isfinite(composite_score):
                    composite_score = 0.5
                    
            except Exception as comp_error:
                print(f"Composite score calculation error: {str(comp_error)}")
                composite_score = 0.5
            
            # CRITICAL FIX 4: Regime adjustment with protection
            try:
                regime = regime_data.get('regime', 'RANGING')
                regime_multiplier = self.regime_multipliers.get(regime, 1.0)
                confidence = regime_data.get('confidence', 0.8)
                
                if not isinstance(confidence, (int, float)) or not np.isfinite(confidence):
                    confidence = 0.8
                confidence = max(0.1, min(1.0, confidence))
                
                final_score = composite_score * regime_multiplier * confidence
                
                if not np.isfinite(final_score):
                    final_score = composite_score
                    
            except Exception as regime_adj_error:
                print(f"Regime adjustment error: {str(regime_adj_error)}")
                final_score = composite_score
            
            # CRITICAL FIX 5: Convert to signal strength with protection
            try:
                enhanced_strength = max(0, min(10, final_score * 10))
                
                if enhanced_strength >= 8:
                    quality = "EXCELLENT"
                elif enhanced_strength >= 6:
                    quality = "GOOD"
                elif enhanced_strength >= 4:
                    quality = "FAIR"
                else:
                    quality = "POOR"
                    
            except Exception as strength_error:
                print(f"Strength calculation error: {str(strength_error)}")
                enhanced_strength = 0
                quality = "POOR"
            
            return {
                'enhanced_strength': round(float(enhanced_strength), 2),
                'enhanced_quality': quality,
                'composite_score': round(float(composite_score), 3),
                'regime_adjusted_score': round(float(final_score), 3),
                'feature_scores': {
                    'trend_alignment': round(float(trend_score), 3),
                    'momentum': round(float(momentum_score), 3),
                    'volatility': round(float(volatility_score), 3),
                    'volume': round(float(volume_score), 3),
                    'regime_fit': round(float(regime_score), 3)
                },
                'regime_multiplier': float(regime_multiplier),
                'regime': str(regime),
                'confidence': float(confidence),
                'zero_division_protected': True,
                'calculation_status': 'SUCCESS'
            }
            
        except Exception as e:
            print(f"Enhanced scoring error: {str(e)}")
            return {
                'enhanced_strength': signal_data.get('strength', 0),
                'enhanced_quality': signal_data.get('entry_quality', 'POOR'),
                'error': str(e),
                'calculation_status': 'ERROR',
                'zero_division_protected': True
            }

    def calculate_trend_score_safe(self, timeframe_analysis: Dict) -> float:
        """Calculate trend score with zero division protection"""
        try:
            if not timeframe_analysis:
                return 0.5
            
            trend_votes = 0
            total_timeframes = 0
            
            for tf_name, tf_data in timeframe_analysis.items():
                if not isinstance(tf_data, dict):
                    continue
                    
                total_timeframes += 1
                trend_bias = tf_data.get('trend_bias', 'NEUTRAL')
                
                if trend_bias in ['BULLISH', 'BEARISH']:
                    trend_votes += 1
            
            if total_timeframes == 0:
                return 0.5
                
            alignment_score = trend_votes / total_timeframes
            return max(0.0, min(1.0, alignment_score))
            
        except Exception as e:
            print(f"Trend score calculation error: {str(e)}")
            return 0.5

    def calculate_momentum_score_safe(self, signal_data: Dict, timeframe_analysis: Dict) -> float:
        """Calculate momentum score with zero division protection"""
        try:
            # Base momentum from signal strength with protection
            strength = signal_data.get('strength', 0)
            if not isinstance(strength, (int, float)) or not np.isfinite(strength):
                strength = 0
            base_momentum = max(0, min(10, strength)) / 10
            
            # Timeframe momentum alignment with protection
            strong_signals = 0
            total_signals = 0
            
            for tf_name, tf_data in timeframe_analysis.items():
                if not isinstance(tf_data, dict):
                    continue
                    
                tf_signal = tf_data.get('signal', 'NONE')
                total_signals += 1
                
                if tf_signal in ['STRONG_BUY', 'STRONG_SELL', 'BUY', 'SELL']:
                    strong_signals += 1
            
            if total_signals == 0:
                tf_momentum = 0
            else:
                tf_momentum = strong_signals / total_signals
            
            # Combine scores with protection
            combined_momentum = (base_momentum + tf_momentum) / 2
            return max(0.0, min(1.0, combined_momentum))
            
        except Exception as e:
            print(f"Momentum score calculation error: {str(e)}")
            return 0.5

    def calculate_volatility_score_safe(self, regime_data: Dict) -> float:
        """Calculate volatility score with protection"""
        try:
            volatility_percentile = regime_data.get('volatility_percentile', 50)
            if not isinstance(volatility_percentile, (int, float)) or not np.isfinite(volatility_percentile):
                volatility_percentile = 50
            
            volatility_percentile = max(0, min(100, volatility_percentile))
            
            # Optimal volatility range (30-70 percentile)
            if 30 <= volatility_percentile <= 70:
                return 1.0
            elif 20 <= volatility_percentile <= 80:
                return 0.8
            elif 10 <= volatility_percentile <= 90:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            print(f"Volatility score calculation error: {str(e)}")
            return 0.5

    def calculate_volume_score_safe(self, signal_data: Dict) -> float:
        """Calculate volume score with protection"""
        try:
            volume_ratio = signal_data.get('volumeRatio', signal_data.get('volume_ratio', 1.0))
            if not isinstance(volume_ratio, (int, float)) or not np.isfinite(volume_ratio):
                volume_ratio = 1.0
            
            volume_ratio = max(0, volume_ratio)
            
            if volume_ratio >= 1.5:
                return 1.0
            elif volume_ratio >= 1.2:
                return 0.8
            elif volume_ratio >= 0.8:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            print(f"Volume score calculation error: {str(e)}")
            return 0.5

    def calculate_regime_fit_score_safe(self, signal_data: Dict, regime_data: Dict) -> float:
        """Calculate regime fit score with protection"""
        try:
            signal = signal_data.get('signal', 'NONE')
            regime = regime_data.get('regime', 'RANGING')
            
            if not isinstance(signal, str):
                signal = 'NONE'
            if not isinstance(regime, str):
                regime = 'RANGING'
            
            # Check if signal fits regime
            if signal in ['BUY', 'STRONG_BUY'] and 'BULLISH' in regime:
                return 1.0
            elif signal in ['SELL', 'STRONG_SELL'] and 'BEARISH' in regime:
                return 1.0
            elif signal == 'NONE' and 'RANGING' in regime:
                return 0.8
            elif 'HIGH_VOLATILITY' in regime:
                return 0.6  # High volatility = higher risk
            else:
                return 0.4
                
        except Exception as e:
            print(f"Regime fit score calculation error: {str(e)}")
            return 0.5
    
# ============================================================================
# 4. INTEGRATION HELPER CLASS
# ============================================================================

class AdvancedTradingIntegrator:
    """Helper class สำหรับ integrate advanced features"""
    
    def __init__(self):
        self.regime_detector = SimpleMarketRegimeDetector()
        self.signal_scorer = AdvancedSignalScorer()
        self.position_sizer = DynamicPositionSizer()
        
        print("Advanced Trading Features Initialized!")
        print("- Market Regime Detection: ON")
        print("- Enhanced Signal Scoring: ON") 
        print("- Dynamic Position Sizing: ON")
    
    def enhance_signal_analysis(self, symbol: str, basic_signal_data: Dict, 
                              timeframe_data: Dict) -> Dict:
        """เพิ่มความสามารถให้กับ Signal Analysis ที่มีอยู่"""
        try:
            # 1. Detect Market Regime
            if 'H4' in timeframe_data and 'H1' in timeframe_data:
                regime_data = self.regime_detector.detect_regime(
                    timeframe_data['H4'], timeframe_data['H1']
                )
            else:
                regime_data = {
                    'regime': MarketRegime.RANGING,
                    'confidence': 0.5,
                    'trend_strength': 0.0,
                    'volatility_percentile': 50.0
                }
            
            # 2. Enhanced Signal Scoring
            timeframe_analysis = basic_signal_data.get('enhanced_analysis', {}).get('timeframe_analysis', {})
            enhanced_score = self.signal_scorer.calculate_enhanced_score(
                basic_signal_data, regime_data, timeframe_analysis
            )
            
            # 3. Dynamic Position Sizing
            if basic_signal_data.get('stop_loss', 0) > 0:
                position_info = self.position_sizer.calculate_enhanced_position_size(
                    account_balance=basic_signal_data.get('account_balance', 10000),
                    base_risk_percent=1.5,
                    signal_data=basic_signal_data,
                    enhanced_score=enhanced_score,
                    entry_price=basic_signal_data.get('optimal_entry', 0),
                    stop_loss=basic_signal_data.get('stop_loss', 0),
                    symbol=symbol
                )
            else:
                position_info = {'lot_size': 0.01, 'error': 'No stop loss defined'}
            
            # 4. รวมข้อมูลทั้งหมด
            enhanced_result = basic_signal_data.copy()
            enhanced_result.update({
                'enhanced_strength': enhanced_score['enhanced_strength'],
                'enhanced_quality': enhanced_score['enhanced_quality'],
                'market_regime': regime_data['regime'].value if hasattr(regime_data['regime'], 'value') else str(regime_data['regime']),
                'regime_confidence': regime_data['confidence'],
                'trend_strength': regime_data['trend_strength'],
                'volatility_percentile': regime_data['volatility_percentile'],
                'enhanced_lot_size': position_info['lot_size'],
                'enhanced_risk_amount': position_info.get('actual_risk_amount', 0),
                'enhanced_risk_percent': position_info.get('actual_risk_percent', 0),
                'enhancement_details': {
                    'regime_data': regime_data,
                    'enhanced_score': enhanced_score,
                    'position_info': position_info
                }
            })
            
            # CRITICAL FIX: Ensure all data is JSON serializable
            enhanced_result = clean_data_for_json(enhanced_result)
            
            # Handle specific enum conversions
            if 'market_regime' in enhanced_result:
                if hasattr(enhanced_result['market_regime'], 'value'):
                    enhanced_result['market_regime'] = enhanced_result['market_regime'].value
            
            # Handle enhancement_details
            if 'enhancement_details' in enhanced_result:
                enhanced_result['enhancement_details'] = clean_data_for_json(
                    enhanced_result['enhancement_details']
                )
            
            return enhanced_result
            
        except Exception as e:
            print(f"Enhancement error for {symbol}: {str(e)}")
            # Return original data if enhancement fails
            enhanced_result = basic_signal_data.copy()
            enhanced_result.update({
                'enhanced_strength': basic_signal_data.get('strength', 0),
                'enhanced_quality': basic_signal_data.get('entry_quality', 'POOR'),
                'market_regime': 'UNKNOWN',
                'enhancement_error': str(e)
            })
            return enhanced_result

# Export ตัวหลักสำหรับใช้งาน
__all__ = [
    'AdvancedTradingIntegrator',
    'SimpleMarketRegimeDetector', 
    'AdvancedSignalScorer',
    'DynamicPositionSizer',
    'MarketRegime'
]

print("Advanced Features Module Ready for Integration!")