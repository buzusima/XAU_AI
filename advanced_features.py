# แก้ไขไฟล์ "advanced_features.py" - FIXED VERSION

"""
Advanced Trading Features - Universal Broker Support + EXTENDED VERSION - FIXED
===============================================================================
เพิ่มความสามารถขั้นสูงในระบบที่มีอยู่แล้ว + เพิ่มฟีเจอร์ใหม่
UNIVERSAL: ใช้ได้กับทุกโบรกเกอร์ผ่านระบบ BrokerSymbolAdapter
EXTENDED: เพิ่ม Advanced Pattern Recognition + Smart Risk Scaling
FIXED: Missing methods, duplicate returns, enum handling, JSON serialization
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import logging
from datetime import datetime, timedelta

def clean_data_for_json(data):
    """Clean data for JSON serialization - Universal Broker Version"""
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

class MarketRegime(Enum):
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH" 
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

class UniversalMarketRegimeDetector:
    """Universal Market Regime Detector - รองรับทุกโบรกเกอร์"""
    
    def __init__(self, symbol_adapter=None):
        """Initialize with universal symbol adapter"""
        self.symbol_adapter = symbol_adapter
        self.logger = logging.getLogger(__name__)
        self.current_regime = 'NORMAL'
        self.volatility_state = 'MEDIUM'
        self.trend_strength = 0.5
        self.enhancement_active = True
        self.last_regime_update = datetime.now()
        
        # Initialize components
        self.regime_detector = None  # จะ initialize ทีหลัง
        self.signal_scorer = None    
        self.position_sizer = None        
        # Performance tracking
        self.enhancement_stats = {
            'signals_enhanced': 0,
            'performance_improvement': 0.0,
            'last_update': datetime.now()
        }

    def _calculate_real_market_regime(self, symbol: str, timeframe_data: Dict) -> Dict:
        """[TARGET] REAL Market Regime Detection using ALL 4 timeframes"""
        try:
            # ดึงข้อมูลทั้ง 4 timeframes
            timeframes_needed = ['H4', 'H1', 'M15', 'M5']
            timeframe_dfs = {}
            
            for tf in timeframes_needed:
                tf_data = timeframe_data.get(tf) if timeframe_data else None
                
                # ถ้าไม่มีข้อมูลใน timeframe_data ให้ดึงจาก MT5
                if tf_data is None:
                    tf_mt5_map = {
                        'H4': mt5.TIMEFRAME_H4,
                        'H1': mt5.TIMEFRAME_H1,
                        'M15': mt5.TIMEFRAME_M15,
                        'M5': mt5.TIMEFRAME_M5
                    }
                    
                    periods = 100 if tf in ['H4', 'H1'] else 150  # M15, M5 ต้องการข้อมูลมากกว่า
                    tf_rates = mt5.copy_rates_from_pos(symbol, tf_mt5_map[tf], 0, periods)
                    
                    if tf_rates is not None and len(tf_rates) > 20:
                        timeframe_dfs[tf] = pd.DataFrame(tf_rates)
                        self.logger.info(f"[OK] Fetched {tf} data: {len(tf_rates)} periods for {symbol}")
                    else:
                        self.logger.warning(f"[WARN] Failed to fetch {tf} data for {symbol}")
                else:
                    if len(tf_data) > 20:
                        timeframe_dfs[tf] = tf_data
                        # self.logger.info(f"[OK] Using existing {tf} data: {len(tf_data)} periods for {symbol}")
            
            # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
            if len(timeframe_dfs) < 2:
                self.logger.warning(f"[WARN] Insufficient timeframe data for {symbol}: only {len(timeframe_dfs)} timeframes")
                return self._get_fallback_regime("Insufficient timeframe data")
            
            # [HOT] Multi-Timeframe Analysis
            regime_scores = {}
            
            # H4 Analysis - Main Trend (Weight: 40%)
            if 'H4' in timeframe_dfs:
                h4_regime = self._analyze_h4_regime(timeframe_dfs['H4'])
                regime_scores['H4'] = {'data': h4_regime, 'weight': 0.40}
                # self.logger.info(f"[CHART] H4 Regime for {symbol}: {h4_regime['regime']} (confidence: {h4_regime['confidence']:.2f})")
            
            # H1 Analysis - Setup Confirmation (Weight: 30%)
            if 'H1' in timeframe_dfs:
                h1_regime = self._analyze_h1_regime(timeframe_dfs['H1'])
                regime_scores['H1'] = {'data': h1_regime, 'weight': 0.30}
                # self.logger.info(f"[CHART] H1 Regime for {symbol}: {h1_regime['regime']} (confidence: {h1_regime['confidence']:.2f})")
            
            # M15 Analysis - Entry Timing (Weight: 20%)
            if 'M15' in timeframe_dfs:
                m15_regime = self._analyze_m15_regime(timeframe_dfs['M15'])
                regime_scores['M15'] = {'data': m15_regime, 'weight': 0.20}
                # self.logger.info(f"[CHART] M15 Regime for {symbol}: {m15_regime['regime']} (confidence: {m15_regime['confidence']:.2f})")
            
            # M5 Analysis - Risk Management (Weight: 10%)
            if 'M5' in timeframe_dfs:
                m5_regime = self._analyze_m5_regime(timeframe_dfs['M5'])
                regime_scores['M5'] = {'data': m5_regime, 'weight': 0.10}
                # self.logger.info(f"[CHART] M5 Regime for {symbol}: {m5_regime['regime']} (confidence: {m5_regime['confidence']:.2f})")
            
            # [TARGET] Calculate Final Regime using Weighted Confluence
            final_regime = self._calculate_weighted_regime_confluence(regime_scores)
            
            # เพิ่มข้อมูล debug และ metadata
            final_regime.update({
                'timeframes_analyzed': list(timeframe_dfs.keys()),
                'timeframe_count': len(timeframe_dfs),
                'regime_breakdown': {tf: scores['data']['regime'] for tf, scores in regime_scores.items()},
                'confidence_breakdown': {tf: scores['data']['confidence'] for tf, scores in regime_scores.items()},
                'calculation_method': 'MULTI_TIMEFRAME_WEIGHTED_CONFLUENCE',
                'symbol': symbol
            })
            
            # self.logger.info(f"[TARGET] Final Multi-TF Regime for {symbol}: {final_regime['regime']} (confidence: {final_regime['confidence']:.2f})")
            # self.logger.info(f"   Timeframes used: {', '.join(timeframe_dfs.keys())}")
            
            return final_regime
            
        except Exception as e:
            self.logger.error(f"[ERR] Multi-timeframe regime calculation error for {symbol}: {str(e)}")
            return self._get_fallback_regime(f"Calculation error: {str(e)}")

    def _analyze_h4_regime(self, df_h4: pd.DataFrame) -> Dict:
        """Analyze H4 timeframe for main trend direction"""
        try:
            close = df_h4['close']
            
            # Long-term EMAs for H4
            ema_20 = close.ewm(span=20).mean()
            ema_50 = close.ewm(span=50).mean()
            ema_100 = close.ewm(span=100).mean() if len(close) >= 100 else ema_50
            
            current_price = close.iloc[-1]
            
            # Strong trend detection
            strong_bullish = (current_price > ema_20.iloc[-1] > ema_50.iloc[-1] > ema_100.iloc[-1])
            strong_bearish = (current_price < ema_20.iloc[-1] < ema_50.iloc[-1] < ema_100.iloc[-1])
            
            # EMA slope analysis
            ema_20_slope = (ema_20.iloc[-1] - ema_20.iloc[-10]) / ema_20.iloc[-10] if len(ema_20) > 10 else 0
            ema_50_slope = (ema_50.iloc[-1] - ema_50.iloc[-20]) / ema_50.iloc[-20] if len(ema_50) > 20 else 0
            
            # ATR for volatility
            high, low = df_h4['high'], df_h4['low']
            tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            atr_percentile = self._calculate_atr_percentile(tr.rolling(14).mean().dropna(), atr)
            
            # Regime determination
            if strong_bullish and ema_20_slope > 0.001:
                regime = 'TRENDING_BULLISH'
                confidence = 0.85 + min(0.1, ema_20_slope * 100)
            elif strong_bearish and ema_20_slope < -0.001:
                regime = 'TRENDING_BEARISH'
                confidence = 0.85 + min(0.1, abs(ema_20_slope) * 100)
            elif atr_percentile >= 80:
                regime = 'HIGH_VOLATILITY'
                confidence = 0.80
            elif atr_percentile <= 20:
                regime = 'LOW_VOLATILITY'
                confidence = 0.75
            else:
                regime = 'RANGING'
                confidence = 0.60
            
            return {
                'regime': regime,
                'confidence': min(0.95, confidence),
                'trend_strength': abs(ema_20_slope) + abs(ema_50_slope),
                'atr_percentile': atr_percentile,
                'ema_alignment': {'bullish': strong_bullish, 'bearish': strong_bearish}
            }
            
        except Exception as e:
            return {'regime': 'ERROR', 'confidence': 0.0, 'error': str(e)}

    def _analyze_h1_regime(self, df_h1: pd.DataFrame) -> Dict:
        """Analyze H1 timeframe for setup confirmation"""
        try:
            close = df_h1['close']
            
            # Medium-term EMAs
            ema_9 = close.ewm(span=9).mean()
            ema_21 = close.ewm(span=21).mean()
            ema_50 = close.ewm(span=50).mean()
            
            current_price = close.iloc[-1]
            
            # MACD for momentum
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - macd_signal
            
            # RSI for momentum confirmation
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 0.001)
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            current_macd_hist = macd_histogram.iloc[-1]
            
            # Setup analysis
            bullish_setup = (current_price > ema_9.iloc[-1] > ema_21.iloc[-1] and 
                            current_macd_hist > 0 and 30 <= current_rsi <= 70)
            bearish_setup = (current_price < ema_9.iloc[-1] < ema_21.iloc[-1] and 
                            current_macd_hist < 0 and 30 <= current_rsi <= 70)
            
            # Regime determination
            if bullish_setup:
                regime = 'SETUP_BULLISH'
                confidence = 0.75 + min(0.15, current_macd_hist * 1000)
            elif bearish_setup:
                regime = 'SETUP_BEARISH'
                confidence = 0.75 + min(0.15, abs(current_macd_hist) * 1000)
            elif abs(current_macd_hist) < 0.0001:
                regime = 'SETUP_NEUTRAL'
                confidence = 0.60
            else:
                regime = 'SETUP_WEAK'
                confidence = 0.50
            
            return {
                'regime': regime,
                'confidence': min(0.90, confidence),
                'rsi': current_rsi,
                'macd_histogram': current_macd_hist,
                'trend_strength': abs(current_macd_hist)
            }
            
        except Exception as e:
            return {'regime': 'ERROR', 'confidence': 0.0, 'error': str(e)}

    def _analyze_m15_regime(self, df_m15: pd.DataFrame) -> Dict:
        """Analyze M15 timeframe for entry timing"""
        try:
            close = df_m15['close']
            
            # Fast EMAs for M15
            ema_5 = close.ewm(span=5).mean()
            ema_13 = close.ewm(span=13).mean()
            ema_21 = close.ewm(span=21).mean()
            
            current_price = close.iloc[-1]
            
            # Fast RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(9).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
            rs = gain / loss.replace(0, 0.001)
            rsi_fast = 100 - (100 / (1 + rs))
            current_rsi = rsi_fast.iloc[-1]
            
            # Momentum
            momentum_5 = (current_price - close.iloc[-6]) / close.iloc[-6] * 100 if len(close) > 6 else 0
            
            # Entry timing analysis
            fast_bullish = (current_price > ema_5.iloc[-1] > ema_13.iloc[-1] and 
                        25 <= current_rsi <= 75 and momentum_5 > 0.05)
            fast_bearish = (current_price < ema_5.iloc[-1] < ema_13.iloc[-1] and 
                        25 <= current_rsi <= 75 and momentum_5 < -0.05)
            
            # Regime determination
            if fast_bullish:
                regime = 'ENTRY_BULLISH'
                confidence = 0.70 + min(0.2, momentum_5 / 100)
            elif fast_bearish:
                regime = 'ENTRY_BEARISH'
                confidence = 0.70 + min(0.2, abs(momentum_5) / 100)
            else:
                regime = 'ENTRY_WAIT'
                confidence = 0.50
            
            return {
                'regime': regime,
                'confidence': min(0.90, confidence),
                'rsi_fast': current_rsi,
                'momentum_5': momentum_5,
                'trend_strength': abs(momentum_5) / 100
            }
            
        except Exception as e:
            return {'regime': 'ERROR', 'confidence': 0.0, 'error': str(e)}

    def _analyze_m5_regime(self, df_m5: pd.DataFrame) -> Dict:
        """Analyze M5 timeframe for risk management"""
        try:
            close = df_m5['close']
            
            # Very fast EMAs
            ema_3 = close.ewm(span=3).mean()
            ema_8 = close.ewm(span=8).mean()
            
            current_price = close.iloc[-1]
            
            # Immediate momentum
            momentum_3 = (current_price - close.iloc[-4]) / close.iloc[-4] * 100 if len(close) > 4 else 0
            
            # Recent price action
            last_5_candles = close.tail(5)
            is_rising = all(last_5_candles.iloc[i] <= last_5_candles.iloc[i+1] for i in range(4))
            is_falling = all(last_5_candles.iloc[i] >= last_5_candles.iloc[i+1] for i in range(4))
            
            # Support/Resistance levels
            recent_data = close.tail(20) if len(close) >= 20 else close
            resistance = recent_data.max()
            support = recent_data.min()
            
            near_resistance = abs(current_price - resistance) / current_price < 0.001
            near_support = abs(current_price - support) / current_price < 0.001
            
            # Risk management analysis
            if current_price > ema_3.iloc[-1] > ema_8.iloc[-1] and is_rising and not near_resistance:
                regime = 'MANAGE_HOLD_LONG'
                confidence = 0.75
            elif current_price < ema_3.iloc[-1] < ema_8.iloc[-1] and is_falling and not near_support:
                regime = 'MANAGE_HOLD_SHORT'
                confidence = 0.75
            elif near_resistance or near_support:
                regime = 'MANAGE_WATCH'
                confidence = 0.80
            else:
                regime = 'MANAGE_NEUTRAL'
                confidence = 0.60
            
            return {
                'regime': regime,
                'confidence': confidence,
                'momentum_3': momentum_3,
                'near_resistance': near_resistance,
                'near_support': near_support,
                'trend_strength': abs(momentum_3) / 100
            }
            
        except Exception as e:
            return {'regime': 'ERROR', 'confidence': 0.0, 'error': str(e)}

    def _calculate_weighted_regime_confluence(self, regime_scores: Dict) -> Dict:
        """Calculate final regime using weighted confluence from all timeframes"""
        try:
            # Regime mapping for scoring
            regime_mappings = {
                # Bullish regimes
                'TRENDING_BULLISH': 3,
                'SETUP_BULLISH': 2,
                'ENTRY_BULLISH': 2,
                'MANAGE_HOLD_LONG': 1,
                
                # Bearish regimes
                'TRENDING_BEARISH': -3,
                'SETUP_BEARISH': -2,
                'ENTRY_BEARISH': -2,
                'MANAGE_HOLD_SHORT': -1,
                
                # Neutral/Special regimes
                'HIGH_VOLATILITY': 0,
                'LOW_VOLATILITY': 0,
                'RANGING': 0,
                'SETUP_NEUTRAL': 0,
                'ENTRY_WAIT': 0,
                'MANAGE_WATCH': 0,
                'MANAGE_NEUTRAL': 0,
                
                # Error states
                'ERROR': 0,
                'SETUP_WEAK': 0
            }
            
            total_weighted_score = 0
            total_weight = 0
            total_confidence = 0
            volatility_indicators = []
            trend_strengths = []
            
            # Calculate weighted scores
            for tf, score_data in regime_scores.items():
                regime_data = score_data['data']
                weight = score_data['weight']
                
                regime = regime_data.get('regime', 'ERROR')
                confidence = regime_data.get('confidence', 0.0)
                trend_strength = regime_data.get('trend_strength', 0.0)
                
                # Get regime score
                regime_score = regime_mappings.get(regime, 0)
                
                # Apply weight and confidence
                weighted_score = regime_score * weight * confidence
                total_weighted_score += weighted_score
                total_weight += weight * confidence
                total_confidence += confidence * weight
                
                # Collect data for final analysis
                if 'VOLATILITY' in regime:
                    volatility_indicators.append(regime)
                
                trend_strengths.append(trend_strength)
            
            # Normalize
            if total_weight > 0:
                final_score = total_weighted_score / total_weight
                avg_confidence = total_confidence / sum(score_data['weight'] for score_data in regime_scores.values())
            else:
                final_score = 0
                avg_confidence = 0.5
            
            # Determine final regime
            if len(volatility_indicators) >= 2:
                # Multiple timeframes showing volatility
                if 'HIGH_VOLATILITY' in volatility_indicators:
                    final_regime = 'HIGH_VOLATILITY'
                    final_confidence = avg_confidence * 0.9
                else:
                    final_regime = 'LOW_VOLATILITY'
                    final_confidence = avg_confidence * 0.8
            elif final_score >= 1.5:
                final_regime = 'TRENDING_BULLISH'
                final_confidence = min(0.95, avg_confidence + 0.1)
            elif final_score <= -1.5:
                final_regime = 'TRENDING_BEARISH'
                final_confidence = min(0.95, avg_confidence + 0.1)
            elif abs(final_score) <= 0.5:
                final_regime = 'RANGING'
                final_confidence = avg_confidence
            else:
                # Weak trend
                if final_score > 0:
                    final_regime = 'TRENDING_BULLISH'
                else:
                    final_regime = 'TRENDING_BEARISH'
                final_confidence = avg_confidence * 0.8
            
            # Calculate overall trend strength
            avg_trend_strength = sum(trend_strengths) / len(trend_strengths) if trend_strengths else 0
            
            # Risk factor calculation
            risk_factor = self._calculate_risk_factor(final_regime, final_confidence)
            
            return {
                'regime': final_regime,
                'confidence': round(min(0.95, max(0.1, final_confidence)), 2),
                'trend_strength': round(avg_trend_strength, 4),
                'weighted_score': round(final_score, 2),
                'risk_factor': risk_factor,
                'calculation_method': 'WEIGHTED_MULTI_TIMEFRAME_CONFLUENCE'
            }
            
        except Exception as e:
            self.logger.error(f"[ERR] Weighted confluence calculation error: {str(e)}")
            return self._get_fallback_regime(f"Confluence error: {str(e)}")

    def _calculate_atr_percentile(self, atr_series: pd.Series, current_atr: float) -> float:
        """Calculate ATR percentile for volatility assessment"""
        try:
            if len(atr_series) < 20:
                return 50.0
            
            percentile = (atr_series <= current_atr).sum() / len(atr_series) * 100
            return round(float(percentile), 1)
        except:
            return 50.0
    
    def _calculate_risk_factor(self, regime: str, confidence: float) -> float:
        """Calculate risk adjustment factor based on regime - FIXED METHOD"""
        try:
            risk_factors = {
                'TRENDING_BULLISH': 1.2,
                'TRENDING_BEARISH': 1.2,
                'HIGH_VOLATILITY': 0.7,
                'LOW_VOLATILITY': 0.8,
                'RANGING': 0.9,
                'ERROR': 0.5,
                'UNKNOWN': 0.5
            }
            
            base_factor = risk_factors.get(regime, 1.0)
            confidence_adjustment = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
            
            return round(base_factor * confidence_adjustment, 2)
        except:
            return 1.0
    
    def _get_fallback_regime(self, reason: str = None) -> Dict:
        """Fallback regime when calculation fails - FIXED METHOD"""
        return {
            'regime': 'RANGING',
            'confidence': 0.5,
            'trend_strength': 0.0,
            'volatility_percentile': 50.0,
            'current_atr': 0.001,
            'risk_factor': 1.0,
            'calculation_method': 'FALLBACK',
            'error': reason
        }
    
    def calculate_universal_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """คำนวณ ATR - Universal version รองรับทุกโบรกเกอร์"""
        try:
            if len(df) < period + 1:
                return 0.001
                
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            if pd.isna(atr) or atr <= 0 or not np.isfinite(atr):
                price_range = df['high'].iloc[-period:].max() - df['low'].iloc[-period:].min()
                atr = price_range / period
                
            return max(0.00001, float(atr))
            
        except Exception as e:
            self.logger.error(f"Universal ATR calculation error: {str(e)}")
            return 0.001
    
    def calculate_price_volatility_regime(self, df: pd.DataFrame, period: int = 20) -> float:
        """Price Volatility สำหรับ regime detection"""
        try:
            if len(df) < period:
                return 0.5
                
            close = df['close']
            
            # คำนวณ rolling standard deviation ของ returns
            returns = close.pct_change().dropna()
            volatility = returns.rolling(window=period).std().iloc[-1]
            
            if pd.isna(volatility):
                return 0.5
                
            # Normalize ให้อยู่ในช่วง 0-1
            # ปกติ forex volatility อยู่ที่ 0.01-0.03 per day
            normalized_vol = min(1.0, max(0.0, volatility * 50))  # Scale appropriately
            
            return normalized_vol
            
        except Exception as e:
            self.logger.error(f"Price volatility calculation error: {str(e)}")
            return 0.5
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """คำนวณ Trend Strength - Universal version"""
        try:
            if len(df) < 50:
                return 0.0
                
            close = df['close']
            ema_10 = close.ewm(span=10).mean()
            ema_20 = close.ewm(span=20).mean()
            ema_50 = close.ewm(span=50).mean()
            
            current_price = close.iloc[-1]
            
            bullish_score = 0
            if current_price > ema_10.iloc[-1]: bullish_score += 1
            if ema_10.iloc[-1] > ema_20.iloc[-1]: bullish_score += 1
            if ema_20.iloc[-1] > ema_50.iloc[-1]: bullish_score += 1
            
            bearish_score = 0
            if current_price < ema_10.iloc[-1]: bearish_score += 1
            if ema_10.iloc[-1] < ema_20.iloc[-1]: bearish_score += 1
            if ema_20.iloc[-1] < ema_50.iloc[-1]: bearish_score += 1
            
            return max(bullish_score, bearish_score) / 3.0
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {str(e)}")
            return 0.0

class UniversalAdvancedSignalScorer:
    """Universal Advanced Signal Scoring System - รองรับทุกโบรกเกอร์"""
    
    def __init__(self, symbol_adapter=None):
        """Initialize with universal symbol adapter"""
        self.symbol_adapter = symbol_adapter
        self.logger = logging.getLogger(__name__)
        
        self.weights = {
            'trend_alignment': 0.3,
            'momentum': 0.25, 
            'volatility': 0.2,
            'volume': 0.15,
            'regime_fit': 0.1
        }
        
        self.regime_multipliers = {
            MarketRegime.TRENDING_BULLISH: 1.2,
            MarketRegime.TRENDING_BEARISH: 1.2,
            MarketRegime.HIGH_VOLATILITY: 0.8,
            MarketRegime.LOW_VOLATILITY: 0.6,
            MarketRegime.RANGING: 0.7
        }
    
    def calculate_enhanced_score(self, signal_data: Dict, regime_data: Dict, timeframe_analysis: Dict) -> Dict:
        """คำนวณ Enhanced Signal Score - Universal Version"""
        try:
            if not isinstance(signal_data, dict):
                signal_data = {}
            if not isinstance(regime_data, dict):
                regime_data = {'regime': MarketRegime.RANGING, 'confidence': 0.5}
            if not isinstance(timeframe_analysis, dict):
                timeframe_analysis = {}
            
            trend_score = self._calculate_trend_score_safe(timeframe_analysis)
            momentum_score = self._calculate_momentum_score_safe(signal_data, timeframe_analysis)
            volatility_score = self._calculate_volatility_score_safe(regime_data)
            volume_score = self._calculate_volume_score_safe(signal_data)
            regime_score = self._calculate_regime_fit_score_safe(signal_data, regime_data)
            
            composite_score = (
                trend_score * self.weights['trend_alignment'] +
                momentum_score * self.weights['momentum'] +
                volatility_score * self.weights['volatility'] +
                volume_score * self.weights['volume'] +
                regime_score * self.weights['regime_fit']
            )
            
            if not np.isfinite(composite_score):
                composite_score = 0.5
            
            regime = regime_data.get('regime', MarketRegime.RANGING)
            if isinstance(regime, str):
                try:
                    regime = MarketRegime(regime)
                except ValueError:
                    regime = MarketRegime.RANGING
            
            regime_multiplier = self.regime_multipliers.get(regime, 1.0)
            confidence = regime_data.get('confidence', 0.8)
            
            if not isinstance(confidence, (int, float)) or not np.isfinite(confidence):
                confidence = 0.8
            confidence = max(0.1, min(1.0, confidence))
            
            final_score = composite_score * regime_multiplier * confidence
            
            if not np.isfinite(final_score):
                final_score = composite_score
            
            enhanced_strength = max(0, min(10, final_score * 10))
            
            if enhanced_strength >= 8:
                quality = "EXCELLENT"
            elif enhanced_strength >= 6:
                quality = "GOOD"
            elif enhanced_strength >= 4:
                quality = "FAIR"
            else:
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
                'regime': regime.value if hasattr(regime, 'value') else str(regime),
                'confidence': float(confidence),
                'calculation_method': 'UNIVERSAL_ENHANCED'
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced scoring error: {str(e)}")
            return {
                'enhanced_strength': signal_data.get('strength', 0),
                'enhanced_quality': signal_data.get('entry_quality', 'POOR'),
                'error': str(e),
                'calculation_method': 'ERROR_FALLBACK'
            }
    
    def _calculate_trend_score_safe(self, timeframe_analysis: Dict) -> float:
        """Calculate trend score with universal protection"""
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
            self.logger.error(f"Trend score calculation error: {str(e)}")
            return 0.5

    def _calculate_momentum_score_safe(self, signal_data: Dict, timeframe_analysis: Dict) -> float:
        """Calculate momentum score with universal protection"""
        try:
            strength = signal_data.get('strength', 0)
            if not isinstance(strength, (int, float)) or not np.isfinite(strength):
                strength = 0
            base_momentum = max(0, min(10, strength)) / 10
            
            strong_signals = 0
            total_signals = 0
            
            for tf_name, tf_data in timeframe_analysis.items():
                if not isinstance(tf_data, dict):
                    continue
                    
                tf_signal = tf_data.get('signal', 'NONE')
                total_signals += 1
                
                if tf_signal in ['STRONG_BUY', 'STRONG_SELL', 'BUY', 'SELL']:
                    strong_signals += 1
            
            tf_momentum = strong_signals / total_signals if total_signals > 0 else 0
            
            combined_momentum = (base_momentum + tf_momentum) / 2
            return max(0.0, min(1.0, combined_momentum))
            
        except Exception as e:
            self.logger.error(f"Momentum score calculation error: {str(e)}")
            return 0.5

    def _calculate_volatility_score_safe(self, regime_data: Dict) -> float:
        """Calculate volatility score with protection"""
        try:
            volatility_percentile = regime_data.get('volatility_percentile', 50)
            if not isinstance(volatility_percentile, (int, float)) or not np.isfinite(volatility_percentile):
                volatility_percentile = 50
            
            volatility_percentile = max(0, min(100, volatility_percentile))
            
            if 30 <= volatility_percentile <= 70:
                return 1.0
            elif 20 <= volatility_percentile <= 80:
                return 0.8
            elif 10 <= volatility_percentile <= 90:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            self.logger.error(f"Volatility score calculation error: {str(e)}")
            return 0.5

    def _calculate_volume_score_safe(self, signal_data: Dict) -> float:
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
            self.logger.error(f"Volume score calculation error: {str(e)}")
            return 0.5

    def _calculate_regime_fit_score_safe(self, signal_data: Dict, regime_data: Dict) -> float:
        """Calculate regime fit score with protection"""
        try:
            signal = signal_data.get('signal', 'NONE')
            regime = regime_data.get('regime', MarketRegime.RANGING)
            
            if not isinstance(signal, str):
                signal = 'NONE'
            
            if hasattr(regime, 'value'):
                regime_str = regime.value
            else:
                regime_str = str(regime)
            
            if signal in ['BUY', 'STRONG_BUY'] and 'BULLISH' in regime_str:
                return 1.0
            elif signal in ['SELL', 'STRONG_SELL'] and 'BEARISH' in regime_str:
                return 1.0
            elif signal == 'NONE' and 'RANGING' in regime_str:
                return 0.8
            elif 'HIGH_VOLATILITY' in regime_str:
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            self.logger.error(f"Regime fit score calculation error: {str(e)}")
            return 0.5

class UniversalDynamicPositionSizer:
    """Universal Dynamic Position Sizing System - รองรับทุกโบรกเกอร์"""
    
    def __init__(self, symbol_adapter=None):
        """Initialize with universal symbol adapter"""
        self.symbol_adapter = symbol_adapter
        self.logger = logging.getLogger(__name__)
    
    def calculate_enhanced_position_size(self, account_balance: float, base_risk_percent: float,
                                    signal_data: Dict, enhanced_score: Dict, entry_price: float,
                                    stop_loss: float, symbol: str) -> Dict:
        """Calculate Position Size - Universal Version รองรับทุกโบรกเกอร์"""
        try:
            if account_balance <= 0:
                return self._get_error_result("Invalid account balance", account_balance, base_risk_percent)
            
            if base_risk_percent <= 0 or base_risk_percent > 100:
                return self._get_error_result("Invalid base risk percent", account_balance, base_risk_percent)
            
            if entry_price <= 0:
                return self._get_error_result("Invalid entry price", account_balance, base_risk_percent)
            
            if stop_loss <= 0:
                return self._get_error_result("Invalid stop loss", account_balance, base_risk_percent)
            
            points_at_risk = abs(entry_price - stop_loss)
            min_risk_threshold = entry_price * 0.0001
            
            if points_at_risk < min_risk_threshold:
                return self._get_error_result(f"Stop loss too close: {points_at_risk:.6f}", account_balance, base_risk_percent)
            
            base_risk_amount = account_balance * (base_risk_percent / 100)
            if base_risk_amount <= 0:
                return self._get_error_result("Base risk amount is zero", account_balance, base_risk_percent)
            
            enhanced_strength = enhanced_score.get('enhanced_strength', 0)
            if not isinstance(enhanced_strength, (int, float)) or not np.isfinite(enhanced_strength):
                enhanced_strength = 0
            
            signal_strength_multiplier = max(0.5, min(2.0, 0.5 + (max(0, enhanced_strength) / 20)))
            
            confidence = enhanced_score.get('confidence', 0.8)
            if not isinstance(confidence, (int, float)) or not np.isfinite(confidence):
                confidence = 0.8
            confidence_multiplier = max(0.1, min(1.0, confidence))
            
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
            
            adjusted_risk_amount = (base_risk_amount * signal_strength_multiplier * 
                                confidence_multiplier * regime_multiplier * quality_multiplier)
            
            if not np.isfinite(adjusted_risk_amount) or adjusted_risk_amount <= 0:
                adjusted_risk_amount = base_risk_amount
                
            max_risk_amount = account_balance * 0.03
            adjusted_risk_amount = min(adjusted_risk_amount, max_risk_amount)
            
            pip_info = self._get_universal_pip_info(symbol, entry_price)
            pip_size = pip_info['pip_size']
            money_per_pip = pip_info['money_per_pip']
            
            pips_at_risk = points_at_risk / pip_size
            
            if pips_at_risk <= 0 or money_per_pip <= 0:
                return self._get_error_result(f"Invalid pip calculation", account_balance, base_risk_percent)
            
            lot_size = adjusted_risk_amount / (pips_at_risk * money_per_pip)
            
            if not np.isfinite(lot_size) or lot_size <= 0:
                return self._get_error_result(f"Invalid lot size: {lot_size}", account_balance, base_risk_percent)
            
            lot_size = max(0.01, min(2.0, lot_size))
            lot_size = round(lot_size, 2)
            
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
                'pips_at_risk': round(float(pips_at_risk), 2),
                'symbol_type': pip_info['symbol_type'],
                'calculation_status': 'SUCCESS',
                'universal_calculation': True,
                'broker_symbol': self._get_broker_symbol(symbol) if self.symbol_adapter else symbol
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced position sizing error for {symbol}: {str(e)}")
            return self._get_error_result(f"Calculation error: {str(e)}", account_balance, base_risk_percent)
    
    def _get_universal_pip_info(self, symbol: str, entry_price: float) -> Dict:
        """Get pip information - Universal version รองรับทุกโบรกเกอร์"""
        try:
            broker_symbol = self._get_broker_symbol(symbol)
            symbol_to_check = broker_symbol.upper()
            
            if 'XAU' in symbol_to_check or 'GOLD' in symbol_to_check:
                return {
                    'pip_size': 0.1,
                    'money_per_pip': 1.0,
                    'symbol_type': 'GOLD'
                }
            elif 'JPY' in symbol_to_check:
                if entry_price <= 0:
                    raise ValueError("Invalid entry price for JPY pair")
                return {
                    'pip_size': 0.01,
                    'money_per_pip': 10.0 / entry_price,
                    'symbol_type': 'JPY_PAIR'
                }
            else:
                return {
                    'pip_size': 0.0001,
                    'money_per_pip': 10.0,
                    'symbol_type': 'STANDARD_FOREX'
                }
                
        except Exception as e:
            self.logger.error(f"Error getting pip info for {symbol}: {str(e)}")
            return {
                'pip_size': 0.0001,
                'money_per_pip': 10.0,
                'symbol_type': 'DEFAULT_FALLBACK'
            }
    
    def _get_broker_symbol(self, system_symbol: str) -> str:
        """Get broker-specific symbol if adapter is available"""
        if self.symbol_adapter and hasattr(self.symbol_adapter, 'get_broker_symbol'):
            try:
                broker_symbol = self.symbol_adapter.get_broker_symbol(system_symbol)
                return broker_symbol if broker_symbol else system_symbol
            except Exception as e:
                self.logger.error(f"Error getting broker symbol for {system_symbol}: {str(e)}")
        
        return system_symbol
    
    def _get_error_result(self, error_message: str, account_balance: float, base_risk_percent: float) -> Dict:
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
            'universal_calculation': True,
            'safe_fallback_used': True
        }

class UniversalAdvancedTradingIntegrator:
    """Universal Helper class สำหรับ integrate advanced features กับทุกโบรกเกอร์"""
    
    def __init__(self, symbol_adapter=None):
        """Initialize with symbol adapter for universal broker support"""
        self.symbol_adapter = symbol_adapter
        self.regime_detector = UniversalMarketRegimeDetector(symbol_adapter)
        self.signal_scorer = UniversalAdvancedSignalScorer(symbol_adapter)
        self.position_sizer = UniversalDynamicPositionSizer(symbol_adapter)
        self.logger = logging.getLogger(__name__)
        
        print("[GO] Universal Advanced Trading Features Initialized!")
        print("[OK] Universal Market Regime Detection: ACTIVE")
        print("[OK] Universal Enhanced Signal Scoring: ACTIVE") 
        print("[OK] Universal Dynamic Position Sizing: ACTIVE")
        print("[OK] Broker Compatibility: ALL BROKERS SUPPORTED")
        
        if symbol_adapter:
            print("[OK] Symbol Adapter: CONNECTED")
        else:
            print("[WARN] Symbol Adapter: NONE (using direct symbols)")

    def enhance_signal_analysis(self, symbol: str, basic_signal_data: Dict, timeframe_data: Dict = None) -> Dict:
        """[HOT] REAL Enhanced Signal Analysis - COMPLETELY FUNCTIONAL"""
        try:
            enhanced_result = basic_signal_data.copy()
            
            # [TARGET] 1. REAL Market Regime Detection
            regime_data = self.regime_detector._calculate_real_market_regime(symbol, timeframe_data)
            
            # [TARGET] 2. REAL Enhanced Strength Calculation
            enhanced_strength = self._calculate_real_enhanced_strength(basic_signal_data, regime_data)
            
            # [TARGET] 3. REAL Volatility Percentile
            volatility_percentile = self._calculate_real_volatility_percentile(symbol, timeframe_data)
            
            # [TARGET] 4. REAL Enhanced Quality Assessment
            enhanced_quality = self._calculate_real_enhanced_quality(basic_signal_data, regime_data)
            
            # [TARGET] 5. REAL Pattern Recognition
            detected_patterns = self._detect_real_patterns(symbol, timeframe_data)
            
            # [TARGET] 6. REAL Portfolio Risk Assessment
            portfolio_risk = self._assess_real_portfolio_risk(symbol)
            
            # [TARGET] 7. REAL Time-based Optimization
            time_optimization = self._calculate_time_based_optimization()
            
            # [HOT] Update with REAL calculations
            enhanced_result.update({
                # Core Enhanced Data
                'enhanced_strength': enhanced_strength,
                'enhanced_quality': enhanced_quality,
                'market_regime': regime_data['regime'],
                'regime_confidence': regime_data['confidence'],
                'trend_strength': regime_data['trend_strength'],
                'volatility_percentile': volatility_percentile,
                
                # Advanced Analytics
                'detected_patterns': detected_patterns,
                'pattern_strength': len([p for p in detected_patterns if p not in ['NO_PATTERN', 'NO_DATA', 'ERROR']]),
                'portfolio_risk_score': portfolio_risk['score'],
                'recommended_max_exposure': portfolio_risk['max_exposure'],
                'current_portfolio_exposure': portfolio_risk['current_exposure'],
                'correlation_risk': portfolio_risk.get('correlation_risk', 0.0),
                
                # Time Optimization
                'time_session_multiplier': time_optimization['session_multiplier'],
                'optimal_trading_window': time_optimization['optimal_window'],
                'session_adjusted_strength': time_optimization['adjusted_strength'],
                
                # Enhanced Position Sizing
                'enhanced_lot_size': self._calculate_enhanced_lot_size(
                    basic_signal_data.get('lot_size', 0.01), 
                    regime_data, 
                    portfolio_risk
                ),
                'risk_adjustment_factor': regime_data.get('risk_factor', 1.0),
                
                # System Information
                'universal_enhanced': True,
                'broker_symbol': symbol,
                'system_symbol': symbol,
                'enhancement_version': 'REAL_CALCULATION_v3.0_COMPLETE_FIXED',
                'enhancement_note': 'Full advanced features with real market regime detection - FIXED',
                'calculation_timestamp': datetime.now().isoformat(),
                
                # Debug Information
                'regime_debug': {
                    'atr_percent': regime_data.get('atr_percent', 0),
                    'price_vs_ema20': regime_data.get('price_vs_ema20', 0),
                    'ema_alignment': regime_data.get('ema_alignment', 0)
                }
            })
            
            # [TARGET] Log successful enhancement
            # self.logger.info(f"[OK] Enhanced analysis completed for {symbol}:")
            # self.logger.info(f"   Regime: {regime_data['regime']} (confidence: {regime_data['confidence']:.2f})")
            # self.logger.info(f"   Enhanced Strength: {enhanced_strength} (was: {basic_signal_data.get('strength', 0)})")
            # self.logger.info(f"   Quality: {enhanced_quality} (was: {basic_signal_data.get('entry_quality', 'POOR')})")
            # self.logger.info(f"   Patterns: {len(detected_patterns)} detected")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"[ERR] Enhanced analysis error for {symbol}: {str(e)}")
            
            # Enhanced error fallback
            enhanced_result = basic_signal_data.copy()
            enhanced_result.update({
                'enhanced_strength': basic_signal_data.get('strength', 0),
                'enhanced_quality': basic_signal_data.get('entry_quality', 'POOR'),
                'market_regime': 'ERROR',
                'regime_confidence': 0.0,
                'trend_strength': 0.0,
                'volatility_percentile': 50.0,
                'detected_patterns': ['ERROR'],
                'portfolio_risk_score': 'MODERATE',
                'recommended_max_exposure': 2.0,
                'enhancement_error': str(e),
                'universal_enhanced': False,
                'error_fallback_used': True,
                'broker_symbol': symbol,
                'system_symbol': symbol
            })
            return enhanced_result

    def _calculate_real_enhanced_strength(self, basic_signal_data: Dict, regime_data: Dict) -> float:
        """[TARGET] REAL Enhanced Strength with regime-based multipliers"""
        try:
            basic_strength = basic_signal_data.get('strength', 0)
            regime = regime_data.get('regime', 'RANGING')
            regime_confidence = regime_data.get('confidence', 0.5)
            trend_strength = regime_data.get('trend_strength', 0.0)
            
            # Advanced regime multipliers
            regime_multipliers = {
                'TRENDING_BULLISH': 1.4,
                'TRENDING_BEARISH': 1.4,
                'HIGH_VOLATILITY': 0.8,
                'LOW_VOLATILITY': 0.9,  # ไม่ลดมากเกินไป
                'RANGING': 1.0,
                'ERROR': 0.5,
                'UNKNOWN': 0.7
            }
            
            # คำนวณ multipliers
            regime_multiplier = regime_multipliers.get(regime, 1.0)
            confidence_multiplier = 0.8 + (regime_confidence * 0.4)  # 0.8-1.2
            trend_multiplier = 1.0 + min(trend_strength * 5, 0.5)  # สูงสุด +0.5
            
            # Signal direction alignment bonus
            signal_direction = basic_signal_data.get('signal', 'NONE')
            alignment_bonus = 1.0
            
            if signal_direction in ['BUY', 'STRONG_BUY'] and regime == 'TRENDING_BULLISH':
                alignment_bonus = 1.2
            elif signal_direction in ['SELL', 'STRONG_SELL'] and regime == 'TRENDING_BEARISH':
                alignment_bonus = 1.2
            
            # คำนวณ final strength
            enhanced_strength = (basic_strength * 
                               regime_multiplier * 
                               confidence_multiplier * 
                               trend_multiplier * 
                               alignment_bonus)
            
            # จำกัดค่าและปัดเศษ
            enhanced_strength = max(0, min(10, enhanced_strength))
            
            return round(enhanced_strength, 1)
            
        except Exception as e:
            self.logger.error(f"[ERR] Enhanced strength calculation error: {str(e)}")
            return float(basic_signal_data.get('strength', 0))

    def _calculate_real_volatility_percentile(self, symbol: str, timeframe_data: Dict) -> float:
        """[TARGET] REAL Volatility Percentile using ATR"""
        try:
            # ใช้ข้อมูลจาก timeframe_data ก่อน
            h1_data = timeframe_data.get('H1') if timeframe_data else None
            
            if h1_data is None or len(h1_data) < 50:
                # ดึงข้อมูลใหม่จาก MT5
                h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 150)
                if h1_rates is None:
                    return 50.0
                h1_data = pd.DataFrame(h1_rates)
            
            if len(h1_data) < 50:
                return 50.0
            
            # คำนวณ ATR series
            high = h1_data['high']
            low = h1_data['low']
            close = h1_data['close']
            
            tr = pd.concat([
                high - low,
                abs(high - close.shift()),
                abs(low - close.shift())
            ], axis=1).max(axis=1)
            
            atr_series = tr.rolling(14).mean().dropna()
            
            if len(atr_series) < 30:
                return 50.0
            
            current_atr = atr_series.iloc[-1]
            
            # คำนวณ percentile
            percentile = (atr_series <= current_atr).sum() / len(atr_series) * 100
            
            return round(float(percentile), 1)
            
        except Exception as e:
            self.logger.error(f"[ERR] Volatility percentile calculation error for {symbol}: {str(e)}")
            return 50.0

    def _calculate_real_enhanced_quality(self, basic_signal_data: Dict, regime_data: Dict) -> str:
        """[TARGET] REAL Enhanced Quality Assessment"""
        try:
            basic_quality = basic_signal_data.get('entry_quality', 'POOR')
            basic_strength = basic_signal_data.get('strength', 0)
            regime = regime_data.get('regime', 'RANGING')
            regime_confidence = regime_data.get('confidence', 0.5)
            trend_strength = regime_data.get('trend_strength', 0.0)
            
            # Quality score mapping
            quality_scores = {'POOR': 1, 'FAIR': 2, 'GOOD': 3, 'EXCELLENT': 4}
            base_score = quality_scores.get(basic_quality, 1)
            
            # Regime quality bonus
            regime_bonuses = {
                'TRENDING_BULLISH': 1.5,
                'TRENDING_BEARISH': 1.5,
                'HIGH_VOLATILITY': 0.5,
                'LOW_VOLATILITY': 0.8,
                'RANGING': 1.0,
                'ERROR': 0.0,
                'UNKNOWN': 0.5
            }
            
            regime_bonus = regime_bonuses.get(regime, 1.0)
            
            # Additional bonuses
            confidence_bonus = regime_confidence * 1.0
            strength_bonus = min(basic_strength / 10, 1.0)
            trend_bonus = min(trend_strength * 2, 0.5)
            
            # Calculate total score
            total_score = (base_score + 
                          regime_bonus + 
                          confidence_bonus + 
                          strength_bonus + 
                          trend_bonus)
            
            # Convert to quality rating
            if total_score >= 6.0:
                return 'EXCELLENT'
            elif total_score >= 4.5:
                return 'GOOD'
            elif total_score >= 3.0:
                return 'FAIR'
            else:
                return 'POOR'
                
        except Exception as e:
            self.logger.error(f"[ERR] Enhanced quality calculation error: {str(e)}")
            return basic_signal_data.get('entry_quality', 'POOR')

    def _detect_real_patterns(self, symbol: str, timeframe_data: Dict) -> List[str]:
        """[TARGET] REAL Pattern Detection"""
        try:
            patterns = []
            
            # ดึงข้อมูล H1 สำหรับ pattern detection
            h1_data = timeframe_data.get('H1') if timeframe_data else None
            
            if h1_data is None:
                h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 50)
                if h1_rates is not None:
                    h1_data = pd.DataFrame(h1_rates)
            
            if h1_data is None or len(h1_data) < 30:
                return ['NO_DATA']
            
            close = h1_data['close']
            high = h1_data['high']
            low = h1_data['low']
            
            # 1. Trend Patterns
            if self._detect_uptrend(close):
                patterns.append('UPTREND')
            elif self._detect_downtrend(close):
                patterns.append('DOWNTREND')
            
            # 2. Reversal Patterns
            if self._detect_double_top(high):
                patterns.append('DOUBLE_TOP')
            elif self._detect_double_bottom(low):
                patterns.append('DOUBLE_BOTTOM')
            
            # 3. Continuation Patterns
            if self._detect_flag_pattern(close, high, low):
                patterns.append('FLAG_PATTERN')
            elif self._detect_triangle_pattern(high, low):
                patterns.append('TRIANGLE')
            
            # 4. Breakout Patterns
            if self._detect_breakout_pattern(close, high, low):
                patterns.append('BREAKOUT')
            
            # 5. Consolidation
            if self._detect_consolidation_pattern(close):
                patterns.append('CONSOLIDATION')
            
            return patterns if patterns else ['NO_PATTERN']
            
        except Exception as e:
            self.logger.error(f"[ERR] Pattern detection error for {symbol}: {str(e)}")
            return ['ERROR']

    def _assess_real_portfolio_risk(self, symbol: str) -> Dict:
        """[TARGET] REAL Portfolio Risk Assessment"""
        try:
            # ดึงข้อมูล positions ปัจจุบัน
            positions = mt5.positions_get()
            
            if not positions:
                return {
                    'score': 'LOW',
                    'max_exposure': 3.0,
                    'current_exposure': 0.0,
                    'symbol_exposure': 0.0,
                    'correlation_risk': 0.0,
                    'position_count': 0
                }
            
            # คำนวณ exposure metrics
            total_volume = sum(pos.volume for pos in positions)
            symbol_volume = sum(pos.volume for pos in positions if pos.symbol == symbol)
            position_count = len(positions)
            
            # คำนวณ correlation risk
            correlation_risk = self._calculate_detailed_correlation_risk(symbol, positions)
            
            # คำนวณ drawdown risk
            total_profit = sum(pos.profit for pos in positions)
            
            # Risk scoring algorithm
            risk_factors = []
            risk_score = 0
            
            # Volume risk
            if total_volume > 5.0:
                risk_factors.append('HIGH_VOLUME')
                risk_score += 3
            elif total_volume > 2.0:
                risk_factors.append('MODERATE_VOLUME')
                risk_score += 1
            
            # Position count risk
            if position_count > 8:
                risk_factors.append('TOO_MANY_POSITIONS')
                risk_score += 2
            elif position_count > 5:
                risk_factors.append('MANY_POSITIONS')
                risk_score += 1
            
            # Correlation risk
            if correlation_risk > 0.7:
                risk_factors.append('HIGH_CORRELATION')
                risk_score += 3
            elif correlation_risk > 0.4:
                risk_factors.append('MODERATE_CORRELATION')
                risk_score += 1
            
            # Drawdown risk
            if total_profit < -1000:
                risk_factors.append('HIGH_DRAWDOWN')
                risk_score += 2
            elif total_profit < -500:
                risk_factors.append('MODERATE_DRAWDOWN')
                risk_score += 1
            
            # Determine risk level and max exposure
            if risk_score >= 6:
                risk_level = 'VERY_HIGH'
                max_exposure = 0.5
            elif risk_score >= 4:
                risk_level = 'HIGH'
                max_exposure = 1.0
            elif risk_score >= 2:
                risk_level = 'MODERATE'
                max_exposure = 2.0
            else:
                risk_level = 'LOW'
                max_exposure = 3.0
            
            return {
                'score': risk_level,
                'max_exposure': max_exposure,
                'current_exposure': round(total_volume, 2),
                'symbol_exposure': round(symbol_volume, 2),
                'correlation_risk': round(correlation_risk, 2),
                'position_count': position_count,
                'total_profit': round(total_profit, 2),
                'risk_factors': risk_factors,
                'risk_score': risk_score
            }
            
        except Exception as e:
            self.logger.error(f"[ERR] Portfolio risk assessment error: {str(e)}")
            return {
                'score': 'MODERATE',
                'max_exposure': 2.0,
                'current_exposure': 0.0,
                'correlation_risk': 0.5,
                'error': str(e)
            }

    def _calculate_time_based_optimization(self) -> Dict:
        """[TARGET] Time-based trading optimization"""
        try:
            current_time = datetime.now()
            current_hour = current_time.hour
            current_weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            
            # Session definitions (UTC time)
            sessions = {
                'ASIAN': {'start': 0, 'end': 9, 'multiplier': 0.8},
                'LONDON': {'start': 8, 'end': 17, 'multiplier': 1.3},
                'NY': {'start': 13, 'end': 22, 'multiplier': 1.4},
                'OVERLAP': {'start': 13, 'end': 17, 'multiplier': 1.5}
            }
            
            # Weekend penalty
            weekend_penalty = 0.7 if current_weekday >= 5 else 1.0
            
            # Determine active session
            active_session = 'QUIET'
            session_multiplier = 0.6
            
            for session_name, session_info in sessions.items():
                start_hour = session_info['start']
                end_hour = session_info['end']
                
                if start_hour <= current_hour <= end_hour:
                    active_session = session_name
                    session_multiplier = session_info['multiplier']
                    break
            
            # Apply weekend penalty
            final_multiplier = session_multiplier * weekend_penalty
            
            # Calculate session strength
            if final_multiplier >= 1.4:
                optimal_window = 'EXCELLENT'
            elif final_multiplier >= 1.2:
                optimal_window = 'GOOD'
            elif final_multiplier >= 1.0:
                optimal_window = 'FAIR'
            else:
                optimal_window = 'POOR'
            
            return {
                'session_multiplier': round(final_multiplier, 2),
                'optimal_window': optimal_window,
                'active_session': active_session,
                'current_hour': current_hour,
                'is_weekend': current_weekday >= 5,
                'adjusted_strength': 0  # Will be calculated when applied
            }
            
        except Exception as e:
            self.logger.error(f"[ERR] Time optimization error: {str(e)}")
            return {
                'session_multiplier': 1.0,
                'optimal_window': 'STANDARD',
                'active_session': 'UNKNOWN',
                'error': str(e)
            }

    def _calculate_enhanced_lot_size(self, base_lot_size: float, regime_data: Dict, portfolio_risk: Dict) -> float:
        """[TARGET] Enhanced position sizing with regime and risk adjustments"""
        try:
            regime = regime_data.get('regime', 'RANGING')
            regime_confidence = regime_data.get('confidence', 0.5)
            risk_level = portfolio_risk.get('score', 'MODERATE')
            
            # Base adjustments
            regime_adjustments = {
                'TRENDING_BULLISH': 1.2,
                'TRENDING_BEARISH': 1.2,
                'HIGH_VOLATILITY': 0.7,
                'LOW_VOLATILITY': 0.9,
                'RANGING': 1.0,
                'ERROR': 0.5
            }
            
            risk_adjustments = {
                'LOW': 1.0,
                'MODERATE': 0.8,
                'HIGH': 0.6,
                'VERY_HIGH': 0.4
            }
            
            # Calculate adjustments
            regime_adj = regime_adjustments.get(regime, 1.0)
            risk_adj = risk_adjustments.get(risk_level, 0.8)
            confidence_adj = 0.7 + (regime_confidence * 0.6)  # 0.7-1.3
            
            # Apply adjustments
            enhanced_lot_size = base_lot_size * regime_adj * risk_adj * confidence_adj
            
            # Apply limits
            enhanced_lot_size = max(0.01, min(2.0, enhanced_lot_size))
            
            return round(enhanced_lot_size, 2)
            
        except Exception as e:
            self.logger.error(f"[ERR] Enhanced lot size calculation error: {str(e)}")
            return base_lot_size

    # Pattern Detection Helper Methods
    def _detect_uptrend(self, close: pd.Series) -> bool:
        try:
            if len(close) < 20:
                return False
            recent = close.tail(20)
            return recent.iloc[-1] > recent.iloc[0] and recent.rolling(5).mean().is_monotonic_increasing
        except:
            return False

    def _detect_downtrend(self, close: pd.Series) -> bool:
        try:
            if len(close) < 20:
                return False
            recent = close.tail(20)
            return recent.iloc[-1] < recent.iloc[0] and recent.rolling(5).mean().is_monotonic_decreasing
        except:
            return False

    def _detect_double_top(self, high: pd.Series) -> bool:
        try:
            recent_highs = high.tail(30)
            max_high = recent_highs.max()
            high_peaks = recent_highs[recent_highs >= max_high * 0.998]
            return len(high_peaks) >= 2
        except:
            return False

    def _detect_double_bottom(self, low: pd.Series) -> bool:
        try:
            recent_lows = low.tail(30)
            min_low = recent_lows.min()
            low_valleys = recent_lows[recent_lows <= min_low * 1.002]
            return len(low_valleys) >= 2
        except:
            return False

    def _detect_flag_pattern(self, close: pd.Series, high: pd.Series, low: pd.Series) -> bool:
        try:
            if len(close) < 30:
                return False
            recent = close.tail(20)
            range_pct = (recent.max() - recent.min()) / recent.mean() * 100
            return 0.5 <= range_pct <= 2.0
        except:
            return False

    def _detect_triangle_pattern(self, high: pd.Series, low: pd.Series) -> bool:
        try:
            if len(high) < 30:
                return False
            recent_highs = high.tail(20)
            recent_lows = low.tail(20)
            high_trend = recent_highs.iloc[-1] < recent_highs.iloc[0]
            low_trend = recent_lows.iloc[-1] > recent_lows.iloc[0]
            return high_trend and low_trend
        except:
            return False

    def _detect_breakout_pattern(self, close: pd.Series, high: pd.Series, low: pd.Series) -> bool:
        try:
            if len(close) < 40:
                return False
            baseline = close.iloc[-40:-10]
            resistance = baseline.max()
            support = baseline.min()
            current = close.iloc[-1]
            return current > resistance * 1.002 or current < support * 0.998
        except:
            return False

    def _detect_consolidation_pattern(self, close: pd.Series) -> bool:
        try:
            if len(close) < 20:
                return False
            recent = close.tail(20)
            volatility = recent.std() / recent.mean()
            return volatility < 0.01
        except:
            return False

    def _calculate_detailed_correlation_risk(self, symbol: str, positions) -> float:
        try:
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            
            correlation_score = 0
            total_volume = sum(pos.volume for pos in positions)
            
            for pos in positions:
                if len(pos.symbol) >= 6:
                    pos_base = pos.symbol[:3]
                    pos_quote = pos.symbol[3:6]
                    
                    # Calculate correlation weight
                    weight = pos.volume / total_volume if total_volume > 0 else 0
                    
                    # Direct correlation
                    if pos_base == base_currency or pos_quote == quote_currency:
                        correlation_score += weight * 0.8
                    elif pos_base == quote_currency or pos_quote == base_currency:
                        correlation_score += weight * 0.6
                    
                    # Cross-correlation (EUR-GBP, GBP-JPY etc.)
                    major_currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
                    if (base_currency in major_currencies and 
                        (pos_base in major_currencies or pos_quote in major_currencies)):
                        correlation_score += weight * 0.3
            
            return min(1.0, correlation_score)
        except:
            return 0.5

    def get_dashboard_data(self) -> Dict:
        """[TARGET] Dashboard data for monitoring"""
        try:
            return {
                'advanced_features_active': True,
                'regime_detector_status': 'ACTIVE',
                'signal_scorer_status': 'ACTIVE',
                'position_sizer_status': 'ACTIVE',
                'pattern_recognition_status': 'ACTIVE',
                'universal_compatibility': True,
                'broker_adapter_connected': self.symbol_adapter is not None,
                'enhancement_version': '3.0_COMPLETE_FIXED',
                'last_update': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e), 'advanced_features_active': False}

    def get_enhancement_status(self) -> Dict:
        """Get comprehensive enhancement status"""
        return {
            'universal_features_active': True,
            'regime_detector': 'ACTIVE',
            'signal_scorer': 'ACTIVE',
            'position_sizer': 'ACTIVE',
            'pattern_recognition': 'ACTIVE',
            'portfolio_risk_management': 'ACTIVE',
            'time_based_optimization': 'ACTIVE',
            'symbol_adapter_connected': self.symbol_adapter is not None,
            'broker_compatibility': 'ALL_BROKERS',
            'enhancement_version': '3.0_COMPLETE_REAL_CALCULATIONS_FIXED',
            'features': [
                'REAL_MARKET_REGIME_DETECTION',
                'REAL_ENHANCED_SIGNAL_SCORING',
                'REAL_PATTERN_RECOGNITION',
                'REAL_PORTFOLIO_RISK_ASSESSMENT',
                'REAL_TIME_OPTIMIZATION',
                'REAL_ENHANCED_POSITION_SIZING'
            ]
        }

__all__ = [
    'UniversalAdvancedTradingIntegrator',
    'UniversalMarketRegimeDetector', 
    'UniversalAdvancedSignalScorer',
    'UniversalDynamicPositionSizer',
    'MarketRegime',
    'clean_data_for_json'
]