"""
Advanced Correlation-Based Hedging System
========================================
Intelligent Cross-Pair Hedging using Currency Correlation Analysis
Professional-grade risk management through correlation matrices
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import MetaTrader5 as mt5
from dataclasses import dataclass
from enum import Enum
import json
import logging
from flask import jsonify, request

class HedgeAction(Enum):
    NO_HEDGE = "NO_HEDGE"
    PARTIAL_HEDGE = "PARTIAL_HEDGE"
    FULL_HEDGE = "FULL_HEDGE"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"
    CLOSE_CONFLICTING = "CLOSE_CONFLICTING"

class CorrelationType(Enum):
    POSITIVE = "POSITIVE"  # +0.7 to +1.0
    NEGATIVE = "NEGATIVE"  # -0.7 to -1.0
    NEUTRAL = "NEUTRAL"    # -0.3 to +0.3
    MODERATE_POS = "MODERATE_POS"  # +0.3 to +0.7
    MODERATE_NEG = "MODERATE_NEG"  # -0.7 to -0.3

@dataclass
class HedgeOpportunity:
    primary_symbol: str
    hedge_symbol: str
    correlation_coefficient: float
    correlation_type: CorrelationType
    hedge_action: HedgeAction
    hedge_ratio: float
    confidence_score: float
    expected_risk_reduction: float
    execution_priority: int
    reasoning: List[str]

class AdvancedCorrelationHedging:
    """
    🎯 Advanced Correlation-Based Hedging System
    ==========================================
    Smart cross-pair hedging using real-time correlation analysis
    """
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.logger = logging.getLogger(__name__)
        
        # 📊 ENHANCED CORRELATION MATRIX
        self.correlation_pairs = {
            # 💰 USD STRENGTH BASKET
            'EURUSD.c': {
                'negative_corr': ['USDCHF.c', 'USDJPY.c', 'USDCAD.c'],
                'positive_corr': ['GBPUSD.c', 'AUDUSD.c', 'NZDUSD.c'],
                'hedge_coefficients': {
                    'USDCHF.c': -0.85, 'USDJPY.c': -0.75, 'USDCAD.c': -0.70,
                    'GBPUSD.c': 0.82, 'AUDUSD.c': 0.75, 'NZDUSD.c': 0.72
                }
            },
            
            # 🇬🇧 GBP STRENGTH BASKET  
            'GBPUSD.c': {
                'negative_corr': ['USDCHF.c', 'USDJPY.c'],
                'positive_corr': ['EURUSD.c', 'GBPJPY.c', 'GBPCHF.c', 'GBPAUD.c', 'GBPNZD.c'],
                'hedge_coefficients': {
                    'USDCHF.c': -0.80, 'USDJPY.c': -0.65,
                    'EURUSD.c': 0.82, 'GBPJPY.c': 0.90, 'GBPCHF.c': 0.88,
                    'GBPAUD.c': 0.75, 'GBPNZD.c': 0.78
                }
            },
            
            # 🇯🇵 JPY STRENGTH BASKET
            'USDJPY.c': {
                'negative_corr': ['EURUSD.c', 'GBPUSD.c', 'AUDUSD.c'],
                'positive_corr': ['USDCHF.c', 'EURJPY.c', 'GBPJPY.c', 'AUDJPY.c', 'NZDJPY.c'],
                'hedge_coefficients': {
                    'EURUSD.c': -0.75, 'GBPUSD.c': -0.65, 'AUDUSD.c': -0.60,
                    'USDCHF.c': 0.70, 'EURJPY.c': 0.85, 'GBPJPY.c': 0.82,
                    'AUDJPY.c': 0.78, 'NZDJPY.c': 0.80
                }
            },
            
            # 🇨🇭 CHF STRENGTH BASKET
            'USDCHF.c': {
                'negative_corr': ['EURUSD.c', 'GBPUSD.c', 'AUDUSD.c'],
                'positive_corr': ['USDJPY.c', 'EURCHF.c', 'GBPCHF.c'],
                'hedge_coefficients': {
                    'EURUSD.c': -0.85, 'GBPUSD.c': -0.80, 'AUDUSD.c': -0.72,
                    'USDJPY.c': 0.70, 'EURCHF.c': 0.88, 'GBPCHF.c': 0.85
                }
            },
            
            # 🇦🇺 COMMODITY CURRENCIES
            'AUDUSD.c': {
                'negative_corr': ['USDCHF.c', 'USDJPY.c'],
                'positive_corr': ['NZDUSD.c', 'EURUSD.c', 'AUDNZD.c', 'AUDJPY.c', 'AUDCHF.c'],
                'hedge_coefficients': {
                    'USDCHF.c': -0.72, 'USDJPY.c': -0.60,
                    'NZDUSD.c': 0.85, 'EURUSD.c': 0.75, 'AUDNZD.c': 0.80,
                    'AUDJPY.c': 0.78, 'AUDCHF.c': 0.75
                }
            },
            
            # 🇳🇿 NZD STRENGTH BASKET
            'NZDUSD.c': {
                'negative_corr': ['USDCHF.c', 'USDJPY.c'],
                'positive_corr': ['AUDUSD.c', 'EURUSD.c', 'AUDNZD.c', 'NZDJPY.c', 'NZDCHF.c'],
                'hedge_coefficients': {
                    'USDCHF.c': -0.68, 'USDJPY.c': -0.58,
                    'AUDUSD.c': 0.85, 'EURUSD.c': 0.72, 'AUDNZD.c': 0.80,
                    'NZDJPY.c': 0.75, 'NZDCHF.c': 0.70
                }
            },
            
            # 🇨🇦 CAD STRENGTH BASKET
            'USDCAD.c': {
                'negative_corr': ['EURUSD.c', 'GBPUSD.c'],
                'positive_corr': ['USDJPY.c', 'CADJPY.c', 'AUDCAD.c'],
                'hedge_coefficients': {
                    'EURUSD.c': -0.70, 'GBPUSD.c': -0.65,
                    'USDJPY.c': 0.65, 'CADJPY.c': 0.85, 'AUDCAD.c': 0.75
                }
            },
            
            # 🇪🇺 EUR CROSS PAIRS
            'EURGBP.c': {
                'negative_corr': ['GBPCHF.c'],
                'positive_corr': ['EURUSD.c', 'GBPUSD.c', 'EURJPY.c', 'GBPJPY.c'],
                'hedge_coefficients': {
                    'GBPCHF.c': -0.60,
                    'EURUSD.c': 0.70, 'GBPUSD.c': 0.75, 'EURJPY.c': 0.80, 'GBPJPY.c': 0.78
                }
            },
            
            'EURJPY.c': {
                'negative_corr': ['USDCHF.c'],
                'positive_corr': ['USDJPY.c', 'EURUSD.c', 'GBPJPY.c', 'AUDJPY.c', 'NZDJPY.c'],
                'hedge_coefficients': {
                    'USDCHF.c': -0.55,
                    'USDJPY.c': 0.85, 'EURUSD.c': 0.75, 'GBPJPY.c': 0.88,
                    'AUDJPY.c': 0.82, 'NZDJPY.c': 0.80
                }
            },
            
            'EURCHF.c': {
                'negative_corr': ['EURUSD.c'],
                'positive_corr': ['USDCHF.c', 'GBPCHF.c', 'AUDCHF.c'],
                'hedge_coefficients': {
                    'EURUSD.c': -0.60,
                    'USDCHF.c': 0.88, 'GBPCHF.c': 0.85, 'AUDCHF.c': 0.78
                }
            },
            
            'EURAUD.c': {
                'negative_corr': ['AUDUSD.c'],
                'positive_corr': ['EURUSD.c', 'AUDNZD.c', 'GBPAUD.c'],
                'hedge_coefficients': {
                    'AUDUSD.c': -0.65,
                    'EURUSD.c': 0.70, 'AUDNZD.c': 0.75, 'GBPAUD.c': 0.72
                }
            },
            
            'EURNZD.c': {
                'negative_corr': ['NZDUSD.c'],
                'positive_corr': ['EURUSD.c', 'AUDNZD.c', 'GBPNZD.c'],
                'hedge_coefficients': {
                    'NZDUSD.c': -0.62,
                    'EURUSD.c': 0.68, 'AUDNZD.c': 0.78, 'GBPNZD.c': 0.75
                }
            },
            
            'EURCAD.c': {
                'negative_corr': ['USDCAD.c'],
                'positive_corr': ['EURUSD.c', 'GBPCAD.c', 'AUDCAD.c'],
                'hedge_coefficients': {
                    'USDCAD.c': -0.68,
                    'EURUSD.c': 0.72, 'GBPCAD.c': 0.78, 'AUDCAD.c': 0.70
                }
            },
            
            # 🇬🇧 GBP CROSS PAIRS
            'GBPJPY.c': {
                'negative_corr': ['USDCHF.c'],
                'positive_corr': ['GBPUSD.c', 'USDJPY.c', 'EURJPY.c', 'AUDJPY.c'],
                'hedge_coefficients': {
                    'USDCHF.c': -0.55,
                    'GBPUSD.c': 0.90, 'USDJPY.c': 0.82, 'EURJPY.c': 0.88, 'AUDJPY.c': 0.85
                }
            },
            
            'GBPCHF.c': {
                'negative_corr': ['EURGBP.c'],
                'positive_corr': ['GBPUSD.c', 'USDCHF.c', 'EURCHF.c'],
                'hedge_coefficients': {
                    'EURGBP.c': -0.60,
                    'GBPUSD.c': 0.88, 'USDCHF.c': 0.85, 'EURCHF.c': 0.85
                }
            },
            
            'GBPAUD.c': {
                'negative_corr': ['AUDUSD.c'],
                'positive_corr': ['GBPUSD.c', 'EURAUD.c', 'AUDNZD.c'],
                'hedge_coefficients': {
                    'AUDUSD.c': -0.62,
                    'GBPUSD.c': 0.75, 'EURAUD.c': 0.72, 'AUDNZD.c': 0.70
                }
            },
            
            'GBPNZD.c': {
                'negative_corr': ['NZDUSD.c'],
                'positive_corr': ['GBPUSD.c', 'EURNZD.c', 'AUDNZD.c'],
                'hedge_coefficients': {
                    'NZDUSD.c': -0.60,
                    'GBPUSD.c': 0.78, 'EURNZD.c': 0.75, 'AUDNZD.c': 0.82
                }
            },
            
            'GBPCAD.c': {
                'negative_corr': ['USDCAD.c'],
                'positive_corr': ['GBPUSD.c', 'EURCAD.c', 'AUDCAD.c'],
                'hedge_coefficients': {
                    'USDCAD.c': -0.65,
                    'GBPUSD.c': 0.80, 'EURCAD.c': 0.78, 'AUDCAD.c': 0.72
                }
            },
            
            # 🇦🇺 AUD CROSS PAIRS
            'AUDCHF.c': {
                'negative_corr': ['AUDUSD.c'],
                'positive_corr': ['USDCHF.c', 'EURCHF.c', 'GBPCHF.c'],
                'hedge_coefficients': {
                    'AUDUSD.c': -0.65,
                    'USDCHF.c': 0.78, 'EURCHF.c': 0.78, 'GBPCHF.c': 0.75
                }
            },
            
            'AUDJPY.c': {
                'negative_corr': ['USDCHF.c'],
                'positive_corr': ['AUDUSD.c', 'USDJPY.c', 'EURJPY.c', 'GBPJPY.c', 'NZDJPY.c'],
                'hedge_coefficients': {
                    'USDCHF.c': -0.58,
                    'AUDUSD.c': 0.78, 'USDJPY.c': 0.78, 'EURJPY.c': 0.82,
                    'GBPJPY.c': 0.85, 'NZDJPY.c': 0.88
                }
            },
            
            'AUDNZD.c': {
                'negative_corr': [],
                'positive_corr': ['AUDUSD.c', 'NZDUSD.c', 'EURAUD.c', 'EURNZD.c'],
                'hedge_coefficients': {
                    'AUDUSD.c': 0.80, 'NZDUSD.c': 0.80, 'EURAUD.c': 0.75, 'EURNZD.c': 0.78
                }
            },
            
            'AUDCAD.c': {
                'negative_corr': ['USDCAD.c'],
                'positive_corr': ['AUDUSD.c', 'EURCAD.c', 'GBPCAD.c'],
                'hedge_coefficients': {
                    'USDCAD.c': -0.68,
                    'AUDUSD.c': 0.72, 'EURCAD.c': 0.70, 'GBPCAD.c': 0.72
                }
            },
            
            # 🇳🇿 NZD CROSS PAIRS
            'NZDJPY.c': {
                'negative_corr': ['USDCHF.c', 'USDJPY.c'],
                'positive_corr': ['AUDJPY.c', 'EURJPY.c', 'GBPJPY.c', 'NZDUSD.c'],
                'hedge_coefficients': {
                    'USDCHF.c': -0.60, 'USDJPY.c': -0.55,
                    'AUDJPY.c': 0.88, 'EURJPY.c': 0.80, 'GBPJPY.c': 0.82, 'NZDUSD.c': 0.75
                }
            },
            
            'NZDCHF.c': {
                'negative_corr': ['NZDUSD.c'],
                'positive_corr': ['USDCHF.c', 'AUDCHF.c', 'EURCHF.c'],
                'hedge_coefficients': {
                    'NZDUSD.c': -0.62,
                    'USDCHF.c': 0.75, 'AUDCHF.c': 0.80, 'EURCHF.c': 0.75
                }
            },
            
            'NZDCAD.c': {
                'negative_corr': ['USDCAD.c'],
                'positive_corr': ['NZDUSD.c', 'AUDCAD.c', 'EURCAD.c'],
                'hedge_coefficients': {
                    'USDCAD.c': -0.65,
                    'NZDUSD.c': 0.70, 'AUDCAD.c': 0.78, 'EURCAD.c': 0.72
                }
            },
            
            # 🇨🇭 CHF CROSS PAIRS
            'CHFJPY.c': {
                'negative_corr': ['USDCHF.c'],
                'positive_corr': ['USDJPY.c', 'EURJPY.c', 'GBPJPY.c'],
                'hedge_coefficients': {
                    'USDCHF.c': -0.70,
                    'USDJPY.c': 0.75, 'EURJPY.c': 0.78, 'GBPJPY.c': 0.75
                }
            },
            
            # 🇨🇦 CAD CROSS PAIRS
            'CADJPY.c': {
                'negative_corr': ['USDCAD.c'],
                'positive_corr': ['USDJPY.c', 'EURJPY.c', 'GBPJPY.c'],
                'hedge_coefficients': {
                    'USDCAD.c': -0.68,
                    'USDJPY.c': 0.75, 'EURJPY.c': 0.72, 'GBPJPY.c': 0.70
                }
            },
            
            # 🥇 GOLD CORRELATIONS
            'XAUUSD.c': {
                'negative_corr': ['USDJPY.c', 'USDCHF.c'],
                'positive_corr': ['EURUSD.c', 'GBPUSD.c'],
                'hedge_coefficients': {
                    'USDJPY.c': -0.45, 'USDCHF.c': -0.40,
                    'EURUSD.c': 0.55, 'GBPUSD.c': 0.50
                }
            }
        }        
        # 🎯 HEDGING PARAMETERS
        self.hedge_thresholds = {
            'min_correlation': 0.60,      # ความสัมพันธ์ขั้นต่ำ
            'max_hedge_ratio': 0.80,      # อัตราส่วน hedge สูงสุด
            'min_confidence': 0.70,       # ความมั่นใจขั้นต่ำ
            'risk_reduction_target': 0.30  # เป้าหมายลดความเสี่ยง 30%
        }
        
        # 📈 REAL-TIME CORRELATION TRACKING
        self.live_correlations = {}
        self.correlation_history = {}
        
        print("🎯 Advanced Correlation Hedging System Initialized!")
        print("💱 Smart Cross-Pair Risk Management Active")
        print("🛡️ Dynamic Hedge Ratio Calculation Ready")
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for compatibility"""
        try:
            # Remove .c suffix if exists
            clean_symbol = symbol.replace('.c', '')
            
            # Add .c suffix for consistency
            if not clean_symbol.endswith('.c'):
                return clean_symbol + '.c'
            return symbol
        except Exception as e:
            self.logger.error(f"Symbol normalization error: {str(e)}")
            return symbol


    def calculate_live_correlation(self, symbol1: str, symbol2: str, 
                                 periods: int = 50) -> float:
        """
        คำนวณ correlation แบบ real-time
        """
        try:
            # ดึงข้อมูลราคา H1 สำหรับ correlation
            rates1 = mt5.copy_rates_from_pos(symbol1, mt5.TIMEFRAME_H1, 0, periods)
            rates2 = mt5.copy_rates_from_pos(symbol2, mt5.TIMEFRAME_H1, 0, periods)
            
            if rates1 is None or rates2 is None or len(rates1) < 20 or len(rates2) < 20:
                # ใช้ correlation จาก matrix ถ้าไม่มีข้อมูลเพียงพอ
                return self._get_historical_correlation(symbol1, symbol2)
            
            # แปลงเป็น price changes
            df1 = pd.DataFrame(rates1)
            df2 = pd.DataFrame(rates2)
            
            returns1 = df1['close'].pct_change().dropna()
            returns2 = df2['close'].pct_change().dropna()
            
            # คำนวณ correlation
            if len(returns1) >= 10 and len(returns2) >= 10:
                # ตัดให้เท่ากัน
                min_len = min(len(returns1), len(returns2))
                correlation = returns1.iloc[-min_len:].corr(returns2.iloc[-min_len:])
                
                # Validate correlation
                if pd.isna(correlation) or not np.isfinite(correlation):
                    return self._get_historical_correlation(symbol1, symbol2)
                
                return float(correlation)
            else:
                return self._get_historical_correlation(symbol1, symbol2)
                
        except Exception as e:
            self.logger.error(f"Correlation calculation error {symbol1}-{symbol2}: {str(e)}")
            return self._get_historical_correlation(symbol1, symbol2)
    
    def _get_historical_correlation(self, symbol1: str, symbol2: str) -> float:
        """ใช้ correlation จาก historical matrix"""
        try:
            if symbol1 in self.correlation_pairs:
                coeffs = self.correlation_pairs[symbol1].get('hedge_coefficients', {})
                if symbol2 in coeffs:
                    return coeffs[symbol2]
            
            if symbol2 in self.correlation_pairs:
                coeffs = self.correlation_pairs[symbol2].get('hedge_coefficients', {})
                if symbol1 in coeffs:
                    return coeffs[symbol1]
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Historical correlation error: {str(e)}")
            return 0.0
    
    def classify_correlation(self, correlation: float) -> CorrelationType:
        """จำแนกประเภท correlation"""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.70:
            return CorrelationType.POSITIVE if correlation > 0 else CorrelationType.NEGATIVE
        elif abs_corr >= 0.30:
            return CorrelationType.MODERATE_POS if correlation > 0 else CorrelationType.MODERATE_NEG
        else:
            return CorrelationType.NEUTRAL
    
    def calculate_optimal_hedge_ratio(self, primary_symbol: str, hedge_symbol: str,
                                    correlation: float, primary_exposure: float) -> float:
        """
        คำนวณอัตราส่วน hedge ที่เหมาะสม
        """
        try:
            abs_correlation = abs(correlation)
            
            # Base hedge ratio จาก correlation strength
            if abs_correlation >= 0.80:
                base_ratio = 0.70  # High correlation = high hedge
            elif abs_correlation >= 0.60:
                base_ratio = 0.50  # Medium correlation = medium hedge
            elif abs_correlation >= 0.40:
                base_ratio = 0.30  # Low correlation = low hedge
            else:
                return 0.0  # No hedge for very low correlation
            
            # ปรับตามขนาด exposure
            exposure_multiplier = min(1.0, primary_exposure / 0.02)  # Normalize to 2%
            
            # ปรับตามประเภท correlation
            if correlation < 0:  # Negative correlation = better hedge
                correlation_multiplier = 1.2
            else:  # Positive correlation = partial hedge only
                correlation_multiplier = 0.8
            
            # คำนวณ optimal ratio
            optimal_ratio = base_ratio * exposure_multiplier * correlation_multiplier
            
            # จำกัดขอบเขต
            return min(self.hedge_thresholds['max_hedge_ratio'], max(0.1, optimal_ratio))
            
        except Exception as e:
            self.logger.error(f"Hedge ratio calculation error: {str(e)}")
            return 0.2  # Safe default
    
    def analyze_hedging_opportunities(self, current_positions: List[Dict]) -> List[HedgeOpportunity]:
        """
        🎯 วิเคราะห์โอกาสในการ hedge
        """
        opportunities = []
        
        try:
            if not current_positions:
                return opportunities
            
            for position in current_positions:
                primary_symbol = position['symbol']
                primary_volume = position['volume']
                primary_type = position['type']  # 'BUY' or 'SELL'
                
                # หาคู่ที่เหมาะสมสำหรับ hedge
                hedge_candidates = self._find_hedge_candidates(primary_symbol)
                
                for hedge_symbol in hedge_candidates:
                    try:
                        # คำนวณ live correlation
                        correlation = self.calculate_live_correlation(primary_symbol, hedge_symbol)
                        correlation_type = self.classify_correlation(correlation)
                        
                        # ข้าม correlation ที่ต่ำเกินไป
                        if abs(correlation) < self.hedge_thresholds['min_correlation']:
                            continue
                        
                        # ตรวจสอบว่ามี position ใน hedge symbol อยู่แล้วหรือไม่
                        existing_hedge = self._check_existing_position(hedge_symbol, current_positions)
                        
                        # คำนวณ hedge ratio
                        primary_exposure = abs(primary_volume * 0.01)  # Simplified exposure calc
                        hedge_ratio = self.calculate_optimal_hedge_ratio(
                            primary_symbol, hedge_symbol, correlation, primary_exposure
                        )
                        
                        # กำหนด hedge action
                        hedge_action = self._determine_hedge_action(
                            correlation, existing_hedge, primary_type
                        )
                        
                        # คำนวณ confidence score
                        confidence = self._calculate_confidence_score(
                            correlation, correlation_type, hedge_symbol
                        )
                        
                        # ประเมิน risk reduction
                        risk_reduction = self._estimate_risk_reduction(
                            correlation, hedge_ratio
                        )
                        
                        # สร้าง reasoning
                        reasoning = self._build_hedge_reasoning(
                            primary_symbol, hedge_symbol, correlation, 
                            correlation_type, hedge_action
                        )
                        
                        # สร้าง HedgeOpportunity
                        opportunity = HedgeOpportunity(
                            primary_symbol=primary_symbol,
                            hedge_symbol=hedge_symbol,
                            correlation_coefficient=correlation,
                            correlation_type=correlation_type,
                            hedge_action=hedge_action,
                            hedge_ratio=hedge_ratio,
                            confidence_score=confidence,
                            expected_risk_reduction=risk_reduction,
                            execution_priority=self._calculate_priority(confidence, risk_reduction),
                            reasoning=reasoning
                        )
                        
                        opportunities.append(opportunity)
                        
                    except Exception as e:
                        self.logger.error(f"Error analyzing hedge for {hedge_symbol}: {str(e)}")
                        continue
            
            # เรียงตาม priority
            opportunities.sort(key=lambda x: x.execution_priority, reverse=True)
            
            return opportunities[:5]  # Top 5 opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing hedging opportunities: {str(e)}")
            return []
    
    def _find_hedge_candidates(self, primary_symbol: str) -> List[str]:
        """หาคู่ที่เป็นไปได้สำหรับ hedge"""
        candidates = []
        
        if primary_symbol in self.correlation_pairs:
            corr_data = self.correlation_pairs[primary_symbol]
            candidates.extend(corr_data.get('negative_corr', []))
            candidates.extend(corr_data.get('positive_corr', []))
        
        # หาจาก currency components
        base_currency = primary_symbol[:3]
        quote_currency = primary_symbol[3:6]
        
        for symbol in self.trading_system.forex_pairs:
            if symbol == primary_symbol:
                continue
            
            symbol_base = symbol[:3]
            symbol_quote = symbol[3:6]
            
            # Currency overlap = potential hedge
            if (base_currency in [symbol_base, symbol_quote] or 
                quote_currency in [symbol_base, symbol_quote]):
                if symbol not in candidates:
                    candidates.append(symbol)
        
        return candidates
    
    def _check_existing_position(self, symbol: str, positions: List[Dict]) -> Optional[Dict]:
        """ตรวจสอบ position ที่มีอยู่"""
        for pos in positions:
            if pos['symbol'] == symbol:
                return pos
        return None
    
    def _determine_hedge_action(self, correlation: float, existing_hedge: Optional[Dict], 
                               primary_type: str) -> HedgeAction:
        """กำหนด hedge action ที่เหมาะสม"""
        abs_corr = abs(correlation)
        
        if existing_hedge:
            # มี position อยู่แล้ว
            existing_type = existing_hedge['type']
            
            if correlation < -0.70:  # Strong negative correlation
                if primary_type != existing_type:
                    return HedgeAction.NO_HEDGE  # Already naturally hedged
                else:
                    return HedgeAction.CLOSE_CONFLICTING  # Conflicting positions
            elif correlation > 0.70:  # Strong positive correlation
                if primary_type == existing_type:
                    return HedgeAction.REDUCE_EXPOSURE  # Too much same direction
                else:
                    return HedgeAction.FULL_HEDGE  # Good hedge
            else:
                return HedgeAction.PARTIAL_HEDGE
        else:
            # ไม่มี position
            if abs_corr >= 0.80:
                return HedgeAction.FULL_HEDGE
            elif abs_corr >= 0.60:
                return HedgeAction.PARTIAL_HEDGE
            else:
                return HedgeAction.NO_HEDGE
    
    def _calculate_confidence_score(self, correlation: float, 
                                  correlation_type: CorrelationType, 
                                  hedge_symbol: str) -> float:
        """คำนวณ confidence score"""
        try:
            base_confidence = min(abs(correlation), 0.95)  # Base from correlation strength
            
            # Type bonus
            type_bonus = {
                CorrelationType.POSITIVE: 0.05,
                CorrelationType.NEGATIVE: 0.10,  # Negative correlation = better hedge
                CorrelationType.MODERATE_POS: 0.03,
                CorrelationType.MODERATE_NEG: 0.08,
                CorrelationType.NEUTRAL: 0.0
            }.get(correlation_type, 0.0)
            
            # Liquidity bonus (major pairs)
            major_pairs = ['EURUSD.c', 'GBPUSD.c', 'USDJPY.c', 'USDCHF.c']
            liquidity_bonus = 0.05 if hedge_symbol in major_pairs else 0.0
            
            total_confidence = base_confidence + type_bonus + liquidity_bonus
            
            return min(1.0, total_confidence)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {str(e)}")
            return 0.5
    
    def _estimate_risk_reduction(self, correlation: float, hedge_ratio: float) -> float:
        """ประเมิน risk reduction ที่คาดหวัง"""
        try:
            # สูตรประมาณ: Risk Reduction = |correlation| × hedge_ratio × efficiency_factor
            efficiency_factor = 0.8  # ประสิทธิภาพของ hedge ในโลกจริง
            
            if correlation < 0:  # Negative correlation = better risk reduction
                risk_reduction = abs(correlation) * hedge_ratio * efficiency_factor * 1.2
            else:  # Positive correlation = partial risk reduction
                risk_reduction = abs(correlation) * hedge_ratio * efficiency_factor * 0.6
            
            return min(0.50, risk_reduction)  # สูงสุด 50% risk reduction
            
        except Exception as e:
            self.logger.error(f"Risk reduction calculation error: {str(e)}")
            return 0.1
    
    def _calculate_priority(self, confidence: float, risk_reduction: float) -> int:
        """คำนวณ execution priority"""
        try:
            # Priority = weighted score × 100
            priority_score = (confidence * 0.6) + (risk_reduction * 0.4)
            return int(priority_score * 100)
        except Exception as e:
            return 50
    
    def _build_hedge_reasoning(self, primary: str, hedge: str, correlation: float,
                              corr_type: CorrelationType, action: HedgeAction) -> List[str]:
        """สร้าง reasoning สำหรับ hedge decision"""
        reasons = []
        
        try:
            # Correlation reasoning
            if abs(correlation) >= 0.80:
                reasons.append(f"🔥 Strong correlation ({correlation:.3f}) with {hedge}")
            elif abs(correlation) >= 0.60:
                reasons.append(f"📊 Good correlation ({correlation:.3f}) with {hedge}")
            
            # Type reasoning
            if correlation < 0:
                reasons.append(f"✅ Negative correlation provides natural hedge")
            else:
                reasons.append(f"⚠️ Positive correlation requires opposite position")
            
            # Action reasoning
            action_explanations = {
                HedgeAction.FULL_HEDGE: "💯 Full hedge recommended for maximum protection",
                HedgeAction.PARTIAL_HEDGE: "🔄 Partial hedge to balance risk/reward",
                HedgeAction.REDUCE_EXPOSURE: "📉 Reduce exposure - too much same direction",
                HedgeAction.CLOSE_CONFLICTING: "❌ Close conflicting positions",
                HedgeAction.NO_HEDGE: "🚫 No hedge needed"
            }
            
            if action in action_explanations:
                reasons.append(action_explanations[action])
            
            # Currency analysis
            primary_base = primary[:3]
            primary_quote = primary[3:6]
            hedge_base = hedge[:3]
            hedge_quote = hedge[3:6]
            
            if primary_base == hedge_base or primary_quote == hedge_quote:
                reasons.append(f"💱 Currency overlap detected ({primary_base}/{primary_quote} vs {hedge_base}/{hedge_quote})")
            
            return reasons
            
        except Exception as e:
            self.logger.error(f"Reasoning building error: {str(e)}")
            return [f"Analysis for {primary} vs {hedge}"]
    
    def execute_hedge_strategy(self, opportunity: HedgeOpportunity) -> Dict:
        """
        🎯 ดำเนินการ hedge strategy
        """
        try:
            if opportunity.hedge_action == HedgeAction.NO_HEDGE:
                return {
                    'success': True,
                    'action': 'NO_ACTION',
                    'message': 'No hedge action required'
                }
            
            # ดึงข้อมูล position ปัจจุบัน
            primary_positions = mt5.positions_get(symbol=opportunity.primary_symbol)
            if not primary_positions:
                return {
                    'success': False,
                    'error': 'Primary position not found'
                }
            
            primary_pos = primary_positions[0]
            primary_volume = primary_pos.volume
            primary_type = primary_pos.type  # 0=BUY, 1=SELL
            
            # คำนวณ hedge volume
            hedge_volume = round(primary_volume * opportunity.hedge_ratio, 2)
            hedge_volume = max(0.01, min(2.0, hedge_volume))  # Limits
            
            # กำหนด hedge direction
            if opportunity.correlation_coefficient < 0:
                # Negative correlation: same direction
                hedge_type = primary_type
            else:
                # Positive correlation: opposite direction  
                hedge_type = 1 - primary_type
            
            # เตรียม order request
            tick = mt5.symbol_info_tick(opportunity.hedge_symbol)
            if not tick:
                return {
                    'success': False,
                    'error': f'No tick data for {opportunity.hedge_symbol}'
                }
            
            price = tick.ask if hedge_type == 0 else tick.bid
            
            hedge_request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': opportunity.hedge_symbol,
                'volume': hedge_volume,
                'type': mt5.ORDER_TYPE_BUY if hedge_type == 0 else mt5.ORDER_TYPE_SELL,
                'price': price,
                'deviation': 3,
                'magic': 54321,  # Hedge magic number
                'comment': f'HEDGE-{opportunity.primary_symbol}',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            
            # Execute hedge order
            result = mt5.order_send(hedge_request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': f'Hedge order failed: {result.retcode}',
                    'result': result
                }
            
            # บันทึก hedge information
            hedge_info = {
                'hedge_ticket': result.order,
                'primary_symbol': opportunity.primary_symbol,
                'hedge_symbol': opportunity.hedge_symbol,
                'hedge_volume': hedge_volume,
                'hedge_price': result.price,
                'correlation': opportunity.correlation_coefficient,
                'expected_risk_reduction': opportunity.expected_risk_reduction,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log successful hedge
            self.logger.info(f"🎯 HEDGE EXECUTED: {opportunity.primary_symbol} -> {opportunity.hedge_symbol}")
            self.logger.info(f"   Correlation: {opportunity.correlation_coefficient:.3f}")
            self.logger.info(f"   Hedge Ratio: {opportunity.hedge_ratio:.2f}")
            self.logger.info(f"   Risk Reduction: {opportunity.expected_risk_reduction:.1%}")
            
            return {
                'success': True,
                'action': 'HEDGE_EXECUTED',
                'hedge_info': hedge_info,
                'opportunity': opportunity
            }
            
        except Exception as e:
            self.logger.error(f"Hedge execution error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_portfolio_correlation_matrix(self, positions: List[Dict]) -> Dict:
        """
        🎯 สร้าง correlation matrix ของ portfolio
        """
        try:
            if len(positions) < 2:
                return {'matrix': {}, 'overall_risk': 'LOW'}
            
            symbols = [pos['symbol'] for pos in positions]
            correlation_matrix = {}
            
            # คำนวณ correlation ระหว่างทุกคู่
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    elif symbol2 in correlation_matrix and symbol1 in correlation_matrix[symbol2]:
                        # ใช้ค่าที่คำนวณแล้ว
                        correlation_matrix[symbol1][symbol2] = correlation_matrix[symbol2][symbol1]
                    else:
                        # คำนวณใหม่
                        corr = self.calculate_live_correlation(symbol1, symbol2)
                        correlation_matrix[symbol1][symbol2] = round(corr, 3)
            
            # วิเคราะห์ portfolio risk
            total_correlations = []
            for symbol1 in symbols:
                for symbol2 in symbols:
                    if symbol1 != symbol2:
                        total_correlations.append(abs(correlation_matrix[symbol1][symbol2]))
            
            avg_correlation = np.mean(total_correlations) if total_correlations else 0
            
            # กำหนด risk level
            if avg_correlation >= 0.70:
                overall_risk = 'HIGH'
                risk_comment = 'Strong correlations detected - high portfolio risk'
            elif avg_correlation >= 0.50:
                overall_risk = 'MEDIUM'
                risk_comment = 'Moderate correlations - balanced risk'
            else:
                overall_risk = 'LOW'
                risk_comment = 'Low correlations - diversified portfolio'
            
            return {
                'matrix': correlation_matrix,
                'overall_risk': overall_risk,
                'average_correlation': round(avg_correlation, 3),
                'risk_comment': risk_comment,
                'symbols_analyzed': symbols,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio correlation analysis error: {str(e)}")
            return {
                'matrix': {},
                'overall_risk': 'UNKNOWN',
                'error': str(e)
            }
    
    def get_hedge_recommendations(self) -> Dict:
        """Get hedge recommendations with symbol format fix"""
        try:
            # ดึง positions ปัจจุบัน
            positions = mt5.positions_get()
            if not positions:
                return {
                    'success': True,
                    'message': 'No positions to hedge',
                    'opportunities': [],
                    'portfolio_analysis': {
                        'matrix': {},
                        'overall_risk': 'LOW',
                        'symbols_analyzed': []
                    },
                    'debug_info': {
                        'positions_found': 0,
                        'mt5_connected': mt5.terminal_info() is not None,
                        'account_info': mt5.account_info() is not None if mt5.terminal_info() else False
                    }
                }
            
            # แปลง positions เป็น list of dict พร้อม normalize symbols
            position_list = []
            for pos in positions:
                try:
                    # Normalize symbol format
                    normalized_symbol = self.normalize_symbol(pos.symbol)
                    
                    position_list.append({
                        'symbol': normalized_symbol,
                        'original_symbol': pos.symbol,  # เก็บ original ไว้ด้วย
                        'type': 'BUY' if pos.type == 0 else 'SELL',
                        'volume': pos.volume,
                        'ticket': pos.ticket,
                        'profit': pos.profit
                    })
                    
                    self.logger.info(f"Position found: {pos.symbol} → {normalized_symbol}")
                    
                except Exception as pos_error:
                    self.logger.error(f"Error processing position {pos.symbol}: {str(pos_error)}")
                    continue
            
            if not position_list:
                return {
                    'success': True,
                    'message': 'No valid positions found after normalization',
                    'opportunities': [],
                    'portfolio_analysis': {
                        'matrix': {},
                        'overall_risk': 'LOW',
                        'symbols_analyzed': []
                    },
                    'debug_info': {
                        'raw_positions': len(positions),
                        'processed_positions': 0,
                        'normalization_failed': True
                    }
                }
            
            # วิเคราะห์โอกาส hedge
            opportunities = self.analyze_hedging_opportunities(position_list)
            
            # วิเคราะห์ portfolio correlation
            portfolio_analysis = self.get_portfolio_correlation_matrix(position_list)
            
            # สรุปคำแนะนำ
            recommendations = []
            if opportunities:
                for opp in opportunities[:3]:  # Top 3
                    rec = {
                        'primary_pair': opp.primary_symbol.replace('.c', ''),
                        'recommended_hedge': opp.hedge_symbol.replace('.c', ''),
                        'action': opp.hedge_action.value,
                        'correlation': round(opp.correlation_coefficient, 3),
                        'confidence': f"{opp.confidence_score:.1%}",
                        'risk_reduction': f"{opp.expected_risk_reduction:.1%}",
                        'priority': opp.execution_priority,
                        'reasoning': opp.reasoning[:2]  # Top 2 reasons
                    }
                    recommendations.append(rec)
            
            return {
                'success': True,
                'opportunities': recommendations,
                'portfolio_analysis': portfolio_analysis,
                'total_positions': len(position_list),
                'hedge_opportunities_found': len(opportunities),
                'analysis_timestamp': datetime.now().isoformat(),
                'system_status': 'ACTIVE',
                'debug_info': {
                    'raw_positions': len(positions),
                    'processed_positions': len(position_list),
                    'symbols_found': [pos['symbol'] for pos in position_list],
                    'original_symbols': [pos['original_symbol'] for pos in position_list]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Hedge recommendations error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'opportunities': [],
                'portfolio_analysis': {
                    'matrix': {},
                    'overall_risk': 'UNKNOWN',
                    'error': str(e)
                },
                'debug_info': {
                    'error_occurred': True,
                    'error_type': type(e).__name__,
                    'mt5_terminal_info': mt5.terminal_info() is not None
                }
            }
    
# 🎯 INTEGRATION HELPER
class HedgeSystemIntegrator:
    """Helper class สำหรับ integrate กับระบบหลัก"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.hedge_system = AdvancedCorrelationHedging(trading_system)
        
        print("🎯 Correlation Hedging System Integrated!")
        print("💱 Cross-Pair Risk Management Ready")
    
    def setup_hedge_routes(self, app):
        """เพิ่ม API routes สำหรับ hedging"""
        print("setup_hedge_routes")
        @app.route('/api/hedge/recommendations')
        def get_hedge_recommendations():
            """ดู hedge recommendations"""
            try:
                recommendations = self.hedge_system.get_hedge_recommendations()
                return jsonify(recommendations)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/hedge/correlation-matrix')
        def get_correlation_matrix():
            """ดู portfolio correlation matrix"""
            try:
                positions = mt5.positions_get()
                if not positions:
                    return jsonify({
                        'success': True,
                        'message': 'No positions for correlation analysis'
                    })
                
                position_list = [{'symbol': pos.symbol, 'volume': pos.volume} for pos in positions]
                matrix = self.hedge_system.get_portfolio_correlation_matrix(position_list)
                
                return jsonify({
                    'success': True,
                    'correlation_analysis': matrix
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/api/hedge/execute/<hedge_id>', methods=['POST'])
        def execute_hedge_strategy(hedge_id):
            """Execute hedge strategy"""
            try:
                # ในระบบจริงจะเก็บ opportunities และ execute ตาม ID
                recommendations = self.hedge_system.get_hedge_recommendations()
                
                if not recommendations['success']:
                    return jsonify(recommendations)
                
                opportunities = recommendations.get('opportunities', [])
                if not opportunities:
                    return jsonify({
                        'success': False,
                        'error': 'No hedge opportunities available'
                    })
                
                # Execute first opportunity (ในระบบจริงจะเลือกตาม hedge_id)
                # นี่เป็นตัวอย่าง - ในการใช้งานจริงควรมีการยืนยันจากผู้ใช้
                return jsonify({
                    'success': True,
                    'message': 'Hedge execution feature ready',
                    'note': 'Manual execution recommended for safety'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        print("🎯 Hedge API routes added successfully!")

# Export classes
__all__ = [
    'AdvancedCorrelationHedging',
    'HedgeSystemIntegrator', 
    'HedgeOpportunity',
    'HedgeAction',
    'CorrelationType'
]

print("🎯 Advanced Correlation Hedging System Ready!")
print("💱 Intelligent Cross-Pair Risk Management")
print("🛡️ Dynamic Hedge Ratio Calculation")
print("📊 Real-time Correlation Analysis")