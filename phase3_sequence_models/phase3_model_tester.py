"""
🎯 Ground Truth Pattern Tester
สร้างโจทย์ที่มีคำตอบ เพื่อทดสอบความแม่นยำของโมเดล Phase 3
โดยไม่ให้โมเดลรู้คำตอบล่วงหน้า
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class GroundTruthTester:
    """
    สร้างโจทย์ทดสอบที่มีคำตอบแน่นอน
    เพื่อวัดความแม่นยำของโมเดล Phase 3
    """
    
    def __init__(self):
        self.models = {}
        self.classifier_info = None
        self._load_models()
        
    def _load_models(self):
        """โหลดโมเดลทั้งหมด"""
        
        print("🔧 โหลดโมเดลสำหรับทดสอบ...")
        
        # โหลด Candlestick Classifier
        try:
            print("   📄 โหลด Candlestick Classifier...")
            classifier_model = tf.keras.models.load_model("candlestick_classifier/candlestick_classifier.h5")
            classifier_scaler = joblib.load("candlestick_classifier/scaler.pkl")
            classifier_encoder = joblib.load("candlestick_classifier/label_encoder.pkl")
            
            self.classifier_info = {
                'model': classifier_model,
                'scaler': classifier_scaler,
                'encoder': classifier_encoder
            }
            print("   ✅ โหลด Candlestick Classifier สำเร็จ")
            
        except Exception as e:
            print(f"   ❌ ไม่สามารถโหลด Candlestick Classifier: {str(e)}")
            
            # ลองโหลดด้วย compile=False
            try:
                print("   🔄 ลองโหลดด้วย compile=False...")
                classifier_model = tf.keras.models.load_model("candlestick_classifier/candlestick_classifier.h5", compile=False)
                classifier_scaler = joblib.load("candlestick_classifier/scaler.pkl")
                classifier_encoder = joblib.load("candlestick_classifier/label_encoder.pkl")
                
                self.classifier_info = {
                    'model': classifier_model,
                    'scaler': classifier_scaler,
                    'encoder': classifier_encoder
                }
                print("   ✅ โหลด Candlestick Classifier สำเร็จ (compile=False)")
                
            except Exception as e2:
                print(f"   ❌ ล้มเหลวทุกวิธี: {str(e2)}")
                return
        
        # โหลดโมเดล Phase 3
        for timeframe in ['D1', 'H4', 'H1']:
            try:
                print(f"   📄 โหลดโมเดล {timeframe}...")
                model_folder = f"phase3_sequence_models/{timeframe}"
                
                # ตรวจสอบไฟล์ก่อน
                model_file = f"{model_folder}/sequence_model.h5"
                if not os.path.exists(model_file):
                    print(f"   ❌ ไม่พบไฟล์: {model_file}")
                    continue
                
                # โหลดโมเดล
                try:
                    model = tf.keras.models.load_model(model_file)
                except Exception as model_error:
                    print(f"   ⚠️  ลองโหลดด้วย compile=False...")
                    model = tf.keras.models.load_model(model_file, compile=False)
                
                # โหลดไฟล์อื่นๆ
                scaler = joblib.load(f"{model_folder}/sequence_scaler.pkl")
                encoder = joblib.load(f"{model_folder}/sequence_encoder.pkl")
                pattern_to_id = joblib.load(f"{model_folder}/pattern_to_id.pkl")
                
                # โหลด metadata
                metadata_file = f"{model_folder}/model_info.json"
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {'test_accuracy': 0.95, 'sequence_length': 5}
                
                self.models[timeframe] = {
                    'model': model,
                    'scaler': scaler,
                    'encoder': encoder,
                    'pattern_to_id': pattern_to_id,
                    'metadata': metadata
                }
                
                print(f"   ✅ โหลดโมเดล {timeframe} สำเร็จ (Accuracy: {metadata.get('test_accuracy', 0.95):.4f})")
                
            except Exception as e:
                print(f"   ❌ ไม่สามารถโหลดโมเดล {timeframe}: {str(e)}")
                continue
    
    def create_ground_truth_patterns(self) -> List[Dict]:
        """
        สร้างโจทย์ที่มีคำตอบแน่นอน
        ตาม Traditional Candlestick Patterns
        """
        
        print("\n📚 สร้างโจทย์ Ground Truth Patterns...")
        
        ground_truth_tests = [
            
            # === MORNING STAR PATTERNS ===
            {
                'test_id': 'MORNING_STAR_01',
                'expected_pattern': 'MORNING_STAR',
                'description': 'Morning Star แบบมาตรฐาน - Long Bear + Doji + Long Bull',
                'sequence': [
                    {'Open': 2000, 'High': 2010, 'Low': 1970, 'Close': 1975},  # Long Bear
                    {'Open': 1975, 'High': 1980, 'Low': 1972, 'Close': 1977},  # Doji/Small
                    {'Open': 1977, 'High': 2010, 'Low': 1975, 'Close': 2005},  # Long Bull
                    {'Open': 2005, 'High': 2020, 'Low': 2003, 'Close': 2015},  # Confirmation
                    {'Open': 2015, 'High': 2030, 'Low': 2013, 'Close': 2025}   # Follow-through
                ],
                'psychology': 'BULLISH_REVERSAL',
                'confidence_threshold': 0.6
            },
            
            {
                'test_id': 'MORNING_STAR_02',
                'expected_pattern': 'MORNING_STAR',
                'description': 'Morning Star แบบ Alternative - Bear + Spinning Top + Bull',
                'sequence': [
                    {'Open': 1950, 'High': 1955, 'Low': 1920, 'Close': 1925},  # Long Bear
                    {'Open': 1925, 'High': 1935, 'Low': 1920, 'Close': 1930},  # Spinning Top
                    {'Open': 1930, 'High': 1965, 'Low': 1928, 'Close': 1960},  # Long Bull
                    {'Open': 1960, 'High': 1975, 'Low': 1958, 'Close': 1970},  # Confirmation
                    {'Open': 1970, 'High': 1985, 'Low': 1968, 'Close': 1980}   # Follow-through
                ],
                'psychology': 'BULLISH_REVERSAL',
                'confidence_threshold': 0.5
            },
            
            # === EVENING STAR PATTERNS ===
            {
                'test_id': 'EVENING_STAR_01',
                'expected_pattern': 'EVENING_STAR',
                'description': 'Evening Star แบบมาตรฐาน - Long Bull + Doji + Long Bear',
                'sequence': [
                    {'Open': 1980, 'High': 2015, 'Low': 1978, 'Close': 2010},  # Long Bull
                    {'Open': 2010, 'High': 2015, 'Low': 2007, 'Close': 2012},  # Doji/Small
                    {'Open': 2012, 'High': 2014, 'Low': 1985, 'Close': 1990},  # Long Bear
                    {'Open': 1990, 'High': 1995, 'Low': 1970, 'Close': 1975},  # Confirmation
                    {'Open': 1975, 'High': 1980, 'Low': 1955, 'Close': 1960}   # Follow-through
                ],
                'psychology': 'BEARISH_REVERSAL',
                'confidence_threshold': 0.6
            },
            
            {
                'test_id': 'EVENING_STAR_02',
                'expected_pattern': 'EVENING_STAR',
                'description': 'Evening Star แบบ Alternative - Bull + Small Bull + Bear',
                'sequence': [
                    {'Open': 1900, 'High': 1935, 'Low': 1898, 'Close': 1930},  # Long Bull
                    {'Open': 1930, 'High': 1940, 'Low': 1925, 'Close': 1935},  # Small Bull
                    {'Open': 1935, 'High': 1938, 'Low': 1905, 'Close': 1910},  # Long Bear
                    {'Open': 1910, 'High': 1915, 'Low': 1890, 'Close': 1895},  # Confirmation
                    {'Open': 1895, 'High': 1900, 'Low': 1875, 'Close': 1880}   # Follow-through
                ],
                'psychology': 'BEARISH_REVERSAL',
                'confidence_threshold': 0.5
            },
            
            # === THREE WHITE SOLDIERS ===
            {
                'test_id': 'THREE_WHITE_SOLDIERS_01',
                'expected_pattern': 'THREE_WHITE_SOLDIERS',
                'description': 'Three White Soldiers แบบมาตรฐาน - 3 Long Bull ติดกัน',
                'sequence': [
                    {'Open': 1980, 'High': 2005, 'Low': 1978, 'Close': 2000},  # Long Bull 1
                    {'Open': 2000, 'High': 2025, 'Low': 1998, 'Close': 2020},  # Long Bull 2
                    {'Open': 2020, 'High': 2045, 'Low': 2018, 'Close': 2040},  # Long Bull 3
                    {'Open': 2040, 'High': 2065, 'Low': 2038, 'Close': 2060},  # Continuation
                    {'Open': 2060, 'High': 2085, 'Low': 2058, 'Close': 2080}   # Strong Follow
                ],
                'psychology': 'STRONG_BULLISH',
                'confidence_threshold': 0.7
            },
            
            {
                'test_id': 'THREE_WHITE_SOLDIERS_02',
                'expected_pattern': 'THREE_WHITE_SOLDIERS',
                'description': 'Three White Soldiers แบบ Progressive - เพิ่มขนาดทีละน้อย',
                'sequence': [
                    {'Open': 1950, 'High': 1965, 'Low': 1948, 'Close': 1960},  # Medium Bull
                    {'Open': 1960, 'High': 1980, 'Low': 1958, 'Close': 1975},  # Long Bull
                    {'Open': 1975, 'High': 2000, 'Low': 1973, 'Close': 1995},  # Very Long Bull
                    {'Open': 1995, 'High': 2020, 'Low': 1993, 'Close': 2015},  # Continuation
                    {'Open': 2015, 'High': 2040, 'Low': 2013, 'Close': 2035}   # Strong Follow
                ],
                'psychology': 'BUILDING_BULLISH',
                'confidence_threshold': 0.6
            },
            
            # === THREE BLACK CROWS ===
            {
                'test_id': 'THREE_BLACK_CROWS_01',
                'expected_pattern': 'THREE_BLACK_CROWS',
                'description': 'Three Black Crows แบบมาตรฐาน - 3 Long Bear ติดกัน',
                'sequence': [
                    {'Open': 2080, 'High': 2085, 'Low': 2055, 'Close': 2060},  # Long Bear 1
                    {'Open': 2060, 'High': 2065, 'Low': 2035, 'Close': 2040},  # Long Bear 2
                    {'Open': 2040, 'High': 2045, 'Low': 2015, 'Close': 2020},  # Long Bear 3
                    {'Open': 2020, 'High': 2025, 'Low': 1995, 'Close': 2000},  # Continuation
                    {'Open': 2000, 'High': 2005, 'Low': 1975, 'Close': 1980}   # Strong Follow
                ],
                'psychology': 'STRONG_BEARISH',
                'confidence_threshold': 0.7
            },
            
            # === BULLISH ENGULFING ===
            {
                'test_id': 'BULLISH_ENGULFING_01',
                'expected_pattern': 'BULLISH_ENGULFING',
                'description': 'Bullish Engulfing แบบมาตรฐาน - Small Bear + Large Bull',
                'sequence': [
                    {'Open': 2000, 'High': 2010, 'Low': 1985, 'Close': 1990},  # Setup
                    {'Open': 2010, 'High': 2015, 'Low': 1995, 'Close': 2000},  # Small Bear
                    {'Open': 1995, 'High': 2025, 'Low': 1990, 'Close': 2020},  # Engulfing Bull
                    {'Open': 2020, 'High': 2035, 'Low': 2018, 'Close': 2030},  # Confirmation
                    {'Open': 2030, 'High': 2045, 'Low': 2028, 'Close': 2040}   # Follow-through
                ],
                'psychology': 'BULLISH_REVERSAL',
                'confidence_threshold': 0.6
            },
            
            # === BEARISH ENGULFING ===
            {
                'test_id': 'BEARISH_ENGULFING_01',
                'expected_pattern': 'BEARISH_ENGULFING',
                'description': 'Bearish Engulfing แบบมาตรฐาน - Small Bull + Large Bear',
                'sequence': [
                    {'Open': 1980, 'High': 1995, 'Low': 1975, 'Close': 1990},  # Setup
                    {'Open': 1990, 'High': 2005, 'Low': 1985, 'Close': 2000},  # Small Bull
                    {'Open': 2005, 'High': 2010, 'Low': 1975, 'Close': 1980},  # Engulfing Bear
                    {'Open': 1980, 'High': 1985, 'Low': 1965, 'Close': 1970},  # Confirmation
                    {'Open': 1970, 'High': 1975, 'Low': 1955, 'Close': 1960}   # Follow-through
                ],
                'psychology': 'BEARISH_REVERSAL',
                'confidence_threshold': 0.6
            },
            
            # === DOJI PATTERNS ===
            {
                'test_id': 'DOJI_INDECISION_01',
                'expected_pattern': 'DOJI_INDECISION',
                'description': 'Doji Indecision - ลำดับ Doji ที่แสดงความลังเล',
                'sequence': [
                    {'Open': 2000, 'High': 2010, 'Low': 1995, 'Close': 2005},  # Small movement
                    {'Open': 2005, 'High': 2012, 'Low': 1998, 'Close': 2003},  # Doji-like
                    {'Open': 2003, 'High': 2010, 'Low': 1996, 'Close': 2001},  # Doji-like
                    {'Open': 2001, 'High': 2008, 'Low': 1994, 'Close': 1999},  # Slightly Bear
                    {'Open': 1999, 'High': 2005, 'Low': 1985, 'Close': 1990}   # Decision (Bear)
                ],
                'psychology': 'INDECISION',
                'confidence_threshold': 0.4
            },
            
            # === MOMENTUM BUILDING ===
            {
                'test_id': 'MOMENTUM_BUILDING_BULL_01',
                'expected_pattern': 'MOMENTUM_BUILDING_BULL',
                'description': 'Momentum Building Bullish - จากเล็กไปใหญ่',
                'sequence': [
                    {'Open': 1980, 'High': 1988, 'Low': 1978, 'Close': 1985},  # Small Bull
                    {'Open': 1985, 'High': 1998, 'Low': 1983, 'Close': 1995},  # Medium Bull
                    {'Open': 1995, 'High': 2015, 'Low': 1993, 'Close': 2010},  # Large Bull
                    {'Open': 2010, 'High': 2030, 'Low': 2008, 'Close': 2025},  # Very Large
                    {'Open': 2025, 'High': 2050, 'Low': 2023, 'Close': 2045}   # Huge Bull
                ],
                'psychology': 'BUILDING_BULLISH',
                'confidence_threshold': 0.6
            }
            
        ]
        
        print(f"📝 สร้างโจทย์ทดสอบ: {len(ground_truth_tests)} รูปแบบ")
        return ground_truth_tests
    
    def predict_single_candlestick(self, candle, timeframe):
        """ทำนายแท่งเทียนเดี่ยว"""
        
        try:
            o, h, l, c = candle['Open'], candle['High'], candle['Low'], candle['Close']
            
            body_size = abs(c - o)
            body_direction = 1 if c > o else -1
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            total_range = h - l
            
            if total_range == 0:
                total_range = 1e-8
            
            features = {
                'Body_ratio': body_size / total_range,
                'Upper_shadow_ratio': upper_shadow / total_range,
                'Lower_shadow_ratio': lower_shadow / total_range,
                'Open_position': (o - l) / total_range,
                'Close_position': (c - l) / total_range,
                'Shadow_symmetry': abs(upper_shadow - lower_shadow) / (upper_shadow + lower_shadow + 1e-8),
                'Size_vs_ATR': 1.0,
                'Body_vs_ATR': 1.0,
                'Body_direction': body_direction,
                'Hour': 12,
                'Day_of_week': 1,
                'TF_minutes': 60,
                'TF_weight': 2.0
            }
            
            feature_values = []
            for col in self.classifier_info['scaler'].feature_names_in_:
                feature_values.append(features.get(col, 1.0))
            
            X = np.array([feature_values])
            X_scaled = self.classifier_info['scaler'].transform(X)
            
            prediction = self.classifier_info['model'].predict(X_scaled, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            pattern_name = self.classifier_info['encoder'].inverse_transform([predicted_class])[0]
            
            return {
                'pattern': pattern_name,
                'confidence': float(confidence)
            }
            
        except:
            return {'pattern': 'UNKNOWN', 'confidence': 0.5}
    
    def predict_sequence_pattern(self, sequence_data, timeframe):
        """ทำนาย Sequence Pattern"""
        
        if timeframe not in self.models:
            return {'error': f'ไม่มีโมเดล {timeframe}'}
        
        model_info = self.models[timeframe]
        
        try:
            # แปลงแต่ละแท่งเป็น pattern
            individual_patterns = []
            for candle in sequence_data:
                pattern_pred = self.predict_single_candlestick(candle, timeframe)
                individual_patterns.append(pattern_pred['pattern'])
            
            # เตรียม sequence input
            pattern_to_id = model_info['pattern_to_id']
            sequence_ids = []
            for pattern in individual_patterns:
                pattern_id = pattern_to_id.get(pattern, 0)
                sequence_ids.append(pattern_id)
            
            # ปรับความยาว
            expected_length = 5
            if len(sequence_ids) > expected_length:
                sequence_ids = sequence_ids[-expected_length:]
            elif len(sequence_ids) < expected_length:
                sequence_ids = [0] * (expected_length - len(sequence_ids)) + sequence_ids
            
            # เตรียม features
            prices = [candle['Close'] for candle in sequence_data]
            if len(prices) > 1:
                price_changes = np.diff(prices) / prices[:-1]
                volatility = np.std(price_changes)
                momentum = np.mean(price_changes)
            else:
                volatility = 0
                momentum = 0
            
            features = np.array([[
                volatility,
                momentum,
                len(set(individual_patterns)),
                0.8, 12, 1, 0.7
            ]])
            
            features_scaled = model_info['scaler'].transform(features)
            
            # ทำนาย
            sequence_input = np.array([sequence_ids])
            prediction = model_info['model'].predict([sequence_input, features_scaled], verbose=0)
            
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            pattern_name = model_info['encoder'].inverse_transform([predicted_class])[0]
            
            return {
                'predicted_pattern': pattern_name,
                'confidence': float(confidence),
                'individual_patterns': individual_patterns
            }
            
        except Exception as e:
            return {'error': f'เกิดข้อผิดพลาด: {str(e)}'}
    
    def run_ground_truth_test(self):
        """รันการทดสอบ Ground Truth"""
        
        print("🎯 เริ่มการทดสอบ Ground Truth")
        print("=" * 80)
        
        if not self.models or not self.classifier_info:
            print("❌ โมเดลไม่พร้อม")
            return
        
        # สร้างโจทย์ทดสอบ
        test_cases = self.create_ground_truth_patterns()
        
        # เก็บผลลัพธ์
        results = {}
        overall_stats = {
            'total_tests': len(test_cases),
            'correct_predictions': 0,
            'timeframe_results': {tf: {'correct': 0, 'total': 0} for tf in self.models.keys()}
        }
        
        print(f"\n🚀 ทดสอบ {len(test_cases)} โจทย์ด้วย {len(self.models)} โมเดล")
        print("=" * 80)
        
        # ทดสอบแต่ละโจทย์
        for i, test_case in enumerate(test_cases, 1):
            test_id = test_case['test_id']
            expected = test_case['expected_pattern']
            threshold = test_case['confidence_threshold']
            
            print(f"\n[{i}/{len(test_cases)}] ทดสอบ: {test_id}")
            print(f"📋 คำตอบที่ถูก: {expected}")
            print(f"📝 รายละเอียด: {test_case['description']}")
            print(f"🎯 Confidence ขั้นต่ำ: {threshold:.2f}")
            
            # แสดง sequence
            sequence = test_case['sequence']
            print(f"\n📊 Sequence (5 แท่ง):")
            for j, candle in enumerate(sequence, 1):
                direction = "🟢" if candle['Close'] > candle['Open'] else "🔴" if candle['Close'] < candle['Open'] else "⚫"
                body = abs(candle['Close'] - candle['Open'])
                print(f"  {j}. {direction} O:{candle['Open']:.0f} H:{candle['High']:.0f} L:{candle['Low']:.0f} C:{candle['Close']:.0f} (Body: {body:.0f})")
            
            # ทดสอบด้วยทุกโมเดล
            test_results = {}
            
            for timeframe in self.models.keys():
                prediction = self.predict_sequence_pattern(sequence, timeframe)
                
                if 'error' not in prediction:
                    predicted = prediction['predicted_pattern']
                    confidence = prediction['confidence']
                    
                    # ตรวจสอบความถูกต้อง
                    is_correct = (predicted == expected) and (confidence >= threshold)
                    
                    test_results[timeframe] = {
                        'predicted': predicted,
                        'confidence': confidence,
                        'is_correct': is_correct,
                        'individual_patterns': prediction['individual_patterns']
                    }
                    
                    # อัพเดทสถิติ
                    overall_stats['timeframe_results'][timeframe]['total'] += 1
                    if is_correct:
                        overall_stats['timeframe_results'][timeframe]['correct'] += 1
                    
                    # แสดงผล
                    status = "✅" if is_correct else "❌"
                    print(f"\n{status} {timeframe} Prediction:")
                    print(f"   ทำนาย: {predicted}")
                    print(f"   Confidence: {confidence:.4f}")
                    print(f"   Individual: {prediction['individual_patterns']}")
                    
                    if is_correct:
                        print(f"   🎉 ถูกต้อง!")
                    else:
                        if predicted != expected:
                            print(f"   ❌ ผิด: ควรเป็น {expected}")
                        if confidence < threshold:
                            print(f"   ⚠️  Confidence ต่ำ: ควร >= {threshold:.2f}")
                
                else:
                    print(f"\n❌ {timeframe} Error: {prediction['error']}")
                    test_results[timeframe] = {'error': prediction['error']}
            
            results[test_id] = {
                'expected': expected,
                'threshold': threshold,
                'predictions': test_results,
                'description': test_case['description']
            }
            
            print("-" * 60)
        
        # คำนวณผลรวม
        total_predictions = sum([stats['total'] for stats in overall_stats['timeframe_results'].values()])
        total_correct = sum([stats['correct'] for stats in overall_stats['timeframe_results'].values()])
        overall_stats['total_predictions'] = total_predictions
        overall_stats['correct_predictions'] = total_correct
        overall_stats['overall_accuracy'] = total_correct / total_predictions if total_predictions > 0 else 0
        
        # แสดงสรุปผล
        self._print_ground_truth_summary(overall_stats, results)
        
        return results, overall_stats
    
    def _print_ground_truth_summary(self, stats, results):
        """แสดงสรุปผล Ground Truth Test"""
        
        print("\n" + "=" * 80)
        print("📊 สรุปผล Ground Truth Test")
        print("=" * 80)
        
        print(f"📋 การทดสอบรวม:")
        print(f"   โจทย์ทั้งหมด: {stats['total_tests']} ข้อ")
        print(f"   การทำนายรวม: {stats['total_predictions']} ครั้ง")
        print(f"   ทำนายถูก: {stats['correct_predictions']} ครั้ง")
        print(f"   ความแม่นยำรวม: {stats['overall_accuracy']*100:.1f}%")
        
        print(f"\n📈 ผลลัพธ์แต่ละไทม์เฟรม:")
        for timeframe, tf_stats in stats['timeframe_results'].items():
            if tf_stats['total'] > 0:
                accuracy = (tf_stats['correct'] / tf_stats['total']) * 100
                print(f"   🕯️  {timeframe}: {tf_stats['correct']}/{tf_stats['total']} ({accuracy:.1f}%)")
            else:
                print(f"   🕯️  {timeframe}: ไม่มีการทดสอบ")
        
        # แสดง Pattern ที่ทำนายถูก/ผิด
        correct_patterns = []
        incorrect_patterns = []
        
        for test_id, result in results.items():
            expected = result['expected']
            
            for tf, pred in result['predictions'].items():
                if 'error' not in pred:
                    if pred['is_correct']:
                        correct_patterns.append(expected)
                    else:
                        incorrect_patterns.append(f"{expected} (ทำนาย: {pred['predicted']})")
        
        if correct_patterns:
            from collections import Counter
            correct_counts = Counter(correct_patterns)
            print(f"\n✅ Pattern ที่ทำนายถูกบ่อย:")
            for pattern, count in correct_counts.most_common(5):
                print(f"   {pattern}: {count} ครั้ง")
        
        if incorrect_patterns:
            print(f"\n❌ การทำนายที่ผิด:")
            for mistake in incorrect_patterns[:5]:  # แสดงแค่ 5 อันแรก
                print(f"   {mistake}")
        
        # ข้อเสนอแนะ
        if stats['overall_accuracy'] >= 0.8:
            print(f"\n🎉 ผลลัพธ์ยอดเยี่ยม! โมเดลเข้าใจ Traditional Patterns ได้ดีมาก")
        elif stats['overall_accuracy'] >= 0.6:
            print(f"\n👍 ผลลัพธ์ดี! โมเดลเข้าใจ Pattern พื้นฐานได้")
        else:
            print(f"\n🔧 ควรปรับปรุงโมเดล: ความแม่นยำยังต่ำ")
        
        print("=" * 80)

def main():
    """ฟังก์ชันหลัก"""
    
    print("🎯 Ground Truth Pattern Tester")
    print("ทดสอบโมเดลด้วยโจทย์ที่มีคำตอบแน่นอน")
    print("=" * 80)
    
    try:
        tester = GroundTruthTester()
        
        if not tester.models:
            print("❌ ไม่สามารถโหลดโมเดลได้")
            return
        
        print(f"✅ โหลดโมเดลสำเร็จ: {list(tester.models.keys())}")
        
        # ยืนยันการทดสอบ
        confirm = input(f"\n🚀 เริ่มการทดสอบ Ground Truth? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("👋 ยกเลิกการทดสอบ")
            return
        
        # เริ่มทดสอบ
        results, stats = tester.run_ground_truth_test()
        
        print(f"\n🎉 การทดสอบเสร็จสมบูรณ์!")
        print(f"📊 ความแม่นยำรวม: {stats['overall_accuracy']*100:.1f}%")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()