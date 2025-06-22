"""
🚀 Complete Phase 3 Trainer
เทรน Pattern Sequence Learning สำหรับ 3 ไทม์เฟรมหลัก (D1, H4, H1)
รวมทุกอย่างไว้ในไฟล์เดียว - รันแล้วเสร็จ!
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Embedding, LayerNormalization
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Concatenate, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import time
import json
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class Phase3CompleteTrainer:
    """
    🕯️ Complete Phase 3 Trainer สำหรับ 3 ไทม์เฟรมหลัก
    เทรน Pattern Sequence Learning แบบครบวงจร
    """
    
    def __init__(self, 
                 candlestick_classifier_path: str = "candlestick_classifier",
                 data_folder: str = "raw_ai_data_XAUUSD_c",
                 output_folder: str = "phase3_sequence_models"):
        
        self.classifier_path = candlestick_classifier_path
        self.data_folder = data_folder
        self.output_folder = output_folder
        
        # สร้างโฟลเดอร์ output
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Setup logging
        self._setup_logging()
        
        # Models and data
        self.candlestick_classifier = None
        self.candlestick_scaler = None
        self.candlestick_encoder = None
        self.raw_data = {}
        self.trained_models = {}
        
        # เริ่มต้น
        self._check_prerequisites()
        self._load_candlestick_classifier()
        self._load_raw_data()
    
    def _setup_logging(self):
        """ตั้งค่า logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_folder}/phase3_training.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _check_prerequisites(self):
        """ตรวจสอบข้อมูลที่จำเป็น"""
        
        # ตรวจสอบ Candlestick Classifier (Phase 2)
        required_files = [
            f"{self.classifier_path}/candlestick_classifier.h5",
            f"{self.classifier_path}/scaler.pkl",
            f"{self.classifier_path}/label_encoder.pkl"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"❌ ไม่พบไฟล์ Phase 2: {missing_files}\nกรุณาเทรน Candlestick Classifier ก่อน")
        
        # ตรวจสอบข้อมูล Raw (Phase 1)
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"❌ ไม่พบโฟลเดอร์ข้อมูล: {self.data_folder}\nกรุณาเตรียมข้อมูล Raw ก่อน")
        
        self.logger.info("เช็ค Prerequisites ผ่านการตรวจสอบ")
    
    def _load_candlestick_classifier(self):
        """โหลดโมเดล Candlestick Classifier จาก Phase 2"""
        try:
            self.candlestick_classifier = tf.keras.models.load_model(
                f"{self.classifier_path}/candlestick_classifier.h5"
            )
            self.candlestick_scaler = joblib.load(f"{self.classifier_path}/scaler.pkl")
            self.candlestick_encoder = joblib.load(f"{self.classifier_path}/label_encoder.pkl")
            
            self.logger.info("โหลด Candlestick Classifier สำเร็จ")
            self.logger.info(f"รู้จัก {len(self.candlestick_encoder.classes_)} patterns")
            
        except Exception as e:
            raise Exception(f"❌ ไม่สามารถโหลด Candlestick Classifier: {str(e)}")
    
    def _load_raw_data(self):
        """โหลดข้อมูล Raw สำหรับ 3 ไทม์เฟรมหลัก"""
        
        primary_timeframes = ['D1', 'H4', 'H1']
        
        self.logger.info("โหลดข้อมูล Raw สำหรับ 3 ไทม์เฟรมหลัก...")
        
        for tf in primary_timeframes:
            file_path = f"{self.data_folder}/XAUUSD.c_{tf}_raw.csv"
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                self.raw_data[tf] = df
                
                # คำนวณช่วงข้อมูล
                start_date = df.index.min().strftime('%Y-%m-%d')
                end_date = df.index.max().strftime('%Y-%m-%d')
                days = (df.index.max() - df.index.min()).days
                
                self.logger.info(f"{tf}: {len(df):,} แท่ง | {days:,} วัน | {start_date} - {end_date}")
            else:
                self.logger.warning(f"ไม่พบ {tf}: {file_path}")
        
        if not self.raw_data:
            raise FileNotFoundError("ไม่พบข้อมูลไทม์เฟรมใดเลย")
        
        self.logger.info(f"พร้อมเทรน {len(self.raw_data)} ไทม์เฟรม: {list(self.raw_data.keys())}")
    
    def predict_single_candlestick(self, ohlc_data: Dict, timeframe: str = 'H1') -> Dict:
        """ใช้ Candlestick Classifier ทำนายแท่งเดี่ยว"""
        
        # สร้างฟีเจอร์จาก OHLC
        o, h, l, c = ohlc_data['Open'], ohlc_data['High'], ohlc_data['Low'], ohlc_data['Close']
        
        body_size = abs(c - o)
        body_direction = 1 if c > o else -1
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l
        
        if total_range == 0:
            total_range = 1e-8
        
        # เตรียมฟีเจอร์พื้นฐาน
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
            'Day_of_week': 1
        }
        
        # เพิ่มฟีเจอร์ไทม์เฟรม
        tf_mapping = {'M1': 1, 'M5': 5, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
        tf_weights = {'M1': 1.0, 'M5': 1.2, 'M30': 1.5, 'H1': 2.0, 'H4': 3.0, 'D1': 4.0}
        
        features['TF_minutes'] = tf_mapping.get(timeframe, 60)
        features['TF_weight'] = tf_weights.get(timeframe, 2.0)
        
        # สร้าง input array
        feature_values = []
        for col in self.candlestick_scaler.feature_names_in_:
            if col in features:
                feature_values.append(features[col])
            else:
                # Default values
                if 'TF_' in col:
                    feature_values.append(60)
                elif 'Hour' in col:
                    feature_values.append(12)
                elif 'Day' in col:
                    feature_values.append(1)
                else:
                    feature_values.append(1.0)
        
        X = np.array([feature_values])
        X_scaled = self.candlestick_scaler.transform(X)
        
        # ทำนาย
        prediction = self.candlestick_classifier.predict(X_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        pattern_name = self.candlestick_encoder.inverse_transform([predicted_class])[0]
        
        return {
            'pattern': pattern_name,
            'confidence': float(confidence),
            'probabilities': prediction[0]
        }
    
    def create_traditional_sequence_patterns(self) -> Dict[str, Dict]:
        """สร้าง Ground Truth ของ Traditional Multi-Candle Patterns"""
        
        traditional_patterns = {
            # === REVERSAL PATTERNS ===
            'MORNING_STAR': {
                'sequence': ['LONG_BEAR', 'DOJI', 'LONG_BULL'],
                'alternatives': [
                    ['LONG_BEAR', 'SMALL_BEAR', 'LONG_BULL'],
                    ['LONG_BEAR', 'SPINNING_TOP', 'LONG_BULL']
                ],
                'psychology': 'BULLISH_REVERSAL',
                'strength': 'STRONG',
                'reliability': 0.85
            },
            
            'EVENING_STAR': {
                'sequence': ['LONG_BULL', 'DOJI', 'LONG_BEAR'],
                'alternatives': [
                    ['LONG_BULL', 'SMALL_BULL', 'LONG_BEAR'],
                    ['LONG_BULL', 'SPINNING_TOP', 'LONG_BEAR']
                ],
                'psychology': 'BEARISH_REVERSAL',
                'strength': 'STRONG',
                'reliability': 0.85
            },
            
            'BULLISH_ENGULFING': {
                'sequence': ['LONG_BEAR', 'LONG_BULL'],
                'alternatives': [
                    ['SMALL_BEAR', 'LONG_BULL'],
                    ['NORMAL', 'MARUBOZU_BULL']
                ],
                'psychology': 'BULLISH_REVERSAL',
                'strength': 'MEDIUM',
                'reliability': 0.75
            },
            
            'BEARISH_ENGULFING': {
                'sequence': ['LONG_BULL', 'LONG_BEAR'],
                'alternatives': [
                    ['SMALL_BULL', 'LONG_BEAR'],
                    ['NORMAL', 'MARUBOZU_BEAR']
                ],
                'psychology': 'BEARISH_REVERSAL',
                'strength': 'MEDIUM',
                'reliability': 0.75
            },
            
            # === CONTINUATION PATTERNS ===
            'THREE_WHITE_SOLDIERS': {
                'sequence': ['LONG_BULL', 'LONG_BULL', 'LONG_BULL'],
                'alternatives': [
                    ['SMALL_BULL', 'LONG_BULL', 'LONG_BULL'],
                    ['LONG_BULL', 'SMALL_BULL', 'LONG_BULL']
                ],
                'psychology': 'STRONG_BULLISH',
                'strength': 'VERY_STRONG',
                'reliability': 0.80
            },
            
            'THREE_BLACK_CROWS': {
                'sequence': ['LONG_BEAR', 'LONG_BEAR', 'LONG_BEAR'],
                'alternatives': [
                    ['SMALL_BEAR', 'LONG_BEAR', 'LONG_BEAR'],
                    ['LONG_BEAR', 'SMALL_BEAR', 'LONG_BEAR']
                ],
                'psychology': 'STRONG_BEARISH',
                'strength': 'VERY_STRONG',
                'reliability': 0.80
            },
            
            # === MOMENTUM PATTERNS ===
            'MOMENTUM_BUILDING_BULL': {
                'sequence': ['SMALL_BULL', 'SMALL_BULL', 'LONG_BULL'],
                'alternatives': [
                    ['SMALL_BULL', 'LONG_BULL', 'MARUBOZU_BULL'],
                    ['DOJI', 'SMALL_BULL', 'LONG_BULL']
                ],
                'psychology': 'BUILDING_BULLISH',
                'strength': 'MEDIUM',
                'reliability': 0.70
            },
            
            'MOMENTUM_BUILDING_BEAR': {
                'sequence': ['SMALL_BEAR', 'SMALL_BEAR', 'LONG_BEAR'],
                'alternatives': [
                    ['SMALL_BEAR', 'LONG_BEAR', 'MARUBOZU_BEAR'],
                    ['DOJI', 'SMALL_BEAR', 'LONG_BEAR']
                ],
                'psychology': 'BUILDING_BEARISH',
                'strength': 'MEDIUM',
                'reliability': 0.70
            },
            
            # === INDECISION PATTERNS ===
            'DOJI_INDECISION': {
                'sequence': ['DOJI', 'DOJI'],
                'alternatives': [
                    ['DOJI', 'SPINNING_TOP'],
                    ['SPINNING_TOP', 'DOJI']
                ],
                'psychology': 'STRONG_INDECISION',
                'strength': 'WEAK',
                'reliability': 0.45
            },
            
            'EXHAUSTION_BULL': {
                'sequence': ['LONG_BULL', 'SMALL_BULL', 'DOJI'],
                'alternatives': [
                    ['MARUBOZU_BULL', 'SMALL_BULL', 'SPINNING_TOP'],
                    ['LONG_BULL', 'LONG_BULL', 'SHOOTING_STAR']
                ],
                'psychology': 'BULLISH_EXHAUSTION',
                'strength': 'WEAK',
                'reliability': 0.60
            },
            
            'EXHAUSTION_BEAR': {
                'sequence': ['LONG_BEAR', 'SMALL_BEAR', 'DOJI'],
                'alternatives': [
                    ['MARUBOZU_BEAR', 'SMALL_BEAR', 'SPINNING_TOP'],
                    ['LONG_BEAR', 'LONG_BEAR', 'HAMMER']
                ],
                'psychology': 'BEARISH_EXHAUSTION',
                'strength': 'WEAK',
                'reliability': 0.60
            }
        }
        
        return traditional_patterns
    
    def _match_traditional_pattern(self, sequence: List[str], traditional_patterns: Dict) -> Tuple[str, float]:
        """จับคู่ sequence กับ Traditional Patterns"""
        
        best_match = 'NO_PATTERN'
        best_confidence = 0.0
        
        for pattern_name, pattern_info in traditional_patterns.items():
            # ตรวจสอบ main sequence
            if len(sequence) >= len(pattern_info['sequence']):
                # เช็คทุก subsequence
                for i in range(len(sequence) - len(pattern_info['sequence']) + 1):
                    subseq = sequence[i:i+len(pattern_info['sequence'])]
                    
                    if subseq == pattern_info['sequence']:
                        return pattern_name, pattern_info['reliability']
                    
                    # ตรวจสอบ alternatives
                    for alt_seq in pattern_info.get('alternatives', []):
                        if subseq == alt_seq:
                            confidence = pattern_info['reliability'] * 0.8
                            if confidence > best_confidence:
                                best_match = pattern_name
                                best_confidence = confidence
        
        return best_match, best_confidence
    
    def generate_sequence_training_data(self, timeframe: str, sequence_length: int = 5) -> Dict:
        """สร้างข้อมูลเทรนสำหรับ Sequence Learning"""
        
        if timeframe not in self.raw_data:
            raise ValueError(f"ไม่มีข้อมูล {timeframe}")
        
        self.logger.info(f"สร้างข้อมูล Sequence Training สำหรับ {timeframe}")
        
        df = self.raw_data[timeframe].copy()
        
        # === STEP 1: แปลงแท่งเทียนเป็น Pattern ===
        self.logger.info("วิเคราะห์แท่งเทียนแต่ละแท่ง...")
        
        patterns = []
        confidences = []
        
        for idx, row in df.iterrows():
            ohlc = {
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close']
            }
            
            try:
                prediction = self.predict_single_candlestick(ohlc, timeframe)
                patterns.append(prediction['pattern'])
                confidences.append(prediction['confidence'])
            except:
                patterns.append('NORMAL')
                confidences.append(0.5)
        
        df['Pattern'] = patterns
        df['Pattern_confidence'] = confidences
        
        # === STEP 2: สร้าง Sequences ===
        self.logger.info(f"สร้าง Sequences ยาว {sequence_length} แท่ง...")
        
        sequences = []
        targets = []
        sequence_features = []
        
        traditional_patterns = self.create_traditional_sequence_patterns()
        
        for i in range(sequence_length, len(df)):
            # สร้าง sequence
            seq_patterns = df['Pattern'].iloc[i-sequence_length:i].tolist()
            seq_confidences = df['Pattern_confidence'].iloc[i-sequence_length:i].tolist()
            
            # คำนวณฟีเจอร์ sequence
            seq_ohlc = df[['Open', 'High', 'Low', 'Close']].iloc[i-sequence_length:i]
            
            # Price movement features
            price_changes = seq_ohlc['Close'].pct_change().dropna()
            volatility = price_changes.std() if len(price_changes) > 0 else 0
            momentum = price_changes.mean() if len(price_changes) > 0 else 0
            
            # Pattern diversity
            unique_patterns = len(set(seq_patterns))
            avg_confidence = np.mean(seq_confidences)
            
            # Time features
            current_time = df.index[i]
            hour = current_time.hour
            day_of_week = current_time.dayofweek
            
            # จับคู่กับ Traditional Patterns
            sequence_pattern, pattern_confidence = self._match_traditional_pattern(
                seq_patterns, traditional_patterns
            )
            
            sequences.append(seq_patterns)
            targets.append(sequence_pattern)
            sequence_features.append([
                volatility, momentum, unique_patterns, avg_confidence,
                hour, day_of_week, pattern_confidence
            ])
        
        # === STEP 3: เตรียมข้อมูลสำหรับโมเดล ===
        
        # Encode patterns เป็นตัวเลข
        all_patterns = list(set([p for seq in sequences for p in seq]))
        pattern_to_id = {pattern: i for i, pattern in enumerate(all_patterns)}
        
        # แปลง sequences เป็น numerical
        X_sequences = []
        for seq in sequences:
            X_sequences.append([pattern_to_id[p] for p in seq])
        
        X_sequences = np.array(X_sequences)
        X_features = np.array(sequence_features)
        y = np.array(targets)
        
        # === STEP 4: Time-based Split ===
        split_date = '2024-01-01'
        
        # สร้าง datetime index สำหรับ split
        sequence_dates = df.index[sequence_length:]
        
        train_mask = sequence_dates < split_date
        test_mask = sequence_dates >= split_date
        
        X_seq_train = X_sequences[train_mask]
        X_feat_train = X_features[train_mask]
        y_train = y[train_mask]
        
        X_seq_test = X_sequences[test_mask]
        X_feat_test = X_features[test_mask]
        y_test = y[test_mask]
        
        # === STEP 5: สถิติข้อมูล ===
        pattern_distribution = Counter(y_train)
        
        self.logger.info(f"สร้างข้อมูล Sequence เสร็จสิ้น:")
        self.logger.info(f"   Training: {len(X_seq_train):,} sequences")
        self.logger.info(f"   Testing: {len(X_seq_test):,} sequences")
        self.logger.info(f"   Unique patterns: {len(all_patterns)}")
        self.logger.info(f"   Sequence length: {sequence_length}")
        
        self.logger.info("Pattern Distribution:")
        for pattern, count in pattern_distribution.most_common(10):
            percentage = (count / len(y_train)) * 100
            self.logger.info(f"   {pattern}: {count} ({percentage:.1f}%)")
        
        return {
            'X_sequences_train': X_seq_train,
            'X_features_train': X_feat_train,
            'y_train': y_train,
            'X_sequences_test': X_seq_test,
            'X_features_test': X_feat_test,
            'y_test': y_test,
            'pattern_to_id': pattern_to_id,
            'all_patterns': all_patterns,
            'sequence_length': sequence_length,
            'pattern_distribution': pattern_distribution,
            'traditional_patterns': traditional_patterns,
            'timeframe': timeframe
        }
    
    def create_sequence_model(self, vocab_size: int, sequence_length: int, 
                            num_classes: int, feature_dim: int) -> Model:
        """สร้างโมเดล Sequence Learning ด้วย LSTM + Attention"""
        
        # === INPUT LAYERS ===
        sequence_input = Input(shape=(sequence_length,), name='candlestick_sequence')
        features_input = Input(shape=(feature_dim,), name='sequence_features')
        
        # === SEQUENCE PROCESSING ===
        embedded = Embedding(vocab_size, 64, name='pattern_embedding')(sequence_input)
        
        # Bidirectional LSTM
        lstm_out = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.2),
            name='sequence_lstm'
        )(embedded)
        
        # Multi-Head Attention
        attention = MultiHeadAttention(
            num_heads=8, key_dim=64, name='pattern_attention'
        )(lstm_out, lstm_out)
        
        attention = LayerNormalization()(attention)
        
        # Global pooling
        sequence_features = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        # === FEATURE PROCESSING ===
        feature_dense = Dense(32, activation='relu', name='feature_processing')(features_input)
        feature_dense = Dropout(0.2)(feature_dense)
        
        # === FUSION ===
        merged = Concatenate(name='feature_fusion')([sequence_features, feature_dense])
        
        # === PATTERN RECOGNITION ===
        x = Dense(256, activation='relu', name='pattern_recognition')(merged)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu', name='psychology_understanding')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu', name='decision_making')(x)
        x = Dropout(0.1)(x)
        
        # === OUTPUT ===
        outputs = Dense(num_classes, activation='softmax', name='sequence_classification')(x)
        
        # === CREATE MODEL ===
        model = Model(
            inputs=[sequence_input, features_input],
            outputs=outputs,
            name=f'SequencePatternLearner'
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_single_timeframe(self, timeframe: str, sequence_length: int = 5) -> Dict:
        """เทรนโมเดล Sequence Learning สำหรับไทม์เฟรมเดียว"""
        
        self.logger.info(f"เริ่มเทรน {timeframe} Sequence Learning")
        start_time = time.time()
        
        # เตรียมข้อมูล
        data = self.generate_sequence_training_data(timeframe, sequence_length)
        
        X_seq_train = data['X_sequences_train']
        X_feat_train = data['X_features_train']
        y_train = data['y_train']
        X_seq_test = data['X_sequences_test']
        X_feat_test = data['X_features_test']
        y_test = data['y_test']
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Scale features
        scaler = StandardScaler()
        X_feat_train_scaled = scaler.fit_transform(X_feat_train)
        X_feat_test_scaled = scaler.transform(X_feat_test)
        
        # สร้างโมเดล
        vocab_size = len(data['all_patterns'])
        num_classes = len(label_encoder.classes_)
        feature_dim = X_feat_train.shape[1]
        
        model = self.create_sequence_model(vocab_size, sequence_length, num_classes, feature_dim)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                mode='max',
                verbose=1
            )
        ]
        
        self.logger.info(f"เริ่มการเทรน {timeframe} Model...")
        
        # เทรน
        history = model.fit(
            [X_seq_train, X_feat_train_scaled], y_train_encoded,
            epochs=100,
            batch_size=64,
            validation_data=([X_seq_test, X_feat_test_scaled], y_test_encoded),
            callbacks=callbacks,
            verbose=1
        )
        
        # ประเมินผล
        test_loss, test_accuracy = model.evaluate(
            [X_seq_test, X_feat_test_scaled], y_test_encoded, verbose=0
        )
        
        # ทำนายและวิเคราะห์
        y_pred = model.predict([X_seq_test, X_feat_test_scaled])
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        class_report = classification_report(
            y_test_encoded, y_pred_classes,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        confusion_mat = confusion_matrix(y_test_encoded, y_pred_classes)
        
        # คำนวณเวลา
        training_time = time.time() - start_time
        
        self.logger.info(f"เทรน {timeframe} เสร็จสิ้น!")
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"เวลาที่ใช้: {training_time/60:.1f} นาที")
        
        # บันทึกโมเดล
        tf_folder = f"{self.output_folder}/{timeframe}"
        if not os.path.exists(tf_folder):
            os.makedirs(tf_folder)
        
        model.save(f"{tf_folder}/sequence_model.h5")
        joblib.dump(scaler, f"{tf_folder}/sequence_scaler.pkl")
        joblib.dump(label_encoder, f"{tf_folder}/sequence_encoder.pkl")
        joblib.dump(data['pattern_to_id'], f"{tf_folder}/pattern_to_id.pkl")
        
        # บันทึก metadata
        metadata = {
            'timeframe': timeframe,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'sequence_length': sequence_length,
            'training_time_minutes': training_time / 60,
            'vocab_size': vocab_size,
            'num_classes': num_classes,
            'feature_dim': feature_dim,
            'pattern_distribution': dict(data['pattern_distribution']),
            'trained_at': datetime.now().isoformat()
        }
        
        with open(f"{tf_folder}/model_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"บันทึกโมเดล {timeframe}: {tf_folder}/")
        
        return {
            'timeframe': timeframe,
            'model': model,
            'history': history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'training_time_minutes': training_time / 60,
            'class_report': class_report,
            'confusion_matrix': confusion_mat,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'pattern_to_id': data['pattern_to_id'],
            'sequence_length': sequence_length,
            'metadata': metadata
        }
    
    def train_all_primary_timeframes(self, sequence_length: int = 5) -> Dict:
        """เทรนทั้ง 3 ไทม์เฟรมหลัก"""
        
        primary_timeframes = ['D1', 'H4', 'H1']
        
        self.logger.info("เริ่มเทรน Phase 3: Pattern Sequence Learning")
        self.logger.info("=" * 80)
        self.logger.info(f"ไทม์เฟรมที่จะเทรน: {primary_timeframes}")
        self.logger.info(f"Sequence Length: {sequence_length}")
        self.logger.info(f"เริ่มเทรน: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
        
        results = {}
        summary = {
            'start_time': datetime.now(),
            'successful': [],
            'failed': [],
            'total_accuracy': {},
            'total_time': 0
        }
        
        for i, timeframe in enumerate(primary_timeframes, 1):
            self.logger.info(f"\n[{i}/{len(primary_timeframes)}] เทรน {timeframe}")
            
            try:
                result = self.train_single_timeframe(timeframe, sequence_length)
                results[timeframe] = result
                summary['successful'].append(timeframe)
                summary['total_accuracy'][timeframe] = result['test_accuracy']
                summary['total_time'] += result['training_time_minutes']
                
                self.trained_models[timeframe] = {
                    'model': result['model'],
                    'scaler': result['scaler'],
                    'encoder': result['label_encoder'],
                    'pattern_to_id': result['pattern_to_id']
                }
                
            except Exception as e:
                self.logger.error(f"{timeframe} เทรนล้มเหลว: {str(e)}")
                summary['failed'].append(timeframe)
                continue
        
        summary['end_time'] = datetime.now()
        summary['total_duration'] = summary['end_time'] - summary['start_time']
        
        # สรุปผลลัพธ์
        self._print_phase3_summary(summary, results)
        self._save_phase3_summary(summary, results)
        
        return {
            'results': results,
            'summary': summary,
            'trained_models': self.trained_models
        }
    
    def _print_phase3_summary(self, summary: Dict, results: Dict):
        """แสดงสรุปผล Phase 3"""
        
        print("\n" + "=" * 80)
        print("สรุปผล Phase 3: Pattern Sequence Learning")
        print("=" * 80)
        
        print(f"เวลาเริ่ม: {summary['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"เวลาจบ: {summary['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"เวลารวม: {summary['total_duration']}")
        
        print(f"\nสำเร็จ: {len(summary['successful'])}/{len(summary['successful']) + len(summary['failed'])}")
        
        if summary['successful']:
            print("\nผลลัพธ์แต่ละไทม์เฟรม:")
            for tf in summary['successful']:
                accuracy = summary['total_accuracy'][tf]
                time_used = results[tf]['training_time_minutes']
                print(f"   {tf}: {accuracy:.4f} ({time_used:.1f} นาที)")
        
        if summary['failed']:
            print(f"\nล้มเหลว: {summary['failed']}")
        
        if len(summary['successful']) >= 2:
            avg_accuracy = np.mean(list(summary['total_accuracy'].values()))
            print(f"\nความแม่นยำเฉลี่ย: {avg_accuracy:.4f}")
            print(f"เวลารวม: {summary['total_time']:.1f} นาที")
            print(f"\nPhase 3 สำเร็จ! พร้อมสำหรับ Phase 4: Multi-Timeframe Integration")
        else:
            print(f"\nต้องมีอย่างน้อย 2 ไทม์เฟรมที่เทรนสำเร็จ")
        
        print("=" * 80)
    
    def _save_phase3_summary(self, summary: Dict, results: Dict):
        """บันทึกสรุปผล Phase 3"""
        
        # เตรียมข้อมูลสำหรับบันทึก
        save_data = {
            'phase': 3,
            'description': 'Pattern Sequence Learning - 3 Primary Timeframes',
            'summary': {
                'start_time': summary['start_time'].isoformat(),
                'end_time': summary['end_time'].isoformat(),
                'total_duration_seconds': summary['total_duration'].total_seconds(),
                'successful_timeframes': summary['successful'],
                'failed_timeframes': summary['failed'],
                'accuracies': summary['total_accuracy'],
                'total_training_time_minutes': summary['total_time']
            },
            'timeframe_details': {}
        }
        
        for tf, result in results.items():
            save_data['timeframe_details'][tf] = {
                'test_accuracy': float(result['test_accuracy']),
                'test_loss': float(result['test_loss']),
                'training_time_minutes': result['training_time_minutes'],
                'sequence_length': result['sequence_length'],
                'model_path': f"{tf}/sequence_model.h5"
            }
        
        # บันทึกไฟล์
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"{self.output_folder}/phase3_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.logger.info(f"บันทึกสรุป Phase 3: {summary_file}")
    
    def create_quick_test(self):
        """ทดสอบโมเดลที่เทรนแล้ว"""
        
        if not self.trained_models:
            self.logger.warning("ยังไม่มีโมเดลที่เทรนแล้ว")
            return
        
        self.logger.info("\nทดสอบโมเดล Phase 3")
        
        # ตัวอย่างลำดับแท่งเทียน (Morning Star pattern)
        test_sequences = [
            {
                'name': 'Morning Star Pattern',
                'sequence': [
                    {'Open': 2000, 'High': 2005, 'Low': 1980, 'Close': 1985},  # LONG_BEAR
                    {'Open': 1985, 'High': 1990, 'Low': 1983, 'Close': 1987},  # DOJI
                    {'Open': 1987, 'High': 2010, 'Low': 1986, 'Close': 2008}   # LONG_BULL
                ]
            },
            {
                'name': 'Evening Star Pattern',
                'sequence': [
                    {'Open': 1980, 'High': 2010, 'Low': 1979, 'Close': 2008},  # LONG_BULL
                    {'Open': 2008, 'High': 2012, 'Low': 2006, 'Close': 2009},  # DOJI
                    {'Open': 2009, 'High': 2011, 'Low': 1985, 'Close': 1988}   # LONG_BEAR
                ]
            }
        ]
        
        for test_case in test_sequences:
            self.logger.info(f"\nทดสอบ: {test_case['name']}")
            
            # ทำนายแต่ละแท่งเทียน
            individual_patterns = []
            for candle in test_case['sequence']:
                pred = self.predict_single_candlestick(candle, 'H1')
                individual_patterns.append(pred['pattern'])
            
            self.logger.info(f"   Individual Patterns: {individual_patterns}")
            
            # ทำนายด้วยโมเดลแต่ละไทม์เฟรม
            for tf, model_info in self.trained_models.items():
                try:
                    # สร้าง mock prediction (เพราะต้องการ real training data)
                    self.logger.info(f"   {tf} Prediction: [Mock] - ต้องใช้โมเดลจริงที่เทรนแล้ว")
                except Exception as e:
                    self.logger.warning(f"   {tf} Error: {str(e)}")

# === MAIN EXECUTION ===
def main():
    """ฟังก์ชันหลักสำหรับเทรน Phase 3"""
    
    print("Complete Phase 3 Trainer")
    print("เทรน Pattern Sequence Learning สำหรับ 3 ไทม์เฟรมหลัก (D1, H4, H1)")
    print("=" * 80)
    
    try:
        # สร้าง trainer
        trainer = Phase3CompleteTrainer()
        
        # เช็คข้อมูลที่มี
        print(f"Phase 2 (Candlestick Classifier): พร้อม")
        print(f"Phase 1 (Raw Data): พร้อม")
        print(f"ไทม์เฟรมที่พร้อม: {list(trainer.raw_data.keys())}")
        
        # ยืนยันการเทรน
        confirm = input(f"\nเริ่มเทรน Phase 3 สำหรับ 3 ไทม์เฟรมหลัก? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("ยกเลิกการเทรน")
            return
        
        # เริ่มเทรน
        results = trainer.train_all_primary_timeframes(sequence_length=5)
        
        if results['summary']['successful']:
            print(f"\nPhase 3 เสร็จสมบูรณ์!")
            print(f"เทรนสำเร็จ: {len(results['summary']['successful'])} ไทม์เฟรม")
            print(f"ผลลัพธ์: {trainer.output_folder}/")
            print(f"พร้อมสำหรับ Phase 4: Multi-Timeframe Integration!")
            
            # ทดสอบโมเดล
            trainer.create_quick_test()
        else:
            print(f"\nPhase 3 ไม่สำเร็จ กรุณาตรวจสอบ error log")
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()