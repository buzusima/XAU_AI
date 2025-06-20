import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from typing import Dict, List, Tuple, Optional
import logging

class TrueCandlestickClassifier:
    """
    สอน AI ให้รู้จักและจำแนกแท่งเทียนแต่ละประเภท
    ตาม Traditional Candlestick Analysis
    """
    
    def __init__(self, data_folder: str = "raw_ai_data_XAUUSD_c"):
        self.data_folder = data_folder
        self.data = {}
        self.scaler = None
        self.label_encoder = None
        self.model = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load raw data
        self._load_raw_data()
    
    def _load_raw_data(self):
        """โหลดข้อมูล Raw"""
        if not os.path.exists(self.data_folder):
            self.logger.error(f"ไม่พบโฟลเดอร์: {self.data_folder}")
            return
        
        self.logger.info("📂 โหลดข้อมูล Raw Candlestick...")
        
        # โหลดข้อมูลทุกไทม์เฟรม
        timeframes = ['M1', 'M5', 'M30', 'H1', 'H4', 'D1']
        
        all_data = []
        successful_loads = 0
        
        for tf in timeframes:
            file_path = f"{self.data_folder}/XAUUSD.c_{tf}_raw.csv"
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    df['Timeframe'] = tf  # เพิ่มข้อมูลไทม์เฟรม
                    
                    # เพิ่ม timeframe weight (ไทม์เฟรมใหญ่มีน้ำหนักมากกว่า)
                    tf_weights = {
                        'M1': 1.0,
                        'M5': 1.2, 
                        'M30': 1.5,
                        'H1': 2.0,
                        'H4': 3.0,
                        'D1': 4.0
                    }
                    df['TF_weight'] = tf_weights[tf]
                    
                    all_data.append(df)
                    successful_loads += 1
                    self.logger.info(f"✅ {tf}: {len(df):,} แท่ง")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  {tf}: ไม่สามารถโหลดได้ - {str(e)}")
            else:
                self.logger.warning(f"❌ ไม่พบไฟล์: {file_path}")
        
        if all_data:
            # รวมข้อมูลทุกไทม์เฟรม
            self.data = pd.concat(all_data, ignore_index=False)
            self.logger.info(f"📊 รวมข้อมูลทั้งหมด: {len(self.data):,} แท่ง จาก {successful_loads} ไทม์เฟรม")
            
            # แสดงการกระจายข้อมูลตามไทม์เฟรม
            tf_distribution = self.data['Timeframe'].value_counts()
            self.logger.info("📋 การกระจายข้อมูลตามไทม์เฟรม:")
            for tf, count in tf_distribution.items():
                percentage = (count / len(self.data)) * 100
                self.logger.info(f"   {tf}: {count:,} แท่ง ({percentage:.1f}%)")
        else:
            self.logger.error("❌ ไม่พบข้อมูลเลย")
    
    def create_candlestick_anatomy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        สร้างฟีเจอร์กายวิภาคแท่งเทียนพื้นฐาน
        เฉพาะสิ่งที่ต้องการสำหรับจำแนกแท่งเทียน
        """
        df_features = df.copy()
        
        # === BASIC ANATOMY ===
        # Body (ตัวเทียน)
        df_features['Body_size'] = abs(df['Close'] - df['Open'])
        df_features['Body_direction'] = np.where(df['Close'] > df['Open'], 1, -1)  # 1=Bullish, -1=Bearish
        
        # Shadows (เงา)
        df_features['Upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df_features['Lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        
        # Total range
        df_features['Total_range'] = df['High'] - df['Low']
        
        # ป้องกันการหารด้วยศูนย์
        df_features['Total_range'] = df_features['Total_range'].replace(0, 1e-8)
        
        # === RATIOS (อัตราส่วนสำคัญ) ===
        # Body ratio = ตัวเทียน / ช่วงราคาทั้งหมด
        df_features['Body_ratio'] = df_features['Body_size'] / df_features['Total_range']
        
        # Shadow ratios
        df_features['Upper_shadow_ratio'] = df_features['Upper_shadow'] / df_features['Total_range']
        df_features['Lower_shadow_ratio'] = df_features['Lower_shadow'] / df_features['Total_range']
        
        # === POSITIONS (ตำแหน่งราคา) ===
        # ตำแหน่งของ Open และ Close ในช่วงราคา (0-1)
        df_features['Open_position'] = (df['Open'] - df['Low']) / df_features['Total_range']
        df_features['Close_position'] = (df['Close'] - df['Low']) / df_features['Total_range']
        
        # === SYMMETRY (ความสมดุล) ===
        # Shadow symmetry
        total_shadow = df_features['Upper_shadow'] + df_features['Lower_shadow']
        total_shadow = total_shadow.replace(0, 1e-8)
        df_features['Shadow_symmetry'] = abs(df_features['Upper_shadow'] - df_features['Lower_shadow']) / total_shadow
        
        # === SIZE CATEGORIES ===
        # แบ่งขนาดเทียน relative กับ recent ATR
        atr_period = 14
        df_features['ATR'] = df_features['Total_range'].rolling(atr_period).mean()
        df_features['Size_vs_ATR'] = df_features['Total_range'] / df_features['ATR']
        df_features['Body_vs_ATR'] = df_features['Body_size'] / df_features['ATR']
        
        # ทำความสะอาด
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(method='ffill').fillna(0)
        
        return df_features
    
    def classify_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        จำแนกแท่งเทียนตาม Traditional Candlestick Analysis
        นี่คือ Ground Truth ที่เราจะสอน AI
        """
        df_classified = df.copy()
        
        # === TRADITIONAL CANDLESTICK PATTERNS ===
        
        # 1. DOJI (ความลังเล, ไม่แน่ใจ)
        doji_threshold = 0.1
        is_doji = df['Body_ratio'] <= doji_threshold
        
        # 2. MARUBOZU (ความแน่วแน่, momentum แรง)
        marubozu_body_threshold = 0.8
        marubozu_shadow_threshold = 0.1
        is_marubozu = (
            (df['Body_ratio'] >= marubozu_body_threshold) &
            (df['Upper_shadow_ratio'] <= marubozu_shadow_threshold) &
            (df['Lower_shadow_ratio'] <= marubozu_shadow_threshold)
        )
        
        # 3. HAMMER (การกลับตัว bullish ที่ support)
        hammer_body_threshold = 0.3
        hammer_lower_threshold = 0.6
        hammer_upper_threshold = 0.1
        is_hammer = (
            (df['Body_ratio'] <= hammer_body_threshold) &
            (df['Lower_shadow_ratio'] >= hammer_lower_threshold) &
            (df['Upper_shadow_ratio'] <= hammer_upper_threshold)
        )
        
        # 4. SHOOTING STAR (การกลับตัว bearish ที่ resistance)
        star_body_threshold = 0.3
        star_upper_threshold = 0.6
        star_lower_threshold = 0.1
        is_shooting_star = (
            (df['Body_ratio'] <= star_body_threshold) &
            (df['Upper_shadow_ratio'] >= star_upper_threshold) &
            (df['Lower_shadow_ratio'] <= star_lower_threshold)
        )
        
        # 5. SPINNING TOP (ความไม่แน่นอน, consolidation)
        spinning_body_threshold = 0.3
        spinning_shadow_threshold = 0.3
        is_spinning_top = (
            (df['Body_ratio'] <= spinning_body_threshold) &
            (df['Upper_shadow_ratio'] >= spinning_shadow_threshold) &
            (df['Lower_shadow_ratio'] >= spinning_shadow_threshold) &
            ~is_doji  # ไม่ใช่ doji
        )
        
        # 6. LONG BODY (แรงซื้อ/ขายแรง)
        long_body_threshold = 0.6
        is_long_bullish = (
            (df['Body_ratio'] >= long_body_threshold) &
            (df['Body_direction'] == 1) &
            ~is_marubozu
        )
        
        is_long_bearish = (
            (df['Body_ratio'] >= long_body_threshold) &
            (df['Body_direction'] == -1) &
            ~is_marubozu
        )
        
        # 7. SMALL BODY (momentum อ่อน)
        small_body_threshold = 0.4
        is_small_body = (
            (df['Body_ratio'] <= small_body_threshold) &
            ~is_doji & ~is_hammer & ~is_shooting_star & ~is_spinning_top
        )
        
        # === ASSIGN PATTERNS ===
        conditions = [
            is_doji,
            is_marubozu & (df['Body_direction'] == 1),   # Bullish Marubozu
            is_marubozu & (df['Body_direction'] == -1),  # Bearish Marubozu
            is_hammer,
            is_shooting_star,
            is_spinning_top,
            is_long_bullish,
            is_long_bearish,
            is_small_body & (df['Body_direction'] == 1), # Small Bullish
            is_small_body & (df['Body_direction'] == -1), # Small Bearish
        ]
        
        choices = [
            'DOJI',
            'MARUBOZU_BULL',
            'MARUBOZU_BEAR', 
            'HAMMER',
            'SHOOTING_STAR',
            'SPINNING_TOP',
            'LONG_BULL',
            'LONG_BEAR',
            'SMALL_BULL',
            'SMALL_BEAR'
        ]
        
        df_classified['Candlestick_pattern'] = np.select(conditions, choices, default='NORMAL')
        
        # === MARKET PSYCHOLOGY ===
        # แปลง pattern เป็น market psychology
        psychology_map = {
            'DOJI': 'INDECISION',
            'MARUBOZU_BULL': 'STRONG_BULLISH',
            'MARUBOZU_BEAR': 'STRONG_BEARISH',
            'HAMMER': 'BULLISH_REVERSAL',
            'SHOOTING_STAR': 'BEARISH_REVERSAL', 
            'SPINNING_TOP': 'UNCERTAINTY',
            'LONG_BULL': 'BULLISH',
            'LONG_BEAR': 'BEARISH',
            'SMALL_BULL': 'WEAK_BULLISH',
            'SMALL_BEAR': 'WEAK_BEARISH',
            'NORMAL': 'NEUTRAL'
        }
        
        df_classified['Market_psychology'] = df_classified['Candlestick_pattern'].map(psychology_map)
        
        return df_classified
    
    def prepare_classification_data(self) -> Dict:
        """
        เตรียมข้อมูลสำหรับเทรน Candlestick Classifier
        """
        if self.data.empty:
            raise ValueError("ไม่มีข้อมูลสำหรับเทรน")
        
        self.logger.info("🔧 เตรียมข้อมูล Candlestick Classification...")
        
        # สร้างฟีเจอร์
        df_features = self.create_candlestick_anatomy_features(self.data)
        
        # จำแนกแท่งเทียน (Ground Truth)
        df_classified = self.classify_candlestick_patterns(df_features)
        
        # เลือกฟีเจอร์สำหรับ AI
        feature_columns = [
            'Body_ratio', 'Upper_shadow_ratio', 'Lower_shadow_ratio',
            'Open_position', 'Close_position', 'Shadow_symmetry',
            'Size_vs_ATR', 'Body_vs_ATR', 'Body_direction',
            'Hour', 'Day_of_week'  # Time context
        ]
        
        # เพิ่มฟีเจอร์ไทม์เฟรม
        if 'Timeframe' in df_classified.columns:
            # Encode timeframe เป็นตัวเลข
            tf_mapping = {'M1': 1, 'M5': 5, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
            df_classified['TF_minutes'] = df_classified['Timeframe'].map(tf_mapping)
            feature_columns.append('TF_minutes')
            
            # เพิ่ม timeframe weight
            if 'TF_weight' in df_classified.columns:
                feature_columns.append('TF_weight')
        
        # กรองเฉพาะ columns ที่มี
        available_features = [col for col in feature_columns if col in df_classified.columns]
        
        # เตรียม X และ y
        X = df_classified[available_features].copy()
        y = df_classified['Candlestick_pattern'].copy()
        
        # ลบข้อมูลที่ไม่สมบูรณ์
        valid_indices = X.notna().all(axis=1) & y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Time-based split
        split_date = '2024-01-01'
        train_mask = X.index < split_date
        test_mask = X.index >= split_date
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        self.logger.info(f"📊 Training: {len(X_train):,} แท่ง")
        self.logger.info(f"📊 Testing: {len(X_test):,} แท่ง")
        
        # แสดงการกระจายข้อมูลตามไทม์เฟรม
        if 'Timeframe' in df_classified.columns:
            self.logger.info("📋 การกระจายข้อมูลเทรนตามไทม์เฟรม:")
            train_tf_dist = df_classified.loc[X_train.index, 'Timeframe'].value_counts()
            for tf, count in train_tf_dist.items():
                percentage = (count / len(X_train)) * 100
                self.logger.info(f"   {tf}: {count:,} แท่ง ({percentage:.1f}%)")
        
        self.logger.info(f"🎯 Pattern distribution:")
        pattern_counts = y_train.value_counts()
        for pattern, count in pattern_counts.items():
            percentage = (count / len(y_train)) * 100
            self.logger.info(f"   {pattern}: {count:,} ({percentage:.1f}%)")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': available_features,
            'pattern_counts': pattern_counts,
            'timeframe_distribution': train_tf_dist if 'Timeframe' in df_classified.columns else None
        }
    
    def create_classifier_model(self, input_shape: int, num_classes: int) -> Model:
        """
        สร้างโมเดล Candlestick Pattern Classifier
        """
        self.logger.info("🧠 สร้างโมเดล Candlestick Pattern Classifier")
        
        inputs = Input(shape=(input_shape,), name='candlestick_anatomy')
        
        # Feature extraction
        x = Dense(128, activation='relu', name='anatomy_analysis')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu', name='pattern_recognition')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu', name='psychology_understanding')(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax', name='pattern_classification')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='CandlestickPatternClassifier')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_classifier(self) -> Dict:
        """
        เทรนโมเดล Candlestick Pattern Classifier
        """
        self.logger.info("🚀 เริ่มเทรน Candlestick Pattern Classifier")
        
        # เตรียมข้อมูล
        data_splits = self.prepare_classification_data()
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # สร้างโมเดล
        self.model = self.create_classifier_model(X_train_scaled.shape[1], len(self.label_encoder.classes_))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, min_lr=1e-6, mode='max')
        ]
        
        self.logger.info("🔥 เริ่มการเทรน...")
        
        # เทรน
        history = self.model.fit(
            X_train_scaled, y_train_encoded,
            epochs=100,
            batch_size=128,
            validation_data=(X_test_scaled, y_test_encoded),
            callbacks=callbacks,
            verbose=1
        )
        
        # ประเมินผล
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
        
        # ทำนายและดู confusion matrix
        y_pred = self.model.predict(X_test_scaled)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        class_report = classification_report(
            y_test_encoded, y_pred_classes,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        confusion_mat = confusion_matrix(y_test_encoded, y_pred_classes)
        
        self.logger.info(f"✅ เทรนเสร็จสิ้น!")
        self.logger.info(f"🎯 Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"📊 Model ตอนนี้รู้จัก {len(self.label_encoder.classes_)} รูปแบบแท่งเทียน")
        
        # แสดงประสิทธิภาพแต่ละ pattern
        self.logger.info("📋 ประสิทธิภาพแต่ละ Pattern:")
        for pattern in self.label_encoder.classes_:
            if pattern in class_report:
                precision = class_report[pattern]['precision']
                recall = class_report[pattern]['recall']
                f1 = class_report[pattern]['f1-score']
                support = class_report[pattern]['support']
                self.logger.info(f"   {pattern}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, N={support}")
        
        return {
            'model': self.model,
            'history': history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'class_report': class_report,
            'confusion_matrix': confusion_mat,
            'pattern_counts': data_splits['pattern_counts'],
            'feature_names': data_splits['feature_names']
        }
    
    def visualize_results(self, results: Dict):
        """
        แสดงผลการเทรน Candlestick Classifier
        """
        plt.figure(figsize=(20, 12))
        
        # 1. Training History
        plt.subplot(3, 4, 1)
        history = results['history']
        plt.plot(history.history['accuracy'], label='Training', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        plt.title('Model Accuracy', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 2)
        plt.plot(history.history['loss'], label='Training', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
        plt.title('Model Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        plt.subplot(3, 4, 3)
        conf_matrix = results['confusion_matrix']
        pattern_names = self.label_encoder.classes_
        
        sns.heatmap(conf_matrix, annot=True, fmt='d',
                   xticklabels=pattern_names, yticklabels=pattern_names,
                   cmap='Blues')
        plt.title('Confusion Matrix', fontweight='bold')
        plt.xlabel('Predicted Pattern')
        plt.ylabel('True Pattern')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 3. Pattern Distribution
        plt.subplot(3, 4, 4)
        pattern_counts = results['pattern_counts']
        plt.bar(pattern_counts.index, pattern_counts.values, alpha=0.8)
        plt.title('Pattern Distribution in Training', fontweight='bold')
        plt.xlabel('Candlestick Pattern')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 4. Per-Pattern Performance
        plt.subplot(3, 4, 5)
        class_report = results['class_report']
        patterns = [p for p in pattern_names if p in class_report]
        precisions = [class_report[p]['precision'] for p in patterns]
        recalls = [class_report[p]['recall'] for p in patterns]
        f1_scores = [class_report[p]['f1-score'] for p in patterns]
        
        x = np.arange(len(patterns))
        width = 0.25
        
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)  
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.title('Per-Pattern Performance', fontweight='bold')
        plt.xlabel('Pattern')
        plt.ylabel('Score')
        plt.xticks(x, patterns, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Sample Predictions
        plt.subplot(3, 4, 6)
        plt.axis('off')
        
        summary_text = f"""
🕯️ CANDLESTICK AI CLASSIFIER

✅ MISSION: สอน AI รู้จักแท่งเทียน

📊 PERFORMANCE:
• Accuracy: {results['test_accuracy']:.3f}
• Patterns: {len(pattern_names)}
• Features: {len(results['feature_names'])}

🎯 PATTERNS LEARNED:
• DOJI (ความลังเล)
• HAMMER (Bullish Reversal)  
• SHOOTING STAR (Bearish Reversal)
• MARUBOZU (Strong Momentum)
• SPINNING TOP (Uncertainty)
• LONG BODY (Strong Direction)

🧠 AI ตอนนี้เข้าใจ:
- กายวิภาคแท่งเทียน
- Market Psychology
- Traditional Patterns

✅ READY FOR NEXT PHASE:
- Pattern Sequences
- Multi-Timeframe Context
- Trading Decisions
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def predict_candlestick(self, ohlc_data: Dict, timeframe: str = 'H1') -> Dict:
        """
        ใช้โมเดลทำนายแท่งเทียน
        
        Args:
            ohlc_data: {'Open': float, 'High': float, 'Low': float, 'Close': float}
            timeframe: ไทม์เฟรมของแท่งเทียนนี้ (default: 'H1')
        
        Returns:
            {'pattern': str, 'psychology': str, 'confidence': float}
        """
        if self.model is None or self.scaler is None:
            raise ValueError("โมเดลยังไม่ได้เทรน กรุณาเทรนก่อน")
        
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
            'Size_vs_ATR': 1.0,  # ไม่มี ATR context
            'Body_vs_ATR': 1.0,
            'Body_direction': body_direction,
            'Hour': 12,  # Default
            'Day_of_week': 1  # Default
        }
        
        # เพิ่มฟีเจอร์ไทม์เฟรม (ถ้าโมเดลต้องการ)
        tf_mapping = {'M1': 1, 'M5': 5, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
        tf_weights = {'M1': 1.0, 'M5': 1.2, 'M30': 1.5, 'H1': 2.0, 'H4': 3.0, 'D1': 4.0}
        
        features['TF_minutes'] = tf_mapping.get(timeframe, 60)
        features['TF_weight'] = tf_weights.get(timeframe, 2.0)
        
        # สร้าง input array ตาม feature names ที่โมเดลต้องการ
        try:
            X = np.array([[features[col] for col in self.scaler.feature_names_in_]])
        except KeyError as e:
            # ถ้ามีฟีเจอร์ที่ขาดหาย ให้ข้ามไป
            missing_feature = str(e).strip("'")
            self.logger.warning(f"⚠️  ฟีเจอร์ {missing_feature} ไม่มีในข้อมูลทำนาย ใช้ค่า default")
            
            # สร้าง features array ที่มีครบ
            feature_values = []
            for col in self.scaler.feature_names_in_:
                if col in features:
                    feature_values.append(features[col])
                else:
                    # ใช้ค่า default สำหรับฟีเจอร์ที่ขาด
                    if 'TF_' in col:
                        feature_values.append(60)  # Default H1
                    elif 'Hour' in col:
                        feature_values.append(12)
                    elif 'Day' in col:
                        feature_values.append(1)
                    else:
                        feature_values.append(1.0)
            
            X = np.array([feature_values])
        
        X_scaled = self.scaler.transform(X)
        
        # ทำนาย
        prediction = self.model.predict(X_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        pattern_name = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Map psychology
        psychology_map = {
            'DOJI': 'INDECISION',
            'MARUBOZU_BULL': 'STRONG_BULLISH',
            'MARUBOZU_BEAR': 'STRONG_BEARISH',
            'HAMMER': 'BULLISH_REVERSAL',
            'SHOOTING_STAR': 'BEARISH_REVERSAL',
            'SPINNING_TOP': 'UNCERTAINTY',
            'LONG_BULL': 'BULLISH',
            'LONG_BEAR': 'BEARISH',
            'SMALL_BULL': 'WEAK_BULLISH',
            'SMALL_BEAR': 'WEAK_BEARISH',
            'NORMAL': 'NEUTRAL'
        }
        
        psychology = psychology_map.get(pattern_name, 'NEUTRAL')
        
        return {
            'pattern': pattern_name,
            'psychology': psychology,
            'confidence': float(confidence),
            'timeframe': timeframe,
            'all_probabilities': {
                self.label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(prediction[0])
            }
        }
    
    def save_classifier(self, folder: str = "candlestick_classifier"):
        """บันทึกโมเดล"""
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if self.model:
            self.model.save(f"{folder}/candlestick_classifier.h5")
            joblib.dump(self.scaler, f"{folder}/scaler.pkl")
            joblib.dump(self.label_encoder, f"{folder}/label_encoder.pkl")
            self.logger.info(f"💾 บันทึกโมเดลเรียบร้อย: {folder}/")

# === Usage Example ===
if __name__ == "__main__":
    print("🕯️ True Candlestick Pattern Classifier")
    print("สอน AI ให้รู้จักแท่งเทียนแต่ละประเภท")
    print("=" * 60)
    
    # สร้าง classifier
    classifier = TrueCandlestickClassifier("raw_ai_data_XAUUSD_c")
    
    if classifier.data.empty:
        print("❌ ไม่พบข้อมูลสำหรับเทรน")
        exit()
    
    print(f"✅ โหลดข้อมูล: {len(classifier.data):,} แท่งเทียน")
    
    try:
        # เทรนโมเดล
        print("\n🚀 เริ่มเทรน Candlestick Pattern Classifier...")
        results = classifier.train_classifier()
        
        # แสดงผลลัพธ์
        classifier.visualize_results(results)
        
        # บันทึกโมเดล
        classifier.save_classifier()
        
        print(f"\n✅ เทรนเสร็จสมบูรณ์!")
        print(f"🧠 AI ตอนนี้รู้จักแท่งเทียน {len(classifier.label_encoder.classes_)} รูปแบบ!")
        print(f"🎯 Accuracy: {results['test_accuracy']:.3f}")
        
        # ทดสอบการทำนาย
        print(f"\n🔮 ทดสอบการทำนาย:")
        
        # ทดสอบหลายแท่งเทียนและหลายไทม์เฟรม (ครบ 6 ไทม์เฟรม!)
        test_candles = [
            ({'Open': 2000, 'High': 2010, 'Low': 1990, 'Close': 2005}, 'M1', 'Normal Bullish M1'),
            ({'Open': 2000, 'High': 2003, 'Low': 1995, 'Close': 2001}, 'M5', 'Small Body M5'),
            ({'Open': 2000, 'High': 2005, 'Low': 1985, 'Close': 1999}, 'M30', 'Hammer-like M30'),
            ({'Open': 2000, 'High': 2010, 'Low': 1990, 'Close': 2005}, 'H1', 'Normal Bullish H1'),
            ({'Open': 2000, 'High': 2002, 'Low': 1980, 'Close': 1999}, 'H4', 'Hammer-like H4'),
            ({'Open': 2000, 'High': 2040, 'Low': 1995, 'Close': 2001}, 'D1', 'Shooting Star-like D1'),
        ]
        
        for candle_data, tf, description in test_candles:
            try:
                prediction = classifier.predict_candlestick(candle_data, tf)
                print(f"   {description}: {prediction['pattern']} | {prediction['psychology']} | {prediction['confidence']:.3f}")
            except Exception as e:
                print(f"   {description}: Error - {str(e)}")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        import traceback
        traceback.print_exc()