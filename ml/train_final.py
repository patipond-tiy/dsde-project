import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class HistoricalMedianEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to compute historical medians (Global, District-level, Type-level).
    Handles unseen values by falling back to broader averages.
    NO LEAKAGE: Must be fit on Train and transformed on Test.
    """
    def __init__(self, target_col='resolution_time_days'):
        self.target_col = target_col
        self.type_medians = {}
        self.dist_medians = {}
        self.dt_medians = {}
        self.global_median = 0
        
    def fit(self, X, y=None):
        # Join X and y for calculation (y must be passed)
        if y is None:
            raise ValueError("Target y required for fitting.")
        
        df = X.copy()
        df[self.target_col] = y
        
        self.global_median = df[self.target_col].median()
        self.type_medians = df.groupby('type_clean')[self.target_col].median().to_dict()
        self.dist_medians = df.groupby('district')[self.target_col].median().to_dict()
        self.dt_medians = df.groupby('district_type')[self.target_col].median().to_dict()
        
        return self
    
    def transform(self, X):
        # Map values
        type_med = X['type_clean'].map(self.type_medians).fillna(self.global_median)
        dist_med = X['district'].map(self.dist_medians).fillna(self.global_median)
        dt_med = X['district_type'].map(self.dt_medians)
        
        # Fallback for interaction: If Bang Rak_Bridge is new, use Bridge median
        dt_med = dt_med.fillna(type_med)
        
        return pd.DataFrame({
            'hist_type_median': type_med,
            'hist_dist_median': dist_med,
            'hist_interaction_median': dt_med
        })

def thai_tokenizer(text):
    if not isinstance(text, str):
        return []
    return word_tokenize(text, engine="newmm")

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    return df

def train_final_model(df):
    print("Preprocessing data (Final Architecture)...")
    
    # 1. Clean & Filter
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    df['district_type'] = df['district'].astype(str) + "_" + df['type_clean'].astype(str)
    
    # Filter short noise (admin closures)
    df = df[df['resolution_time_days'] > 0.05] 
    
    # 2. Cyclical Temporal Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    feature_cols = [
        'type_clean', 'district', 'district_type', 'comment',
        'hour_sin', 'hour_cos', 'day_of_week', 'month_sin', 'month_cos', 'is_weekend', 
        'comment_length',
        'rainfall_mm', 'temperature_max', 'temperature_min'
    ]
    target_col = 'resolution_time_days'
    
    df = df.dropna(subset=feature_cols + [target_col])
    
    # Use most recent 200k
    df = df.sort_values('timestamp', ascending=False).head(200000)
    
    # Target: Log Transform
    # We clip at 365 to handle extreme outliers, but keep the 0.05-365 range.
    y_raw = df[target_col].clip(upper=365)
    y = np.log1p(y_raw)
    X = df[feature_cols]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting Encoders & Transformers...")
    
    # A. Historical Encoder (The "Cheat Sheet")
    # This creates the high-correlation features: hist_interaction_median, etc.
    hist_encoder = HistoricalMedianEncoder(target_col='target_y')
    # Fit on TRAINING target only (No Leakage)
    hist_encoder.fit(X_train, np.expm1(y_train)) # Pass raw days to encoder for meaningful medians
    
    X_train_hist = hist_encoder.transform(X_train)
    X_test_hist = hist_encoder.transform(X_test)
    
    # Apply Log to these features because the relationship is log-linear
    X_train_hist_log = np.log1p(X_train_hist)
    X_test_hist_log = np.log1p(X_test_hist)
    
    # B. Text Pipeline (TF-IDF + SVD)
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, max_features=2000, min_df=10, max_df=0.8)),
        ('svd', TruncatedSVD(n_components=10, random_state=42))
    ])
    
    X_train_text = text_pipeline.fit_transform(X_train['comment'])
    X_test_text = text_pipeline.transform(X_test['comment'])
    
    # C. Numerical Features
    num_cols = ['hour_sin', 'hour_cos', 'day_of_week', 'month_sin', 'month_cos', 
                'is_weekend', 'comment_length', 'rainfall_mm', 'temperature_max', 'temperature_min']
    
    X_train_num = X_train[num_cols].values
    X_test_num = X_test[num_cols].values
    
    # D. Combine
    # Stack: [History(3), Num(10), Text(10)] = 23 features
    X_train_final = np.hstack([X_train_hist_log, X_train_num, X_train_text])
    X_test_final = np.hstack([X_test_hist_log, X_test_num, X_test_text])
    
    print(f"Final Feature Shape: {X_train_final.shape}")
    
    # 4. Train Model
    print("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train_final, y_train)
    
    # 5. Evaluate
    print("Evaluating...")
    y_pred_log = model.predict(X_test_final)
    y_pred = np.expm1(y_pred_log)
    y_test_days = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test_days, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_days, y_pred))
    r2 = r2_score(y_test_days, y_pred)
    
    print(f"\n--- Final Model Performance ---")
    print(f"MAE:  {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R2:   {r2:.4f}")
    
    # 6. Save Artifacts
    print("Saving artifacts...")
    artifacts = {
        'model': model,
        'hist_encoder': hist_encoder,
        'text_pipeline': text_pipeline,
        'feature_names_num': num_cols
    }
    joblib.dump(artifacts, os.path.join(MODEL_DIR, "resolution_model_final.joblib"))
    print("Done.")

if __name__ == "__main__":
    df = load_data()
    train_final_model(df)
