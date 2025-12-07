import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pythainlp.tokenize import word_tokenize

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    return df

def thai_tokenizer(text):
    if not isinstance(text, str):
        return []
    return word_tokenize(text, engine="newmm")

def train_advanced_model(df):
    print("Preprocessing data (Advanced)...")
    
    # 1. Basic Cleaning
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    df['district_type'] = df['district'].astype(str) + "_" + df['type_clean'].astype(str)
    
    # 2. Cyclical Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 3. Filter & Cap
    feature_cols = [
        'type_clean', 'district', 'district_type', 'comment',
        'hour_sin', 'hour_cos', 'day_of_week', 'month_sin', 'month_cos', 'is_weekend', 
        'comment_length',
        'rainfall_mm', 'temperature_max', 'temperature_min'
    ]
    target_col = 'resolution_time_days'
    
    df = df.dropna(subset=feature_cols + [target_col])
    
    # Use 50% subsample for speed if needed, but let's try 100k rows for better generalization
    # If the dataset is huge (600k), training might be slow. Let's use 200k recent.
    df = df.sort_values('timestamp', ascending=False).head(200000)
    
    df['target_capped'] = df[target_col].clip(upper=365)
    y = np.log1p(df['target_capped'])
    X = df[feature_cols]

    print(f"Training on {len(df)} records...")

    # 4. Define Transformers
    
    # Text Pipeline: Tokenize -> TF-IDF -> SVD (reduce to 20 dim)
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, max_features=5000, min_df=5, max_df=0.9)),
        ('svd', TruncatedSVD(n_components=20, random_state=42))
    ])
    
    # Categorical Pipeline: Target Encoding
    # Note: TargetEncoder needs y during fit, so we use it in the main flow or wrapper.
    # We'll handle it separately or use sklearn's Compose which supports y in fit since v1.4?
    # Actually, let's manually split to be safe and clear.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting encoders and transformers...")
    
    # A. Text Features
    print("  - Processing Text (TF-IDF + SVD)... this may take a moment.")
    X_train_text = text_pipeline.fit_transform(X_train['comment'])
    X_test_text = text_pipeline.transform(X_test['comment'])
    
    # B. Target Encoding for Categoricals
    print("  - Target Encoding Categories...")
    te_type = TargetEncoder(smooth="auto")
    te_dist = TargetEncoder(smooth="auto")
    te_dist_type = TargetEncoder(smooth="auto")
    
    X_train_type = te_type.fit_transform(X_train[['type_clean']], y_train)
    X_test_type = te_type.transform(X_test[['type_clean']])
    
    X_train_dist = te_dist.fit_transform(X_train[['district']], y_train)
    X_test_dist = te_dist.transform(X_test[['district']])

    X_train_dt = te_dist_type.fit_transform(X_train[['district_type']], y_train)
    X_test_dt = te_dist_type.transform(X_test[['district_type']])
    
    # C. Numerical Features
    num_cols = ['hour_sin', 'hour_cos', 'day_of_week', 'month_sin', 'month_cos', 
                'is_weekend', 'comment_length', 'rainfall_mm', 'temperature_max', 'temperature_min']
    
    X_train_num = X_train[num_cols].values
    X_test_num = X_test[num_cols].values
    
    # Combine All Features
    X_train_final = np.hstack([X_train_num, X_train_type, X_train_dist, X_train_dt, X_train_text])
    X_test_final = np.hstack([X_test_num, X_test_type, X_test_dist, X_test_dt, X_test_text])
    
    print(f"Final Feature Shape: {X_train_final.shape}")
    
    # 5. Train Model
    print("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        colsample_bytree=0.8,
        subsample=0.8,
        objective='reg:absoluteerror',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train_final, y_train)
    
    # 6. Evaluate
    print("Evaluating...")
    y_pred_log = model.predict(X_test_final)
    y_pred = np.expm1(y_pred_log)
    y_test_days = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test_days, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_days, y_pred))
    r2 = r2_score(y_test_days, y_pred)
    
    print(f"\n--- Advanced Model Performance ---")
    print(f"MAE:  {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R2:   {r2:.4f}")
    
    # 7. Save Artifacts
    print("Saving artifacts...")
    artifacts = {
        'model': model,
        'text_pipeline': text_pipeline,
        'te_type': te_type,
        'te_dist': te_dist,
        'te_dist_type': te_dist_type,
        'feature_names_num': num_cols
    }
    joblib.dump(artifacts, os.path.join(MODEL_DIR, "resolution_model_advanced.joblib"))
    
    # Save Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_days[:2000], y=y_pred[:2000], alpha=0.3)
    plt.plot([0, 365], [0, 365], 'r--')
    plt.title(f"Advanced Model (MAE: {mae:.2f})")
    plt.savefig(os.path.join(MODEL_DIR, "prediction_scatter_advanced.png"))

if __name__ == "__main__":
    df = load_data()
    train_advanced_model(df)
