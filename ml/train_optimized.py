import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pythainlp.tokenize import word_tokenize
from datetime import datetime
import shutil

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

def train_optimized_model(df):
    print("Preprocessing data (Full Scale Optimized Mode)...")
    
    # 1. Feature Engineering
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    df['district_type'] = df['district'].astype(str) + "_" + df['type_clean'].astype(str)
    
    infra_keywords = ['ถนน', 'สะพาน', 'ท่อระบายน้ำ', 'ก่อสร้าง', 'คลอง', 'ทางเท้า', 'แสงสว่าง']
    df['category_group'] = df['type_clean'].apply(lambda x: 'Infrastructure' if x in infra_keywords else 'Service')

    # 2. Strict Filtering
    print(f"Original Count: {len(df)}")
    df = df[df['resolution_time_days'] > 0.04] 
    print(f"Filtered Count (>1hr): {len(df)}")
    
    feature_cols = [
        'type_clean', 'district', 'subdistrict', 'district_type', 'category_group', 'comment',
        'is_rainy_season'
    ]
    target_col = 'resolution_time_days'
    
    df = df.dropna(subset=feature_cols + [target_col])
    
    # USE ALL DATA (No Head)
    df = df.sort_values('timestamp', ascending=False)
    print(f"Training on full dataset: {len(df)} records")
    
    df['target_capped'] = df[target_col].clip(upper=365)
    y = np.log1p(df['target_capped'])
    X = df[feature_cols]

    # 3. Sample Weights
    print("Calculating Sample Weights...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=df['type_clean'])
    
    # 4. Split Data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )
    
    print("Fitting encoders and transformers on Train set...")
    
    # A. Text Pipeline
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, max_features=3000, min_df=10, max_df=0.8)),
        ('svd', TruncatedSVD(n_components=15, random_state=42))
    ])
    
    X_train_text = text_pipeline.fit_transform(X_train['comment'])
    X_test_text = text_pipeline.transform(X_test['comment'])
    
    # B. Target Encoding
    te_type = TargetEncoder(smooth="auto")
    te_dist = TargetEncoder(smooth="auto")
    te_sub = TargetEncoder(smooth="auto")
    te_dist_type = TargetEncoder(smooth="auto")
    te_group = TargetEncoder(smooth="auto")
    
    X_train_type = te_type.fit_transform(X_train[['type_clean']], y_train)
    X_test_type = te_type.transform(X_test[['type_clean']])
    
    X_train_dist = te_dist.fit_transform(X_train[['district']], y_train)
    X_test_dist = te_dist.transform(X_test[['district']])

    X_train_sub = te_sub.fit_transform(X_train[['subdistrict']], y_train)
    X_test_sub = te_sub.transform(X_test[['subdistrict']])

    X_train_dt = te_dist_type.fit_transform(X_train[['district_type']], y_train)
    X_test_dt = te_dist_type.transform(X_test[['district_type']])
    
    X_train_grp = te_group.fit_transform(X_train[['category_group']], y_train)
    X_test_grp = te_group.transform(X_test[['category_group']])
    
    # Delta Feature
    X_train_delta = X_train_dt - X_train_type
    X_test_delta = X_test_dt - X_test_type
    
    # C. Numerical Features
    num_cols = ['is_rainy_season']
    X_train_num = X_train[num_cols].values
    X_test_num = X_test[num_cols].values
    
    # Stack
    X_train_final = np.hstack([X_train_num, X_train_type, X_train_dist, X_train_sub, X_train_dt, X_train_grp, X_train_delta, X_train_text])
    X_test_final = np.hstack([X_test_num, X_test_type, X_test_dist, X_test_sub, X_test_dt, X_test_grp, X_test_delta, X_test_text])
    
    print(f"Final Feature Shape: {X_train_final.shape}")
    
    # 5. Hyperparameter Optimization
    print("\n--- Starting Hyperparameter Optimization (RandomizedSearch) ---")
    param_dist = {
        'n_estimators': [500, 800, 1000],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [6, 9, 12],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'subsample': [0.6, 0.7, 0.8],
        'min_child_weight': [3, 5, 10]
    }
    
    xgb_base = xgb.XGBRegressor(objective='reg:absoluteerror', n_jobs=-1, random_state=42)
    
    # Use a smaller subsample for search to be faster, or full? 
    # Let's use full but few iterations (e.g., 5) to fit in reasonable time
    random_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_dist,
        n_iter=5, # Try 5 combinations
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    # Fit search (Pass sample weights!)
    random_search.fit(X_train_final, y_train, sample_weight=w_train)
    
    best_params = random_search.best_params_
    print(f"Best Parameters: {best_params}")
    
    # 6. Train Final Model
    print("Training Final Model with Best Params...")
    final_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_jobs=-1,
        random_state=42,
        **best_params
    )
    final_model.fit(X_train_final, y_train, sample_weight=w_train)
    
    # 7. Evaluate
    print("Evaluating...")
    y_pred_log = final_model.predict(X_test_final)
    y_pred = np.expm1(y_pred_log)
    y_test_days = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test_days, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_days, y_pred))
    r2 = r2_score(y_test_days, y_pred)
    
    print(f"\n--- Optimized Model Performance ---")
    print(f"MAE:  {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R2:   {r2:.4f}")
    
    # 8. Save Artifacts & History
    artifacts = {
        'model': final_model,
        'text_pipeline': text_pipeline,
        'te_type': te_type,
        'te_dist': te_dist,
        'te_sub': te_sub,
        'te_dist_type': te_dist_type,
        'te_group': te_group,
        'feature_names_num': num_cols,
        'best_params': best_params
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_filename = f"resolution_model_opt_{timestamp}.joblib"
    version_path = os.path.join(MODEL_DIR, version_filename)
    latest_path = os.path.join(MODEL_DIR, "resolution_model_robust.joblib") # Overwrite robust as latest
    
    joblib.dump(artifacts, version_path)
    print(f"Saved version: {version_path}")
    
    shutil.copy(version_path, latest_path)
    print(f"Updated latest: {latest_path}")
    
    # Log History
    history_path = os.path.join(MODEL_DIR, "model_history.csv")
    history_entry = {
        'timestamp': timestamp,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'filename': version_filename,
        'description': f'Optimized Full Data (params={best_params})'
    }
    
    history_df = pd.DataFrame([history_entry])
    if os.path.exists(history_path):
        history_df.to_csv(history_path, mode='a', header=False, index=False)
    else:
        history_df.to_csv(history_path, mode='w', header=True, index=False)
    print(f"Logged history.")

if __name__ == "__main__":
    df = load_data()
    train_optimized_model(df)
