import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pythainlp.tokenize import word_tokenize
from tqdm import tqdm
from resampling_utils import hybrid_resample_regression
from evaluation_utils import evaluate_by_bins, plot_predictions_by_range, calculate_balance_score

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

def train_robust_model(df):
    print("Preprocessing data (Deep Thought Mode)...")
    
    # 1. Basic Cleaning
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    df['district_type'] = df['district'].astype(str) + "_" + df['type_clean'].astype(str)
    
    # 2. Add Grouping (Infra vs Service) - The "Meta-Feature"
    # This helps separate long-term construction from quick fixes.
    infra_keywords = ['ถนน', 'สะพาน', 'ท่อระบายน้ำ', 'ก่อสร้าง', 'คลอง', 'ทางเท้า', 'แสงสว่าง']
    df['category_group'] = df['type_clean'].apply(lambda x: 'Infrastructure' if x in infra_keywords else 'Service')

    # 3. Cyclical Features (Removed low-correlation features: hour, day_of_week, etc.)
    # Based on correlation analysis, only seasonality (is_rainy_season) and core categorical/text features matter.
    
    # 4. Strict Filtering (Remove Administrative Noise)
    # Tickets < 1 hour (0.04 days) are likely noise/duplicates.
    print(f"Original Count: {len(df)}")
    df = df[df['resolution_time_days'] > 0.04] 
    print(f"Filtered Count (>1hr): {len(df)}")
    
    feature_cols = [
        'type_clean', 'district', 'subdistrict', 'district_type', 'category_group', 'comment',
        'is_rainy_season'
    ]
    target_col = 'resolution_time_days'
    
    df = df.dropna(subset=feature_cols + [target_col])
    
    # Use reasonable sample size for training speed
    # We prioritize recent data as operations change over years.
    df = df.sort_values('timestamp', ascending=False).head(250000)
    
    df['target_capped'] = df[target_col].clip(upper=365)

    # 5. Apply Hybrid Resampling (Undersample + Oversample)
    # Address target imbalance by balancing the distribution across time ranges
    # V2: More aggressive oversampling for 90+ days, less aggressive undersampling
    bins = [0, 7, 30, 90, 180, 365]
    labels = ['0-7d', '7-30d', '30-90d', '90-180d', '180-365d']
    target_distribution = {
        '0-7d': 100000,    # 33% - Keep more low-value patterns
        '7-30d': 55000,    # 18% - Keep almost all
        '30-90d': 50000,   # 17% - Moderate oversample
        '90-180d': 50000,  # 17% - AGGRESSIVE oversample (259% increase!)
        '180-365d': 45000  # 15% - VERY AGGRESSIVE oversample (342% increase!)
    }

    df_balanced = hybrid_resample_regression(
        df,
        target_col='target_capped',
        bins=bins,
        target_distribution=target_distribution,
        random_state=42
    )

    # 5b. Add Sample Weights on Top of Resampling
    # Even after resampling, give extra weight to high-value cases during training
    print("Assigning sample weights...")
    bin_weights = {
        '0-7d': 0.5,       # Lower weight for over-represented class
        '7-30d': 0.8,
        '30-90d': 1.0,     # Baseline weight
        '90-180d': 1.5,    # 50% extra weight
        '180-365d': 2.0    # 2x weight for highest-value cases
    }
    df_balanced['sample_weight'] = df_balanced['bin'].map(bin_weights)

    # Update y and X from resampled dataframe
    y = np.log1p(df_balanced['target_capped'])
    X = df_balanced[feature_cols]

    # 6. Split Data with Stratification
    # We need to split the LINEAR target as well for proper encoding
    y_lin = df_balanced['target_capped']

    # Stratified split ensures test set has similar distribution to training set
    X_train, X_test, y_train, y_test, y_train_lin, y_test_lin, w_train, w_test = train_test_split(
        X, y, y_lin, df_balanced['sample_weight'], test_size=0.2, random_state=42, stratify=df_balanced['bin']
    )
    
    print("Fitting encoders and transformers...")
    
    # A. Text Pipeline (TF-IDF + SVD)
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=thai_tokenizer, max_features=3000, min_df=10, max_df=0.8)),
        ('svd', TruncatedSVD(n_components=15, random_state=42))
    ])
    
    X_train_text = text_pipeline.fit_transform(X_train['comment'])
    X_test_text = text_pipeline.transform(X_test['comment'])
    
    # B. Target Encoding (Leak-Proof)
    # CRITICAL FIX: Use Linear Target (y_train_lin) to capture Real Mean, not Log Mean
    te_type = TargetEncoder(smooth="auto")
    te_dist = TargetEncoder(smooth="auto")
    te_sub = TargetEncoder(smooth="auto") # NEW
    te_dist_type = TargetEncoder(smooth="auto")
    te_group = TargetEncoder(smooth="auto")
    
    X_train_type = te_type.fit_transform(X_train[['type_clean']], y_train_lin)
    X_test_type = te_type.transform(X_test[['type_clean']])
    
    X_train_dist = te_dist.fit_transform(X_train[['district']], y_train_lin)
    X_test_dist = te_dist.transform(X_test[['district']])

    X_train_sub = te_sub.fit_transform(X_train[['subdistrict']], y_train_lin) # NEW
    X_test_sub = te_sub.transform(X_test[['subdistrict']]) # NEW

    X_train_dt = te_dist_type.fit_transform(X_train[['district_type']], y_train_lin)
    X_test_dt = te_dist_type.transform(X_test[['district_type']])
    
    X_train_grp = te_group.fit_transform(X_train[['category_group']], y_train_lin)
    X_test_grp = te_group.transform(X_test[['category_group']])
    
    # --- NEW: Delta Feature ---
    # This captures how much this specific case deviates from the general category average.
    # High +ve means "Much slower than usual for this type".
    X_train_delta = X_train_dt - X_train_type
    X_test_delta = X_test_dt - X_test_type
    
    # C. Numerical Features
    num_cols = ['is_rainy_season']
    
    X_train_num = X_train[num_cols].values
    X_test_num = X_test[num_cols].values
    
    # Combine
    # Added X_train_sub to the stack
    X_train_final = np.hstack([X_train_num, X_train_type, X_train_dist, X_train_sub, X_train_dt, X_train_grp, X_train_delta, X_train_text])
    X_test_final = np.hstack([X_test_num, X_test_type, X_test_dist, X_test_sub, X_test_dt, X_test_grp, X_test_delta, X_test_text])
    
    print(f"Final Feature Shape: {X_train_final.shape}")
    
    # 7. Train Model with Sample Weights and Progress Monitoring
    print("Training XGBoost with sample weights and early stopping...")
    print("Progress will be shown every 10 iterations...\n")

    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=9,
        colsample_bytree=0.7,
        subsample=0.7,
        min_child_weight=5,
        objective='reg:absoluteerror',
        eval_metric='mae',
        early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
        n_jobs=-1,
        random_state=42
    )

    # Create validation set for early stopping and progress monitoring
    val_size = 5000
    X_val = X_train_final[:val_size]
    y_val = y_train[:val_size]
    X_train_sub = X_train_final[val_size:]
    y_train_sub = y_train[val_size:]
    w_train_sub = w_train.values[val_size:] if hasattr(w_train, 'values') else w_train[val_size:]

    print(f"Training samples: {len(X_train_sub):,} | Validation samples: {len(X_val):,}\n")

    # Fit with progress monitoring
    model.fit(
        X_train_sub,
        y_train_sub,
        sample_weight=w_train_sub,
        eval_set=[(X_val, y_val)],
        verbose=10  # Print every 10 iterations
    )
    
    # 8. Evaluate
    print("Evaluating...")
    y_pred_log = model.predict(X_test_final)
    y_pred = np.expm1(y_pred_log)
    y_test_days = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test_days, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_days, y_pred))
    r2 = r2_score(y_test_days, y_pred)
    
    print(f"\n--- Overall Model Performance ---")
    print(f"MAE:  {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R2:   {r2:.4f}")

    # Stratified Evaluation by Range
    print("\n--- Performance by Target Range ---")
    stratified_metrics = evaluate_by_bins(y_test_days, y_pred, bins, labels)

    for bin_label, metrics in stratified_metrics.items():
        if metrics['count'] > 0:
            print(f"{bin_label:12s} | MAE: {metrics['mae']:6.2f} | RMSE: {metrics['rmse']:6.2f} | "
                  f"R2: {metrics['r2']:6.4f} | n={metrics['count']:5d} | "
                  f"Mean True: {metrics['mean_true']:6.2f} | Mean Pred: {metrics['mean_pred']:6.2f}")

    # Calculate balance score
    balance_score = calculate_balance_score(stratified_metrics)
    print(f"\nBalance Score (CV of MAEs): {balance_score:.4f} (lower = more balanced)")

    # Extract high-value and low-value MAEs for history
    mae_high_value = stratified_metrics.get('180-365d', {}).get('mae', np.nan)
    mae_low_value = stratified_metrics.get('0-7d', {}).get('mae', np.nan)

    # 9. Save Artifacts
    print("Saving artifacts...")
    artifacts = {
        'model': model,
        'text_pipeline': text_pipeline,
        'te_type': te_type,
        'te_dist': te_dist,
        'te_sub': te_sub,
        'te_dist_type': te_dist_type,
        'te_group': te_group,
        'feature_names_num': num_cols
    }
    
    # Versioning
    from datetime import datetime
    import shutil
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_filename = f"resolution_model_{timestamp}.joblib"
    version_path = os.path.join(MODEL_DIR, version_filename)
    latest_path = os.path.join(MODEL_DIR, "resolution_model_robust.joblib")
    
    # Save Version
    joblib.dump(artifacts, version_path)
    print(f"Saved version: {version_path}")
    
    # Update Latest
    shutil.copy(version_path, latest_path)
    print(f"Updated latest: {latest_path}")
    
    # Log History
    history_path = os.path.join(MODEL_DIR, "model_history.csv")
    history_entry = {
        'timestamp': timestamp,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mae_high_value': mae_high_value,
        'mae_low_value': mae_low_value,
        'balance_score': balance_score,
        'filename': version_filename,
        'description': 'Hybrid Resampling V2 (33-18-17-17-15 distribution, aggressive 90+ oversample, 2% noise, sample weights)'
    }
    
    history_df = pd.DataFrame([history_entry])
    
    if os.path.exists(history_path):
        history_df.to_csv(history_path, mode='a', header=False, index=False)
    else:
        history_df.to_csv(history_path, mode='w', header=True, index=False)
    
    print(f"Logged history to {history_path}")

    # Save Enhanced Visualization
    plot_path = os.path.join(MODEL_DIR, f"prediction_scatter_{timestamp}.png")
    plot_predictions_by_range(
        y_test_days,
        y_pred,
        bins,
        labels,
        plot_path,
        title=f"Hybrid Resampling Model (MAE: {mae:.2f})"
    )

if __name__ == "__main__":
    df = load_data()
    train_robust_model(df)
