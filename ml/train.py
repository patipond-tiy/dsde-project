import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    return df

def preprocess_data(df):
    print("Preprocessing data for ML...")
    
    # Clean 'type' column: remove {}, split by comma, take first
    # Example: "{ความสะอาด}" -> "ความสะอาด"
    # Example: "{น้ำท่วม,ร้องเรียน}" -> "น้ำท่วม"
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    
    # Select features
    feature_cols = [
        'type_clean', 'district', 
        'hour', 'day_of_week', 'month', 'is_weekend', 
        'comment_length',
        'rainfall_mm', 'temperature_max', 'temperature_min'
    ]
    target_col = 'resolution_time_days'
    
    # Drop rows with nulls in key columns
    df = df.dropna(subset=feature_cols + [target_col])
    
    # Filter outliers: Resolution time > 365 days
    print(f"Original row count: {len(df)}")
    df = df[df[target_col] <= 365]
    print(f"Filtered row count (< 365 days): {len(df)}")
    
    X = df[feature_cols].copy()
    # Log transform the target to handle skewness
    y = np.log1p(df[target_col].copy())
    
    # Encode Categoricals
    # We will use LabelEncoding for simplicity in this baseline, 
    # but OneHot is often better for linear models. XGBoost handles numeric labels fine.
    # However, saving the encoders is crucial for inference.
    
    encoders = {}
    for col in ['type_clean', 'district']:
        le = LabelEncoder()
        # Handle potential non-string types by converting to str
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    return X, y, encoders

def train_model(X, y):
    print("Splitting data...")
    # Time-based split is often better for this data, but random split is standard for MVP
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training XGBoost Regressor on {len(X_train)} samples...")
    model = xgb.XGBRegressor(
        objective='reg:absoluteerror', # Optimize for MAE since outliers are common
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred_log = model.predict(X_test)
    
    # Inverse transform (Log -> Days)
    y_pred = np.expm1(y_pred_log)
    y_test_days = np.expm1(y_test)
    
    # Metrics
    mae = mean_absolute_error(y_test_days, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_days, y_pred))
    r2 = r2_score(y_test_days, y_pred)
    
    print(f"\n--- Model Performance ---")
    print(f"MAE:  {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R2:   {r2:.4f}")
    
    return mae, rmse, r2

def save_artifacts(model, encoders, metrics):
    print(f"Saving artifacts to {MODEL_DIR}...")
    joblib.dump(model, os.path.join(MODEL_DIR, "resolution_model.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    
    # Save Feature Importance
    importance = model.feature_importances_
    feature_names = model.feature_names_in_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    fi_df = fi_df.sort_values(by='importance', ascending=False)
    fi_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    
    print("Top 5 Features:")
    print(fi_df.head(5))

def main():
    df = load_data()
    X, y, encoders = preprocess_data(df)
    model, X_test, y_test = train_model(X, y)
    mae, rmse, r2 = evaluate_model(model, X_test, y_test)
    save_artifacts(model, encoders, {'mae': mae, 'rmse': rmse, 'r2': r2})
    print("\nTraining Complete.")

if __name__ == "__main__":
    main()
