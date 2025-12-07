import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def correlation_analysis():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    
    # 1. Clean Data
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    df['district_type'] = df['district'].astype(str) + "_" + df['type_clean'].astype(str)
    
    # Filter noise
    df = df[df['resolution_time_days'] > 0.04]
    
    # 2. Engineer "History-First" Features (for analysis purposes)
    print("Engineering historical features...")
    
    # Global Medians per Type and District (simulating what the model would see)
    type_median = df.groupby('type_clean')['resolution_time_days'].median()
    dist_median = df.groupby('district')['resolution_time_days'].median()
    dt_median = df.groupby('district_type')['resolution_time_days'].median()
    
    # Map them back
    df['hist_type_median'] = df['type_clean'].map(type_median)
    df['hist_dist_median'] = df['district'].map(dist_median)
    df['hist_interaction_median'] = df['district_type'].map(dt_median)
    
    # Fill interactions with type median if missing (though here we trained on all, so no missing)
    df['hist_interaction_median'] = df['hist_interaction_median'].fillna(df['hist_type_median'])
    
    # Log Target
    df['log_target'] = np.log1p(df['resolution_time_days'])
    df['log_hist_interaction'] = np.log1p(df['hist_interaction_median'])
    
    # 3. Select Numerical Features for Correlation
    num_cols = [
        'log_target', 
        'hist_interaction_median', 'log_hist_interaction',
        'hist_type_median', 'hist_dist_median',
        'comment_length', 
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 'is_rainy_season'
    ]
    
    corr_df = df[num_cols].dropna()
    
    # 4. Compute Correlation
    print("\n--- Correlation Matrix (Pearson) ---")
    corr_matrix = corr_df.corr()
    print(corr_matrix[['log_target']].sort_values(by='log_target', ascending=False))
    
    # 5. Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation with Resolution Time (Log)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
    print(f"\nSaved correlation matrix to {OUTPUT_DIR}/correlation_matrix.png")

if __name__ == "__main__":
    correlation_analysis()
