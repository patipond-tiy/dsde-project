import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")

def analyze_new_features():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter noise
    df = df[df['resolution_time_days'] > 0.04]
    df['log_target'] = np.log1p(df['resolution_time_days'])
    
    # 1. Organization Analysis
    print("\n--- Organization Analysis ---")
    n_orgs = df['organization'].nunique()
    print(f"Unique Organizations: {n_orgs}")
    
    # Calculate target mean per org
    org_stats = df.groupby('organization')['log_target'].mean()
    df['hist_org_mean'] = df['organization'].map(org_stats)
    
    # 2. Subdistrict Analysis
    print("\n--- Subdistrict Analysis ---")
    n_sub = df['subdistrict'].nunique()
    print(f"Unique Subdistricts: {n_sub}")
    
    # Calculate target mean per subdistrict
    sub_stats = df.groupby('subdistrict')['log_target'].mean()
    df['hist_sub_mean'] = df['subdistrict'].map(sub_stats)
    
    # 3. Workload Analysis (Simple approximation)
    # Count tickets in the same district in the last 7 days (rolling count)
    # Sort by time
    print("\n--- Workload Analysis (Calculating) ---")
    df = df.sort_values('timestamp')
    
    # We'll use a simple group rolling count. 
    # Since pandas rolling on groups can be slow with strings, we'll try a simpler approach or just use global daily count per district.
    # Let's count "Tickets per day per district" and join it back.
    
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby(['district', 'date']).size().reset_index(name='daily_tickets')
    
    # Calculate 7-day rolling sum per district
    daily_counts['rolling_workload_7d'] = daily_counts.groupby('district')['daily_tickets'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    
    # Shift it so we don't include "future" (though current day is known). 
    # Actually, for prediction at time T, we know the workload up to T. 
    # So rolling sum including today is fine, or shift by 1. Let's use shift 1 to be safe (yesterday's workload).
    daily_counts['prev_7d_workload'] = daily_counts.groupby('district')['rolling_workload_7d'].shift(1).fillna(0)
    
    # Join back
    df = df.merge(daily_counts[['district', 'date', 'prev_7d_workload']], on=['district', 'date'], how='left')
    
    # 4. Correlation
    cols_to_check = ['log_target', 'hist_org_mean', 'hist_sub_mean', 'prev_7d_workload']
    print("\n--- Correlation Matrix ---")
    corr = df[cols_to_check].corr()
    print(corr[['log_target']].sort_values(by='log_target', ascending=False))

if __name__ == "__main__":
    analyze_new_features()