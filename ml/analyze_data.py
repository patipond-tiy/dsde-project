import pandas as pd
import os
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")

def analyze_short_tickets():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    
    # Clean 'type'
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    
    total = len(df)
    print(f"Total Records: {total}")
    
    # Check thresholds
    under_1_hour = len(df[df['resolution_time_days'] < (1/24)])
    under_1_day = len(df[df['resolution_time_days'] < 1])
    
    print(f"Tickets < 1 Hour: {under_1_hour} ({under_1_hour/total*100:.2f}%)")
    print(f"Tickets < 1 Day:  {under_1_day} ({under_1_day/total*100:.2f}%)")
    
    # Specific Check: Bang Rak + Bridge
    subset = df[(df['district'] == "บางรัก") & (df['type_clean'] == "สะพาน")]
    print(f"\n--- Bang Rak - Bridge Stats ---")
    print(f"Total: {len(subset)}")
    print(f"Min: {subset['resolution_time_days'].min():.4f}")
    print(f"Max: {subset['resolution_time_days'].max():.4f}")
    print(f"Median: {subset['resolution_time_days'].median():.4f}")
    print(f"Mean: {subset['resolution_time_days'].mean():.4f}")
    
    # How many are short?
    short_subset = subset[subset['resolution_time_days'] < 1]
    print(f"Short tickets (<1 day): {len(short_subset)}")
    if len(short_subset) > 0:
        print(short_subset[['resolution_time_days', 'comment']].head(10))

if __name__ == "__main__":
    analyze_short_tickets()
