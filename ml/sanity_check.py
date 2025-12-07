import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")

def check_stats():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    
    # Clean type
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    
    # Filter
    district = "บางรัก"
    ctype = "สะพาน"
    
    subset = df[(df['district'] == district) & (df['type_clean'] == ctype)]
    
    print(f"--- Stats for {district} - {ctype} ---")
    print(f"Total Records: {len(subset)}")
    if len(subset) > 0:
        print(f"Mean Resolution Time: {subset['resolution_time_days'].mean():.2f} days")
        print(f"Median Resolution Time: {subset['resolution_time_days'].median():.2f} days")
        print(f"Min: {subset['resolution_time_days'].min():.2f}")
        print(f"Max: {subset['resolution_time_days'].max():.2f}")
    else:
        print("No records found for this combination.")

    print("\n--- Global Stats for 'Bridge' (สะพาน) ---")
    global_subset = df[df['type_clean'] == ctype]
    print(f"Global Mean: {global_subset['resolution_time_days'].mean():.2f} days")
    print(f"Global Median: {global_subset['resolution_time_days'].median():.2f} days")

if __name__ == "__main__":
    check_stats()
