import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")

def check_stats():
    df = pd.read_parquet(DATA_PATH)
    # Clean type
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    
    # Filter for Bridge
    bridge_df = df[df['type_clean'] == 'สะพาน']
    print(f"Stats for 'สะพาน' (Bridge):")
    print(bridge_df['resolution_time_days'].describe())
    
    # Check Bang Rak Bridge
    br_bridge = bridge_df[bridge_df['district'] == 'บางรัก']
    if not br_bridge.empty:
        print("\nStats for 'สะพาน' in 'บางรัก':")
        print(br_bridge['resolution_time_days'].describe())
    else:
        print("\nNo records for Bridge in Bang Rak.")

if __name__ == "__main__":
    check_stats()

