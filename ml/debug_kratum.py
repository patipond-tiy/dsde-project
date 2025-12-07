import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")

def check_kratum_baen():
    df = pd.read_parquet(DATA_PATH)
    # Clean type
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    
    # Filter for Kratum Baen
    kb_df = df[df['district'] == 'กระทุ่มแบน']
    print(f"Total tickets in Kratum Baen: {len(kb_df)}")
    
    # Filter for Bridge
    bridge_kb = kb_df[kb_df['type_clean'] == 'สะพาน']
    print(f"\nStats for 'สะพาน' in 'กระทุ่มแบน':")
    if not bridge_kb.empty:
        print(bridge_kb['resolution_time_days'].describe())
        print("\nSample records:")
        print(bridge_kb[['comment', 'resolution_time_days']].head(5))
    else:
        print("No Bridge tickets in Kratum Baen.")

if __name__ == "__main__":
    check_kratum_baen()

