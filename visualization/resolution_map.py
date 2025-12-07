import pandas as pd
import folium
import json
import os
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")
GEOJSON_PATH = os.path.join(BASE_DIR, "data", "raw", "thailand_districts.geojson")
OUTPUT_HTML = os.path.join(BASE_DIR, "visualization", "resolution_time_map.html")

def create_map():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    
    # Group by district
    # Filter out null districts and outliers for visualization clarity
    df_clean = df.dropna(subset=['district'])
    df_clean = df_clean[df_clean['resolution_time_days'] < 365] # Visualization focus on normal cases
    
    district_stats = df_clean.groupby('district')['resolution_time_days'].mean().reset_index()
    district_stats.columns = ['district', 'avg_days']
    
    print(f"Aggregated stats for {len(district_stats)} districts.")
    
    # Load GeoJSON
    print("Loading GeoJSON...")
    with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
        geo_data = json.load(f)
        
    # Filter for Bangkok (pro_code = '10' or pro_en = 'Bangkok')
    bangkok_features = [
        f for f in geo_data['features'] 
        if f['properties'].get('pro_en') == 'Bangkok' or f['properties'].get('pro_code') == '10'
    ]
    
    bangkok_geojson = {
        "type": "FeatureCollection",
        "features": bangkok_features
    }
    
    print(f"Found {len(bangkok_features)} Bangkok districts in GeoJSON.")
    
    # Create Map
    m = folium.Map(location=[13.7563, 100.5018], zoom_start=11)
    
    # Choropleth
    folium.Choropleth(
        geo_data=bangkok_geojson,
        name='Resolution Time',
        data=district_stats,
        columns=['district', 'avg_days'],
        key_on='feature.properties.amp_th',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Avg Resolution Time (Days)',
        nan_fill_color='white'
    ).add_to(m)
    
    # Save
    m.save(OUTPUT_HTML)
    print(f"Map saved to {OUTPUT_HTML}")

if __name__ == "__main__":
    create_map()
