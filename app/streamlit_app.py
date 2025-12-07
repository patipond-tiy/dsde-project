import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import folium
import streamlit.components.v1 as components
import xgboost # Required for loading the model
from pythainlp.tokenize import word_tokenize

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "resolution_model_robust.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffy_features.parquet")
GEOJSON_PATH = os.path.join(BASE_DIR, "data", "raw", "thailand_districts.geojson")
FEATURE_IMP_PATH = os.path.join(BASE_DIR, "data", "models", "feature_importance_optimized.csv")

# Function required for unpickling the model pipeline
def thai_tokenizer(text):
    if not isinstance(text, str):
        return []
    return word_tokenize(text, engine="newmm")

# Page Config
st.set_page_config(page_title="Traffy Fondue Predictor", layout="wide")

@st.cache_resource
def load_model_resources_v5():
    # Load the dictionary containing model and transformers
    artifacts = joblib.load(MODEL_PATH)
    return artifacts

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    # Clean 'type' column
    df['type_clean'] = df['type'].astype(str).str.replace(r'[{}]', '', regex=True).str.split(',').str[0]
    # Filter reasonable values for visualization
    df = df[df['resolution_time_days'] < 365]
    return df

@st.cache_data
def load_geojson():
    with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
        geo_data = json.load(f)
    
    # Filter for Bangkok (pro_code = '10' or pro_en = 'Bangkok')
    bangkok_features = [
        f for f in geo_data['features'] 
        if f['properties'].get('pro_en') == 'Bangkok' or f['properties'].get('pro_code') == '10'
    ]
    
    return {
        "type": "FeatureCollection",
        "features": bangkok_features
    }

def main():
    st.title("🏙️ Traffy Fondue Resolution Time System")
    st.markdown("Predict resolution times and analyze district performance in Bangkok.")

    # Load Resources
    artifacts = load_model_resources_v5()
    model = artifacts['model']
    
    # Load Data for Dropdowns
    df = load_data()
    district_options = sorted(df['district'].dropna().unique())
    type_options = sorted(df['type_clean'].dropna().unique())
    district_counts = df['district'].value_counts()
    
    # Pre-compute subdistrict map
    subdistrict_map = {}
    for district, group in df.groupby('district'):
        # Get unique valid subdistricts
        subs = group['subdistrict'].dropna().unique()
        subdistrict_map[district] = sorted(subs)
    
    bangkok_geojson = load_geojson()
    
    # --- Sidebar ---
    st.sidebar.header("Prediction Inputs")
    
    selected_district = st.sidebar.selectbox("District", district_options)
    
    # Low Data Warning
    dist_n = district_counts.get(selected_district, 0)
    if dist_n < 50:
        st.sidebar.warning(f"⚠️ Limited data for **{selected_district}** ({dist_n} records). Prediction may be unreliable.")
    
    # Filter subdistricts based on district
    available_subdistricts = subdistrict_map.get(selected_district, [])
    selected_subdistrict = st.sidebar.selectbox("Subdistrict", available_subdistricts)
    
    selected_type = st.sidebar.selectbox("Complaint Type", type_options)
    
    # Temporal Inputs
    month = st.sidebar.slider("Month", 1, 12, 12)
    is_rainy_season = 1 if 5 <= month <= 10 else 0
    if is_rainy_season:
        st.sidebar.info("🌧️ Rainy Season (May - Oct)")
    else:
        st.sidebar.info("☀️ Dry Season (Nov - Apr)")
    
    # Text Input
    comment_text = st.sidebar.text_area("Complaint Description (Thai)", "น้ำท่วมขัง เดินทางลำบาก")
    
    # --- Prediction Logic ---
    if st.sidebar.button("Predict Resolution Time", type="primary"):
        # 1. Prepare Features
        district_type = f"{selected_district}_{selected_type}"
        
        # Meta-Feature logic
        infra_keywords = ['ถนน', 'สะพาน', 'ท่อระบายน้ำ', 'ก่อสร้าง', 'คลอง', 'ทางเท้า', 'แสงสว่าง']
        category_group = 'Infrastructure' if selected_type in infra_keywords else 'Service'

        # Create DataFrame for Transform
        input_df = pd.DataFrame({
            'type_clean': [selected_type],
            'district': [selected_district],
            'subdistrict': [selected_subdistrict],
            'district_type': [district_type],
            'category_group': [category_group],
            'comment': [comment_text],
            'is_rainy_season': [is_rainy_season]
        })
        
        # 2. Apply Transformers (Unpack from artifacts)
        text_pipeline = artifacts['text_pipeline']
        te_type = artifacts['te_type']
        te_dist = artifacts['te_dist']
        te_sub = artifacts['te_sub']
        te_dist_type = artifacts['te_dist_type']
        te_group = artifacts['te_group']
        
        # A. Text
        X_text = text_pipeline.transform(input_df['comment'])
        
        # B. Categorical (Target Encoded)
        X_type = te_type.transform(input_df[['type_clean']])
        X_dist = te_dist.transform(input_df[['district']])
        X_sub = te_sub.transform(input_df[['subdistrict']])
        X_dt = te_dist_type.transform(input_df[['district_type']])
        X_grp = te_group.transform(input_df[['category_group']])
        
        # --- NEW: Delta Feature ---
        X_delta = X_dt - X_type
        
        # C. Numerical
        num_cols = artifacts['feature_names_num']
        X_num = input_df[num_cols].values
        
        # D. Stack (Ensure order matches training!)
        # Train: num, type, dist, sub, dt, grp, delta, text
        X_final = np.hstack([X_num, X_type, X_dist, X_sub, X_dt, X_grp, X_delta, X_text])
        
        # 3. Predict
        pred_log = model.predict(X_final)[0]
        pred_days = np.expm1(pred_log)
        
        st.sidebar.success(f"⏱️ Estimated: **{pred_days:.1f} days**")
        
        if pred_days > 60:
             st.sidebar.error("⚠️ Significant delay expected.")
        elif pred_days > 30:
            st.sidebar.warning("⚠️ Moderate delay expected.")
        else:
            st.sidebar.info("✅ Quick resolution expected.")

    # --- Tabs ---
    tab1, tab2 = st.tabs(["🗺️ Interactive Map", "📊 Analysis & Insights"])
    
    # --- Tab 1: Interactive Map ---
    with tab1:
        st.subheader("Bangkok District Performance Map")
        
        # Map Filters
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            map_types = st.multiselect("Filter by Complaint Type", type_options, default=type_options[:3])
        with col_m2:
            map_metric = st.selectbox("Metric to Visualize", ["Average Resolution Days", "Number of Complaints"])

        # Filter Data for Map
        map_df = df.copy()
        if map_types:
            map_df = map_df[map_df['type_clean'].isin(map_types)]
            
        # Aggregate
        if map_metric == "Average Resolution Days":
            district_stats = map_df.groupby('district')['resolution_time_days'].mean().reset_index()
            district_stats.columns = ['district', 'value']
            legend_title = "Avg Days"
            fill_color = "YlOrRd"
        else:
            district_stats = map_df.groupby('district').size().reset_index()
            district_stats.columns = ['district', 'value']
            legend_title = "Complaint Count"
            fill_color = "PuBu"

        # Generate Map
        if not district_stats.empty:
            m = folium.Map(location=[13.7563, 100.5018], zoom_start=11, tiles="CartoDB positron")
            
            folium.Choropleth(
                geo_data=bangkok_geojson,
                name='choropleth',
                data=district_stats,
                columns=['district', 'value'],
                key_on='feature.properties.amp_th',
                fill_color=fill_color,
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=legend_title,
                nan_fill_color='white'
            ).add_to(m)
            
            # Add tooltips
            style_function = lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.1, 'weight': 0.1}
            highlight_function = lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1}
            folium.features.GeoJson(
                bangkok_geojson,
                style_function=style_function, 
                control=False,
                highlight_function=highlight_function, 
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['amp_th', 'amp_en'],
                    aliases=['District (TH):', 'District (EN):'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                )
            ).add_to(m)

            # Render
            components.html(m._repr_html_(), height=600)
        else:
            st.warning("No data available for the selected filters.")

    # --- Tab 2: Insights ---
    with tab2:
        st.subheader("Feature Importance")
        if os.path.exists(FEATURE_IMP_PATH):
            try:
                fi_df = pd.read_csv(FEATURE_IMP_PATH).sort_values(by='importance', ascending=False).head(10)
                st.bar_chart(fi_df.set_index('feature'))
                st.caption("These features have the highest impact on resolution time.")
            except:
                st.warning("Could not load feature importance.")
        
        st.divider()
        st.subheader("Dataset Overview")
        st.dataframe(df.head(100))

if __name__ == "__main__":
    main()