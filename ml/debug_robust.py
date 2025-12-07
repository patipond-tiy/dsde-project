import pandas as pd
import numpy as np
import joblib
import os
from pythainlp.tokenize import word_tokenize

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "resolution_model_robust.joblib")

def thai_tokenizer(text):
    if not isinstance(text, str):
        return []
    return word_tokenize(text, engine="newmm")

def debug_prediction():
    print("Loading artifacts from robust model...")
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts['model']
    
    # Encoders
    te_type = artifacts['te_type']
    te_dist = artifacts['te_dist']
    te_dist_type = artifacts['te_dist_type']
    te_group = artifacts['te_group']
    text_pipeline = artifacts['text_pipeline']
    
    # User Input
    selected_district = "บางรัก"
    selected_type = "สะพาน"
    comment_text = "น้ำท่วมขัง เดินทางลำบาก"
    
    # Derived Features
    district_type = f"{selected_district}_{selected_type}"
    infra_keywords = ['ถนน', 'สะพาน', 'ท่อระบายน้ำ', 'ก่อสร้าง', 'คลอง', 'ทางเท้า', 'แสงสว่าง']
    category_group = 'Infrastructure' if selected_type in infra_keywords else 'Service'
    
    print(f"\n--- Inputs ---")
    print(f"District: {selected_district}")
    print(f"Type: {selected_type}")
    print(f"Interaction: {district_type}")
    print(f"Group: {category_group}")

    # Create DF
    input_df = pd.DataFrame({
        'type_clean': [selected_type],
        'district': [selected_district],
        'district_type': [district_type],
        'category_group': [category_group],
        'comment': [comment_text]
    })
    
    # 1. Check Encodings
    print(f"\n--- Encoding Check ---")
    try:
        enc_type = te_type.transform(input_df[['type_clean']])[0][0]
        enc_dist = te_dist.transform(input_df[['district']])[0][0]
        enc_dt = te_dist_type.transform(input_df[['district_type']])[0][0]
        enc_grp = te_group.transform(input_df[['category_group']])[0][0]
        
        print(f"Type ({selected_type}): {np.expm1(enc_type):.2f} days")
        print(f"District ({selected_district}): {np.expm1(enc_dist):.2f} days")
        print(f"Interaction ({district_type}): {np.expm1(enc_dt):.2f} days")
        print(f"Group ({category_group}): {np.expm1(enc_grp):.2f} days")
    except Exception as e:
        print(f"Encoding Error: {e}")
        
    # 2. Check Text
    vec_text = text_pipeline.transform(input_df['comment'])
    
    # 3. Predict
    # Construct numericals (dummy values for temporal/weather)
    # num_cols = ['hour_sin', 'hour_cos', 'day_of_week', 'month_sin', 'month_cos', 
    #             'is_weekend', 'comment_length', 'rainfall_mm', 'temperature_max', 'temperature_min']
    num_vals = np.array([[0, 1, 0, 0, 1, 0, len(comment_text), 0, 32, 25]])
    
    # Reshape categorical arrays to ensure 2D (1, 1)
    X_type = te_type.transform(input_df[['type_clean']])
    X_dist = te_dist.transform(input_df[['district']])
    X_dt = te_dist_type.transform(input_df[['district_type']])
    X_grp = te_group.transform(input_df[['category_group']])

    # Stack: num, type, dist, dt, grp, text
    X_final = np.hstack([num_vals, X_type, X_dist, X_dt, X_grp, vec_text])
    
    pred_log = model.predict(X_final)[0]
    print(f"\n--- Final Prediction ---")
    print(f"Log: {pred_log:.4f}")
    print(f"Days: {np.expm1(pred_log):.2f}")

if __name__ == "__main__":
    debug_prediction()
