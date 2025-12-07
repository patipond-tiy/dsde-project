import joblib
import os
from pythainlp.tokenize import word_tokenize

def thai_tokenizer(text):
    if not isinstance(text, str):
        return []
    return word_tokenize(text, engine="newmm")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "resolution_model_robust.joblib")

try:
    print(f"Loading {MODEL_PATH}...")
    artifacts = joblib.load(MODEL_PATH)
    print("Keys found:", list(artifacts.keys()))
    
    required_keys = ['te_type', 'te_dist', 'te_sub', 'te_dist_type', 'te_group', 'feature_names_num']
    missing = [k for k in required_keys if k not in artifacts]
    
    if missing:
        print(f"❌ MISSING KEYS: {missing}")
    else:
        print("✅ All required keys are present.")
        
except Exception as e:
    print(f"Error loading model: {e}")
