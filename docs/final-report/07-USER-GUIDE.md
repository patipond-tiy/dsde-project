# User Guide & Deployment
## Step-by-Step Instructions for Running the Project

**Last Updated:** December 7, 2025
**System Requirements:** Python 3.8+, 4GB+ RAM, Linux/macOS/Windows

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Running the Streamlit App](#3-running-the-streamlit-app)
4. [Running the Spark Pipeline](#4-running-the-spark-pipeline)
5. [Training the ML Model](#5-training-the-ml-model)
6. [Scraping Weather Data](#6-scraping-weather-data)
7. [Troubleshooting](#7-troubleshooting)
8. [Project Structure](#8-project-structure)

---

## 1. PREREQUISITES

### System Requirements
- **Operating System:** Linux, macOS, or Windows 10/11
- **Python:** 3.8 or higher (3.10 recommended)
- **RAM:** Minimum 4GB, recommended 8GB+
- **Disk Space:** 5GB free space
- **CPU:** Multi-core processor (4+ cores recommended for Spark)

### Software Dependencies
```bash
# Check Python version
python --version  # Should be 3.8+

# Check pip
pip --version

# Optional: Java for Spark (bundled with PySpark, but explicit Java 8+ recommended)
java -version  # Should be 1.8+
```

---

## 2. INSTALLATION

### Step 1: Clone/Download the Project
```bash
cd /path/to/your/workspace
# If from git:
git clone <repository-url>
cd dsde

# Or extract from zip:
unzip dsde.zip
cd dsde
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show venv in prompt)
which python  # Should point to venv/bin/python
```

### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This installs:
# - pyspark==4.0.1
# - streamlit==1.52.1
# - folium==0.20.0
# - xgboost==3.1.2
# - scikit-learn==1.7.2
# - pythainlp==5.1.2
# - pandas==2.3.3
# - numpy==2.3.5
# - and many more...

# Installation time: 2-5 minutes depending on internet speed
```

### Step 4: Verify Installation
```bash
# Test imports
python -c "
import pyspark
import streamlit
import xgboost
import pythainlp
print('✅ All core packages installed successfully!')
"
```

---

## 3. RUNNING THE STREAMLIT APP

### Quick Start
```bash
# Ensure you're in project root with venv activated
source venv/bin/activate

# Run Streamlit app
streamlit run app/streamlit_app.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
# Network URL: http://192.168.x.x:8501
```

### Access the Application
1. Open your browser
2. Navigate to `http://localhost:8501`
3. The app should load with:
   - Sidebar for prediction inputs
   - Main area with tabs (Interactive Map, Analysis & Insights)

### Using the Prediction Interface

**Step 1: Select District**
- Use dropdown in sidebar
- Example: Select "หนองจอก" (Nong Chok)

**Step 2: Select Subdistrict**
- Dropdown updates automatically based on district
- Example: Select subdistrict within Nong Chok

**Step 3: Select Complaint Type**
- Choose from Thai categories
- Example: "ถนน" (Road), "สะพาน" (Bridge)

**Step 4: Adjust Month (Optional)**
- Use slider (1-12)
- Seasonal indicator updates automatically
  - Months 5-10: 🌧️ Rainy season
  - Months 11-4: ☀️ Dry season

**Step 5: Enter Complaint Description (Thai)**
- Type complaint in Thai language
- Example: "ถนนเสียหาย ต้องการซ่อมแซม" (Road damaged, needs repair)

**Step 6: Click "Predict Resolution Time"**
- Button in sidebar
- Shows estimated days with confidence indicator:
  - ✅ Quick (< 30 days)
  - ⚠️ Moderate (30-60 days)
  - ⚠️ Significant delay (> 60 days)

### Using the Interactive Map

**Tab 1: 🗺️ Interactive Map**
1. Select complaint types to filter (multiselect)
2. Choose metric:
   - Average Resolution Days (shows slow districts in red)
   - Number of Complaints (shows high-volume districts in blue)
3. Hover over districts to see names
4. Map updates automatically

### Viewing Analytics

**Tab 2: 📊 Analysis & Insights**
1. View top 10 feature importance (bar chart)
2. Scroll down to see sample data (first 100 rows)
3. Examine feature values and patterns

---

## 4. RUNNING THE SPARK PIPELINE

### When to Run
- When you have updated raw data (`bangkok_traffy.csv`)
- To regenerate processed features
- After modifying feature engineering logic

### Prerequisites
- Raw data file must exist: `/home/CHAIN/project/temp/dsde/bangkok_traffy.csv`
- Minimum 4GB free RAM

### Execution
```bash
# Ensure venv is activated
source venv/bin/activate

# Navigate to project root
cd /home/CHAIN/project/temp/dsde

# Run Spark pipeline
python spark_pipeline/main.py

# Expected output:
# Initializing Spark Session: TraffyFonduePrediction
# Loading Traffy data from .../bangkok_traffy.csv...
# Cleaning Traffy data...
# Extracting features...
# Saving processed data to .../traffy_features.parquet...
# +----------+------+-------+...
# |ticket_id | type |comment|...
# +----------+------+-------+...
# ✅ Pipeline completed successfully.
# Total processed records: 500000

# Processing time: 1-3 minutes
```

### Output Location
- **File:** `data/processed/traffy_features.parquet`
- **Format:** Parquet (Snappy compressed)
- **Size:** ~167.5 MB
- **Records:** ~500,000 (completed tickets only)

### Validation
```bash
# Verify output exists and check size
ls -lh data/processed/traffy_features.parquet/

# Quick inspection with pandas
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/traffy_features.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(df.head())
"
```

---

## 5. TRAINING THE ML MODEL

### When to Train
- After running Spark pipeline (to use latest features)
- To experiment with different approaches
- To update model with recent data

### Using the Production Model (Recommended)
```bash
# Ensure venv is activated
source venv/bin/activate

# Train the robust model (best performance)
python ml/train_robust.py

# Expected output:
# Loading data from parquet...
# Loaded 500000 records
# Filtering recent data (250000 records)...
# Creating bins for stratified resampling...
# Applying hybrid resampling strategy...
# Training XGBoost model...
# [0]   validation-mae:45.23
# [10]  validation-mae:38.12
# ...
# [500] validation-mae:34.47
#
# ✅ Training completed!
# Test MAE: 34.47 days
# Test RMSE: 57.55 days
# Test R²: 0.6132
#
# Model saved to: data/models/resolution_model_robust.joblib

# Training time: 5-10 minutes
```

### Trying Other Models
```bash
# Baseline model (simple, fast)
python ml/train.py

# Advanced model (with cyclical features)
python ml/train_advanced.py

# Optimized model (with hyperparameter tuning, slower)
python ml/train_optimized.py

# Final model (historical median encoding)
python ml/train_final.py
```

### Model Artifacts
After training, the following files are created:
- **Model:** `data/models/resolution_model_robust.joblib` (~12 MB)
- **History:** `data/models/model_history.csv` (training metrics log)
- **Feature Importance:** `data/models/feature_importance_optimized.csv`

---

## 6. SCRAPING WEATHER DATA

### When to Run
- To update weather data with latest records
- Initial setup (if weather CSV doesn't exist)

### Execution
```bash
# Ensure venv is activated
source venv/bin/activate

# Run weather scraper
python scraping/weather_scraper.py

# Expected output:
# ✅ Fetching weather data from Open-Meteo API...
# ✅ Retrieved 1280 daily records
# ✅ Saving to data/raw/weather_bangkok.csv
#
# Sample data:
#         date  weather_code  temperature_max  temperature_min  rainfall_mm
# 0 2021-08-01            63             31.9             25.4          3.2
# 1 2021-08-02            61             32.1             25.6          1.8
# ...
# ✅ Weather data saved successfully!

# Scraping time: 5-15 seconds
```

### Output Location
- **File:** `data/raw/weather_bangkok.csv`
- **Records:** 1,280+ (daily from Aug 2021 - Jan 2025)
- **Columns:** date, weather_code, temperature_max, temperature_min, rainfall_mm, temperature_avg

### Note
- Weather data is currently **staged but not actively used** in the ML model
- The `is_rainy_season` feature is hardcoded based on calendar months
- Future versions may integrate weather data via Spark pipeline join

---

## 7. TROUBLESHOOTING

### Common Issues & Solutions

#### Issue 1: "ModuleNotFoundError: No module named 'pyspark'"
**Cause:** Virtual environment not activated or dependencies not installed

**Solution:**
```bash
# Activate venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue 2: "FileNotFoundError: bangkok_traffy.csv"
**Cause:** Raw data file missing or wrong working directory

**Solution:**
```bash
# Verify you're in project root
pwd  # Should show /home/CHAIN/project/temp/dsde

# Check if file exists
ls -l bangkok_traffy.csv

# If missing, download or place the file in project root
```

#### Issue 3: "java.lang.OutOfMemoryError" (Spark)
**Cause:** Insufficient memory allocated to Spark driver

**Solution:**
Edit `spark_pipeline/main.py`:
```python
# Change from 4g to 8g or 12g
spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \  # Increase this
    .getOrCreate()
```

#### Issue 4: Streamlit app won't load / shows blank page
**Cause:** Port 8501 already in use or cache issues

**Solution:**
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Clear Streamlit cache
streamlit cache clear

# Run on different port
streamlit run app/streamlit_app.py --server.port 8502
```

#### Issue 5: "thai_tokenizer not defined" error when loading model
**Cause:** Model joblib file expects `thai_tokenizer` function in scope

**Solution:**
Ensure this function is defined before loading model:
```python
from pythainlp.tokenize import word_tokenize

def thai_tokenizer(text):
    if not isinstance(text, str):
        return []
    return word_tokenize(text, engine="newmm")

# Now load model
import joblib
model_artifacts = joblib.load('data/models/resolution_model_robust.joblib')
```

#### Issue 6: Map shows all white districts
**Cause:** District names don't match between data and GeoJSON

**Solution:**
Check district name spelling in parquet file:
```python
import pandas as pd
df = pd.read_parquet('data/processed/traffy_features.parquet')
print(df['district'].unique())

# Compare with GeoJSON district names (amp_th property)
```

#### Issue 7: Slow Spark processing (> 10 minutes)
**Cause:** Running on HDD instead of SSD, or insufficient CPU cores

**Solution:**
- Move data to SSD if possible
- Check CPU utilization (`top` or `htop`)
- Reduce dataset size for testing:
  ```python
  # In spark_pipeline/main.py, add after loading:
  df = df.limit(100000)  # Process only 100k records for testing
  ```

---

## 8. PROJECT STRUCTURE

```
dsde/
├── app/
│   └── streamlit_app.py                # Streamlit web application
├── spark_pipeline/
│   ├── config.py                       # Paths and Spark configuration
│   ├── feature_engineering.py          # Data transformation functions
│   └── main.py                         # Pipeline orchestration
├── ml/
│   ├── train_robust.py                 # Production ML training script
│   ├── train.py                        # Baseline model
│   ├── train_advanced.py               # Advanced features model
│   ├── train_optimized.py              # Hyperparameter tuning
│   ├── train_final.py                  # Historical median encoding
│   ├── evaluation_utils.py             # Metrics & plotting
│   └── resampling_utils.py             # Hybrid resampling functions
├── scraping/
│   └── weather_scraper.py              # Open-Meteo API scraper
├── data/
│   ├── raw/
│   │   ├── weather_bangkok.csv         # Scraped weather data
│   │   ├── thailand_districts.geojson  # District boundaries
│   │   └── bangkok_districts.geojson   # Bangkok-only boundaries
│   ├── processed/
│   │   └── traffy_features.parquet     # Spark pipeline output
│   └── models/
│       ├── resolution_model_robust.joblib       # Trained XGBoost model
│       ├── model_history.csv                    # Training metrics log
│       └── feature_importance_optimized.csv     # Feature rankings
├── docs/
│   ├── final-report/                   # Comprehensive documentation
│   │   ├── 00-TABLE-OF-CONTENTS.md
│   │   ├── 01-EXECUTIVE-SUMMARY.md
│   │   ├── 04-ML-EXPERIMENTS.md
│   │   └── 07-USER-GUIDE.md (this file)
│   └── temp-report/                    # Draft documentation
├── plan/
│   ├── COMPREHENSIVE_PRD.md            # Product requirements
│   └── USER_STORIES.md                 # User stories
├── bangkok_traffy.csv                  # Raw complaint data (1GB+)
├── requirements.txt                    # Python dependencies
├── CLAUDE.md                           # Project instructions for Claude Code
└── README.md                           # Quick start guide

Key Files:
- bangkok_traffy.csv: 787,026 raw complaint records (Aug 2021 - Jan 2025)
- data/processed/traffy_features.parquet: 500,000 processed records with 21 features
- data/models/resolution_model_robust.joblib: Production XGBoost model
- app/streamlit_app.py: Main web application (prediction + visualization)
```

---

## QUICK REFERENCE COMMANDS

### Complete Workflow (From Scratch)
```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Scrape weather data (optional, data already included)
python scraping/weather_scraper.py

# 3. Run Spark pipeline
python spark_pipeline/main.py

# 4. Train ML model
python ml/train_robust.py

# 5. Run Streamlit app
streamlit run app/streamlit_app.py

# Open browser: http://localhost:8501
```

### Daily Usage (Data Already Processed)
```bash
# Just run the app
source venv/bin/activate
streamlit run app/streamlit_app.py
```

### Update Data & Retrain
```bash
source venv/bin/activate

# 1. Update raw data (replace bangkok_traffy.csv with new data)
# 2. Reprocess features
python spark_pipeline/main.py

# 3. Retrain model
python ml/train_robust.py

# 4. Restart app (model artifacts auto-loaded)
streamlit run app/streamlit_app.py
```

---

## DEPLOYMENT OPTIONS

### Option 1: Local Development (Current Setup)
- Already configured
- Access via http://localhost:8501
- Suitable for: Development, testing, demos

### Option 2: Share on Network
```bash
# Run Streamlit with external access
streamlit run app/streamlit_app.py --server.address 0.0.0.0

# Access from other devices on same network:
# http://<your-ip>:8501

# Find your IP:
# Linux/Mac: ifconfig | grep inet
# Windows: ipconfig
```

### Option 3: Deploy to Streamlit Cloud (Free)
1. Push code to GitHub repository
2. Visit https://streamlit.io/cloud
3. Sign in with GitHub
4. Click "New app"
5. Select repository, branch, and file path: `app/streamlit_app.py`
6. Deploy (takes 2-3 minutes)
7. Access via public URL: `https://<app-name>.streamlit.app`

**Note:** Ensure `requirements.txt` includes all dependencies

### Option 4: Docker Container (Portable)
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0"]
```

```bash
# Build image
docker build -t traffy-predictor .

# Run container
docker run -p 8501:8501 traffy-predictor

# Access: http://localhost:8501
```

---

## SUPPORT & RESOURCES

### Documentation
- **Executive Summary:** `docs/final-report/01-EXECUTIVE-SUMMARY.md`
- **ML Experiments:** `docs/final-report/04-ML-EXPERIMENTS.md`
- **Technical Architecture:** `docs/final-report/02-TECHNICAL-ARCHITECTURE.md`

### External Resources
- **Streamlit Docs:** https://docs.streamlit.io/
- **PySpark Guide:** https://spark.apache.org/docs/latest/api/python/
- **XGBoost Tutorial:** https://xgboost.readthedocs.io/
- **PyThaiNLP:** https://pythainlp.github.io/

### Common Questions

**Q: Can I use this with non-Bangkok data?**
A: Yes, but you'll need to:
- Update GeoJSON file with relevant districts
- Adjust district filtering logic in Streamlit app
- Retrain model on new data

**Q: How often should I retrain the model?**
A: Recommended every 3-6 months as new data accumulates and operational patterns change

**Q: Can I add more features?**
A: Yes! Edit `spark_pipeline/feature_engineering.py` to add features, then rerun pipeline and training

**Q: Why is the model file so large (12MB)?**
A: Contains XGBoost model (600 trees), 5 Target Encoders, TF-IDF vocabulary (3000 words), and SVD components

---

## FINAL CHECKLIST

Before presenting or deploying, ensure:

- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Raw data file exists (`bangkok_traffy.csv`)
- [ ] Processed features exist (`data/processed/traffy_features.parquet`)
- [ ] Trained model exists (`data/models/resolution_model_robust.joblib`)
- [ ] Streamlit app runs without errors
- [ ] Map displays correctly with colored districts
- [ ] Prediction interface accepts inputs and returns estimates
- [ ] Weather data scraped and saved (`data/raw/weather_bangkok.csv`)
- [ ] All documentation files in `docs/final-report/` folder
- [ ] Code is commented and readable

---

**Document Version:** 1.0
**Last Updated:** December 7, 2025
**Maintained By:** DSDE Project Team
