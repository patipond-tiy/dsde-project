# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data science project analyzing Bangkok's Traffy Fondue citizen complaint dataset (Aug 2021 - Jan 2025). The end-to-end pipeline includes:
- **Data Engineering**: PySpark pipeline for data processing
- **ML Component**: XGBoost model predicting complaint resolution time
- **Visualization**: Streamlit app with Folium choropleth maps

## Common Commands

### Run the Streamlit App
```bash
source venv/bin/activate
streamlit run app/streamlit_app.py
```

### Run Spark Data Pipeline
```bash
source venv/bin/activate
python spark_pipeline/main.py
```

### Train ML Model
```bash
source venv/bin/activate
python ml/train_robust.py
```

### Scrape Weather Data
```bash
python scraping/weather_scraper.py
```

## Architecture

### Data Flow
```
bangkok_traffy.csv (raw 1GB+)
        ↓
spark_pipeline/main.py (PySpark processing)
        ↓
data/processed/traffy_features.parquet
        ↓
ml/train_robust.py (XGBoost training)
        ↓
data/models/resolution_model_robust.joblib
        ↓
app/streamlit_app.py (prediction UI + maps)
```

### Key Components

**spark_pipeline/**
- `config.py`: Paths and Spark settings (SPARK_MASTER="local[*]")
- `feature_engineering.py`: PySpark transformations - cleans data, extracts temporal features (hour, day_of_week, is_rainy_season), calculates resolution_time_days
- `main.py`: Orchestrates pipeline, outputs parquet

**ml/**
- `train_robust.py`: Main training script. Uses Target Encoding for categorical features, TF-IDF+SVD for Thai text (pythainlp tokenizer), sample weighting for class balance
- Model artifacts saved as joblib dict containing: model, text_pipeline, target encoders (te_type, te_dist, te_sub, te_dist_type, te_group)

**app/streamlit_app.py**
- Loads model artifacts and applies same transforms for prediction
- Folium choropleth maps using `data/raw/thailand_districts.geojson`
- Filters GeoJSON for Bangkok districts only (pro_en='Bangkok' or pro_code='10')

### Data Files
- `bangkok_traffy.csv`: Raw complaints data (~700k+ records)
- `data/raw/weather_bangkok.csv`: Scraped from Open-Meteo API
- `data/raw/thailand_districts.geojson`: District boundaries for maps
- `data/processed/traffy_features.parquet`: Spark output with engineered features

### Thai Language Processing
Uses `pythainlp.tokenize.word_tokenize` with "newmm" engine for Thai text tokenization in TF-IDF. The `thai_tokenizer` function must be defined in any module loading the model (required for joblib unpickling).

### Target Variable
- `resolution_time_days`: Time between ticket creation (timestamp) and last_activity
- Filtered to completed tickets only (state = 'เสร็จสิ้น')
- Model predicts log1p-transformed values, output is expm1 to get days
