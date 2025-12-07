# PRD: Traffy Fondue Resolution Time Prediction System

## 1. Executive Summary

### 1.1 Product Vision
Build a **Resolution Time Prediction System** for Bangkok Metropolitan Administration (BMA) that predicts how many days a citizen complaint will take to resolve.

### 1.2 Business Problem
- Citizens don't know how long their complaints will take
- Government can't identify slow-performing districts
- No data-driven resource allocation for complaint handling

### 1.3 Solution
An end-to-end data science pipeline that:
1. Predicts resolution time for new complaints
2. Visualizes district performance geographically
3. Identifies key factors affecting resolution speed

---

## 2. Goals & Success Metrics

### 2.1 Primary Goals
| Goal | Description |
|------|-------------|
| **G1** | Predict resolution time with reasonable accuracy |
| **G2** | Identify which factors affect resolution speed |
| **G3** | Visualize performance across Bangkok districts |
| **G4** | Provide interactive prediction demo |

### 2.2 Success Criteria
| Metric | Target | Priority |
|--------|--------|----------|
| Model MAE | < 30 days | HIGH |
| Model R² | > 0.2 | MEDIUM |
| External Data | ≥ 1,000 records | REQUIRED |
| Geospatial Map | Working interactive | REQUIRED |
| Demo App | Functional prediction | REQUIRED |

---

## 3. Data Requirements

### 3.1 Primary Dataset: Traffy Fondue
| Attribute | Value |
|-----------|-------|
| **Location** | `/home/CHAIN/project/temp/dsde/bangkok_traffy.csv` |
| **Records** | 787,026 complaints |
| **Period** | August 2021 - January 2025 |
| **Coverage** | Bangkok & Greater Metropolitan Area |

#### Key Columns
| Column | Type | Use |
|--------|------|-----|
| `type` | `{cat1,cat2}` | Complaint category (multi-label) |
| `district` | string | Geographic location |
| `timestamp` | datetime | When complaint was filed |
| `last_activity` | datetime | When resolved (for target calculation) |
| `state` | Thai string | Status: เสร็จสิ้น = completed |
| `coords` | `lon,lat` | Geographic coordinates |
| `comment` | Thai text | Complaint description |

#### Target Variable
```
resolution_time_days = (last_activity - timestamp).days
```

### 3.2 External Data: Weather (To Scrape)
| Attribute | Value |
|-----------|-------|
| **Source** | Open-Meteo API (free, no API key) |
| **Records Needed** | 1,200+ daily records |
| **Period** | 2021-09-01 to 2025-01-31 |
| **Location** | Bangkok (13.75°N, 100.50°E) |

#### Required Fields
- Date
- Temperature (max/min)
- Precipitation/Rainfall (mm)
- Weather code

---

## 4. Functional Requirements

### FR1: Data Engineering Pipeline
- Ingest 787K records from CSV using Apache Spark
- Parse coordinates and categories
- Calculate resolution time
- Join with weather data
- Export to Parquet

### FR2: Machine Learning Model
- XGBoost/Random Forest Regression
- Predict `resolution_time_days`
- Feature importance analysis

### FR3: Geospatial Visualization
- Folium interactive map
- District performance color-coded

### FR4: Prediction Demo
- Streamlit web application
- Interactive prediction interface

---

## 5. Technology Stack

| Component | Technology |
|-----------|------------|
| Data Processing | Apache Spark (PySpark) |
| ML Framework | scikit-learn, XGBoost |
| Thai NLP | pythainlp |
| Visualization | Folium, Plotly |
| Web App | Streamlit |
| Data Format | Parquet |

---

## 6. Project Structure

```
/home/CHAIN/project/temp/dsde/
├── bangkok_traffy.csv          # [EXISTS] Raw data
├── plan/                       # [EXISTS] This folder
│   ├── PRD.md                  # This document
│   └── STORIES.md              # User stories
├── data/
│   ├── raw/
│   │   └── weather_bangkok.csv # Scraped weather
│   ├── processed/
│   │   └── traffy_features.parquet
│   └── models/
│       ├── resolution_model.joblib
│       └── feature_importance.csv
├── scraping/
│   └── weather_scraper.py
├── spark_pipeline/
│   ├── config.py
│   ├── feature_engineering.py
│   └── main.py
├── ml/
│   └── train.py
├── visualization/
│   ├── resolution_map.py
│   └── resolution_time_map.html
└── app/
    └── streamlit_app.py
```

---

## 7. Course Requirements Mapping

| Course Requirement | Our Implementation |
|--------------------|-------------------|
| AI/ML Component | XGBoost Resolution Time Prediction |
| Data Engineering | Apache Spark Pipeline |
| Visualization (Geospatial) | Folium District Performance Map |
| External Data (1,000+ records) | Weather API Scraping |
| Traffy Data (100K+ records) | Using 787K records |

---

## 8. Glossary

| Thai Term | English | Meaning |
|-----------|---------|---------|
| เสร็จสิ้น | completed | Ticket resolved |
| กำลังดำเนินการ | in_progress | Being worked on |
| รอรับเรื่อง | pending | Waiting for assignment |
| ถนน | Road | Road-related complaints |
| ทางเท้า | Sidewalk | Sidewalk issues |
| น้ำท่วม | Flooding | Flood complaints |
| แสงสว่าง | Lighting | Street lighting |
| ความสะอาด | Cleanliness | Sanitation issues |
| กีดขวาง | Obstruction | Blocking/obstruction |
| ท่อระบายน้ำ | Drainage | Drainage system |

---

*Document Type: Product Requirements Document (PRD)*
*Related: See STORIES.md for implementation stories*
