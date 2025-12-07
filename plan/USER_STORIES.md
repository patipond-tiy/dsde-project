# User Stories & Acceptance Criteria
## Traffy Fondue Resolution Time Prediction System

**Epics:**
1.  **Data Acquisition & Engineering**
2.  **Machine Learning Modeling**
3.  **Visualization & Reporting**
4.  **Application Development**

---

## Epic 1: Data Acquisition & Engineering

### Story 1.1: Scrape Historical Weather Data
**As a** Data Engineer,
**I want to** scrape historical weather data for Bangkok (2021-2025),
**So that** I can correlate weather conditions with complaint resolution times.

**Acceptance Criteria:**
- [ ] Python script `scraping/weather_scraper.py` is created.
- [ ] Script fetches daily data: Max Temp, Min Temp, Rainfall (mm).
- [ ] Data covers the range: Aug 2021 to Jan 2025.
- [ ] Output is saved as `data/raw/weather_bangkok.csv` (or similar).
- [ ] Contains at least 1,200 records.

### Story 1.2: Ingest and Clean Traffy Data (Spark)
**As a** Data Engineer,
**I want to** build a Spark pipeline to ingest the 787K Traffy records,
**So that** I have a clean, structured dataset for analysis.

**Acceptance Criteria:**
- [ ] Spark script `spark_pipeline/main.py` (or individual stages) is functional.
- [ ] Reads `bangkok_traffy.csv`.
- [ ] Parses `timestamp` and `last_activity` columns correctly.
- [ ] Calculates `resolution_time_days`.
- [ ] Filters out invalid records (e.g., negative resolution time).

### Story 1.3: Feature Engineering & Merging
**As a** Data Scientist,
**I want to** create features and join weather data,
**So that** the model has predictive signals.

**Acceptance Criteria:**
- [ ] Extracts features: `hour`, `day_of_week`, `month`, `is_weekend`.
- [ ] Joins Traffy data with Weather data on `date`.
- [ ] Saves the final dataset to `data/processed/traffy_features.parquet`.

---

## Epic 2: Machine Learning Modeling

### Story 2.1: Train Baseline Regression Model
**As a** Data Scientist,
**I want to** train a baseline model (e.g., Random Forest),
**So that** I can establish a performance benchmark.

**Acceptance Criteria:**
- [ ] Script `ml/train.py` splits data into Train (80%) and Test (20%).
- [ ] Trains a valid Scikit-Learn or XGBoost regressor.
- [ ] Targets `resolution_time_days`.

### Story 2.2: Evaluate Model Performance
**As a** Data Scientist,
**I want to** calculate MAE, RMSE, and R² scores,
**So that** I know if the model is accurate enough for production.

**Acceptance Criteria:**
- [ ] Evaluation metrics are printed to console or saved to a log.
- [ ] MAE is calculated (Target: < 30 days).
- [ ] R² is calculated (Target: > 0.3).

### Story 2.3: Analyze Feature Importance
**As a** Stakeholder,
**I want to** know which factors influence resolution time the most,
**So that** I can make policy recommendations.

**Acceptance Criteria:**
- [ ] Generate a chart or list of top 10 important features.
- [ ] Example output: "District is the #1 factor", "Rainfall is #5".

---

## Epic 3: Visualization & Reporting

### Story 3.1: Create Geospatial Resolution Map
**As a** BMA Official,
**I want to** see a map of Bangkok with districts colored by average resolution time,
**So that** I can identify slow-performing zones.

**Acceptance Criteria:**
- [ ] Script `visualization/resolution_map.py` uses Folium.
- [ ] Map displays Bangkok districts.
- [ ] Color scale ranges from Green (Fast) to Red (Slow).
- [ ] Tooltips show average days for that district.
- [ ] Output saved as HTML file.

---

## Epic 4: Application Development

### Story 4.1: Develop Prediction Demo App
**As a** Citizen,
**I want to** enter my complaint details (District, Type) into a web app,
**So that** I can get an estimated resolution date.

**Acceptance Criteria:**
- [ ] Streamlit app `app/streamlit_app.py` runs locally.
- [ ] Input fields: Dropdown for District, Dropdown for Complaint Type.
- [ ] Button: "Predict Resolution Time".
- [ ] Output: Displays "Estimated Time: X Days".

### Story 4.2: Integrate Visualizations into App
**As a** User,
**I want to** see the geospatial map and stats within the same app,
**So that** I have a unified dashboard.

**Acceptance Criteria:**
- [ ] The Folium map from Story 3.1 is embedded in the Streamlit app.
- [ ] Dashboard charts (bar charts of category performance) are displayed.

---

## Technical Tasks (Chore)
- [ ] Set up virtual environment and install `requirements.txt`.
- [ ] Configure `.gitignore` for data and temp files.
- [ ] Create directory structure.
