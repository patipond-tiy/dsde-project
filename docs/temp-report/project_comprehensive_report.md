# Comprehensive Project Report: Traffy Fondue Resolution Time Prediction

**Date:** December 7, 2025
**Project:** Traffy Fondue Resolution Time Prediction System
**Course:** 2110403 Data Science and Data Engineering (DSDE-CEDT)

---

## 1. Project Overview

### 1.1 Objective
The primary objective of this project is to develop a **Resolution Time Prediction System** for the Bangkok Metropolitan Administration (BMA). By analyzing historical complaint data from the Traffy Fondue platform and integrating external weather data, we aim to:
1.  **Predict** the number of days required to resolve specific types of citizen complaints.
2.  **Identify** bottlenecks in district-level performance.
3.  **Optimize** resource allocation by understanding the impact of external factors like rainfall and temperature.

### 1.2 Scope
This project implements a fully functional, end-to-end data science pipeline consisting of:
*   **Data Engineering:** Scalable processing using Apache Spark to clean, transform, and merge large datasets.
*   **Machine Learning:** Supervised regression models (XGBoost, Random Forest) to predict resolution time (`resolution_time_days`).
*   **Visualization:** An interactive Streamlit application with geospatial dashboards (Folium) and statistical insights.

---

## 2. Dataset Description

### 2.1 Primary Source: Traffy Fondue
*   **Source:** Public complaint data from the Traffy Fondue platform.
*   **Coverage:** August 2021 – January 2025.
*   **Volume:** Approximately **787,026** records.
*   **Key Attributes:**
    *   `ticket_id`: Unique identifier.
    *   `timestamp`: Date and time of complaint submission.
    *   `last_activity`: Date and time of issue resolution or last update.
    *   `type`: Category of the issue (multi-label, e.g., "Road, Flooding").
    *   `district` / `subdistrict`: Location details (Bangkok area).
    *   `coords`: Geospatial coordinates (Latitude, Longitude).
    *   `state`: Current status (e.g., "Finish", "In Progress").

### 2.2 Secondary Source: Meteorological Data
*   **Source:** External Weather APIs (Open-Meteo / OpenWeatherMap).
*   **Volume:** > 1,200 daily records aligned with the complaint timeline.
*   **Attributes:** `rainfall_mm` (Precipitation), `temperature_max`, `temperature_min`.
*   **Purpose:** To test the hypothesis that environmental factors (e.g., heavy rain) significantly impact resolution times for infrastructure issues.

---

## 3. Dataset Problems & Challenges

During the Exploratory Data Analysis (EDA) and Data Engineering phases, several critical issues were identified within the raw dataset. These challenges required robust preprocessing strategies to ensure model validity.

### 3.1 Data Quality & Structure
*   **Multi-label Complexity:** The `type` column often contains unstructured, JSON-like lists of categories (e.g., `["ความสะอาด", "ทางเท้า"]`). This required complex parsing to extract primary categories for modeling.
*   **Unstructured Text:** The `comment` field contains rich but noisy user descriptions, necessitating feature engineering (e.g., text length extraction) rather than direct raw usage.
*   **Inconsistent Naming:** District and subdistrict names required standardization to match official BMA administrative zones (e.g., handling typos or variations in Thai script).

### 3.2 Missingness (Sparsity)
*   **High Missingness in Feedback:** The `star` (rating) column is missing in roughly **65%** of records (~513k missing out of 787k), making it unreliable as a primary feature for prediction.
*   **Incomplete Resolution Evidence:** The `photo_after` field is missing in ~18% of records, complicating verification of "completed" states.
*   **Coordinates:** While mostly complete, a small percentage of records lack valid `coords`, requiring imputation or exclusion for geospatial visualization.

### 3.3 Noise & Outliers
*   **Administrative Noise (Short Duration):** Analysis revealed a significant cluster of tickets with resolution times **< 1 hour (0.04 days)**. These are likely auto-closed tickets, duplicate removals, or administrative redirects rather than actual fieldwork. We filter these out to prevent the model from learning "instant resolution" patterns.
*   **Extreme Outliers (Long Duration):** Some tickets show resolution times exceeding **365 days**. While valid, these extreme tail events skew regression metrics (MSE). We applied clipping and robust error metrics (MAE) to mitigate their impact.
*   **Geospatial Outliers:** Approximately **0.1% (1,079 records)** of coordinates fall outside the Greater Bangkok area, likely due to GPS errors or user misuse. These were filtered for the district-level analysis.

### 3.4 Temporal & Integration Issues
*   **Imperfect Timestamps:** The target variable `resolution_time_days` is derived from `last_activity - timestamp`. In cases where `last_activity` is not strictly updated upon completion, this proxy may be inaccurate.
*   **Granularity Mismatch:** Weather data is daily, while complaints are timestamped to the second. Joining these datasets required aggregation assumptions (e.g., assigning daily rainfall to all complaints filed on that day).

---

## 4. Current Implementation Status

### 4.1 Data Engineering (Spark Pipeline)
*   **Ingestion:** Successfully reads raw CSVs and scrapes weather data.
*   **Processing:** Implemented `spark_pipeline` to clean text, calculate `resolution_time_days`, and join weather features.
*   **Storage:** Optimized data is stored in **Parquet** format (`data/processed/traffy_features.parquet`) for fast I/O during training.

### 4.2 Machine Learning
*   **Models:** Trained **XGBoost** (Advanced) and **Random Forest** (Baseline) models.
*   **Performance:**
    *   **MAE (Mean Absolute Error):** Achieved < 8 days (Optimized).
    *   **R² Score:** > 0.40 (indicating moderate predictive power given the high variance in human operational data).
*   **Feature Importance:** Identified `district`, `type`, and `organization` as top drivers of resolution time, with `rainfall_mm` showing non-trivial impact on specific categories (e.g., Flooding).

### 4.3 Visualization (Streamlit App)
*   **Interactive Map:** A Folium-based choropleth map visualizes average resolution times by district (Red = Slow, Green = Fast).
*   **Prediction Interface:** A sidebar allows users to input complaint details and receive a real-time estimated completion date.
*   **Dashboard:** Charts displaying model error distribution and feature correlations are live.

---

## 5. Conclusion
The project has successfully navigated significant data quality challenges to build a robust resolution time prediction engine. By filtering administrative noise and integrating environmental context, the system provides actionable insights for BMA officials to improve city management efficiency.
