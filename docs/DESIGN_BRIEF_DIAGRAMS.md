# Design Brief: Project Visualizations & Diagrams

This document outlines the requirements for creating high-quality visualizations and diagrams for the **Traffy Fondue Resolution Prediction Project**. These assets will be used in the final report and presentation to technical and non-technical stakeholders.

## 1. System Architecture Diagram
**Goal:** Illustrate the end-to-end flow of data from the source to the end-user application.
**Target Audience:** Technical stakeholders, System Architects.

### Key Components:
1.  **Data Source:** 
    *   Traffy Fondue API / Open Data (Raw JSON/CSV).
    *   External Weather Data (Rainfall, Temperature).
2.  **Data Processing (ETL):**
    *   Cleaning & Filtering (Pandas/Spark).
    *   Feature Engineering (Coordinate mapping, Time extraction).
    *   Storage: `traffy_features.parquet`.
3.  **Machine Learning Core:**
    *   Training Pipeline (XGBoost, Scikit-Learn).
    *   Model Artifacts (`.joblib` files).
4.  **Application Layer:**
    *   Streamlit Web App (Python).
    *   User Interface (Inputs: Problem Type, Location, Description).
    *   Prediction Engine (Inference).

### Flow Description:
*   Raw data is ingested and merged with weather data.
*   The processed dataset feeds into the ML Training Pipeline.
*   The trained model is serialized and loaded by the Streamlit App.
*   Users input case details into the App, which generates a predicted resolution time.

### Visual Style:
*   **Layout:** Left-to-Right flow.
*   **Icons:** Database icon, Python/Pandas logo, Gear/Cog for processing, Brain/Network for ML, Web/Mobile screen for App.
*   **Connectors:** Solid arrows for data flow, dashed arrows for model loading.

---

## 2. ML Pipeline & Model Architecture
**Goal:** Explain the specific "Hybrid Resampling" and feature engineering strategy used to handle the imbalanced data.
**Target Audience:** Data Scientists, ML Engineers.

### Key Elements:
1.  **Input Data:** The `traffy_features.parquet` dataset.
2.  **Preprocessing Blocks:**
    *   **Text Analysis:** Thai Tokenizer -> TF-IDF -> SVD (Dimensionality Reduction).
    *   **Categorical:** Target Encoding (District, Issue Type).
    *   **Temporal:** Cyclical features (Sin/Cos for Hour/Month).
    *   **Historical:** Median Resolution Time (by District/Type).
3.  **Resampling Strategy (The "Hybrid" Approach):**
    *   Visual representation of downsampling frequent classes (fast resolution) and upsampling rare classes (slow resolution/outliers).
    *   *Concept:* Balancing the dataset to prevent bias towards short resolution times.
4.  **Model:** XGBoost Regressor.
5.  **Output:** Predicted Log(Days) -> Exponentiated to Actual Days.

### Visual Style:
*   **Layout:** Vertical stack or flowchart.
*   **Emphasis:** Highlight the "Resampling" block as it's the key innovation of this project.
*   **Color Coding:** Blue for data transformations, Green for the ML model, Red for the final output.

---

## 3. Performance Evolution Chart (Infographic Style)
**Goal:** Showcase how the model improved from a baseline to the final version.
**Target Audience:** Project Managers, Business Stakeholders.

### Data Points:
*   **Baseline (Robust V1):** MAE ~24 days, $R^2$ ~0.10 (Poor predictive power).
*   **Intermediate (Resampling):** MAE ~36 days, $R^2$ ~0.37.
*   **Final (Hybrid V2):** MAE ~34 days, $R^2$ ~0.61 (High predictive power).

### Visual Concept:
*   A "Staircase" or "Growth" chart showing the $R^2$ score (Accuracy) increasing.
*   Annotate each step with the key change (e.g., "Added Weather Data", "Implemented Hybrid Resampling", "Tuned Hyperparameters").
*   *Note:* While MAE fluctuates (due to the model learning to predict harder, longer cases), the $R^2$ is the hero metric here.

---

## 4. Feature Importance Map
**Goal:** Visualize what factors most strongly influence the resolution time.
**Target Audience:** General Audience, City Officials.

### Key Insights to Visualize:
1.  **Issue Type:** The specific problem (e.g., "Road", "Electric", "Flooding") is the #1 predictor.
2.  **Location:** District/Subdistrict (Wealthy vs. Peripheral areas).
3.  **Seasonality:** Month of the year (Rainy season impact).
4.  **Description:** The length and content of the user's complaint.

### Visual Style:
*   **Word Cloud or Bubble Chart:** Size of the bubble = Importance of the feature.
*   **Groupings:** Group bubbles by category (Location, Time, Content, Environment).
*   **Example:** A large central bubble for "Issue Type", surrounded by "District" and "Month". Smaller bubbles for "Rainfall" and "Comment Length".

---

## 5. User Journey / Storyboard
**Goal:** Humanize the technology by showing a user scenario.
**Target Audience:** End Users, UX Designers.

### Steps:
1.  **Problem Occurs:** User sees a pothole or broken light.
2.  **Reporting:** User opens the app, selects "Road", selects "Chatuchak", types "Big hole near park".
3.  **Processing:** The system analyzes the text and history (illustrated as a fast compute step).
4.  **Prediction:** App displays: *"Estimated Resolution: 3-7 Days"*.
5.  **Action:** User plans accordingly; Authority is notified.

### Visual Style:
*   **Storyboard:** 3-4 panels (comic strip style).
*   **Characters:** Simple flat-design avatars.
*   **UI Mockups:** Simplified wireframes of the app interface in the panels.

---

## Technical Specifications for Export
*   **Format:** Vector (SVG/PDF) for print, High-Res PNG for web.
*   **Color Palette:**
    *   Primary: Traffy Fondue Green (`#2ca02c`) or similar.
    *   Secondary: Tech Blue (`#3498db`).
    *   Alert/Highlight: Orange/Red (`#e74c3c`).
*   **Fonts:** Clean Sans-Serif (e.g., Roboto, Inter, Sarabun for Thai text support).
