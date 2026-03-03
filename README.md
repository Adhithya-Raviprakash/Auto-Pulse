Predictive Maintenance using NASA CMAPSS Dataset
Project Overview

This project implements an end-to-end predictive maintenance pipeline using the NASA CMAPSS turbofan engine degradation dataset.

The objective is to predict Remaining Useful Life (RUL) of aircraft engines based on multivariate time-series sensor data collected over operational cycles.

This project emphasizes a Data Science workflow:

Degradation pattern analysis

Time-aware feature engineering

Model benchmarking

Statistical evaluation

Diagnostic error analysis

The focus is not only model training, but understanding degradation behavior and interpreting model performance.

Dataset

Dataset: NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)

Each engine:

Operates across multiple cycles

Contains multiple sensor measurements per cycle

Gradually degrades until failure

Target variable:

Remaining Useful Life (RUL)

Important Note About Data

The raw and processed CSV files are not included in this repository due to storage limitations.

To reproduce results:

Download the NASA CMAPSS dataset.

Place raw files inside:

data/raw/

Run preprocessing and feature engineering notebooks to generate processed data inside:

data/processed/

Workflow
1. Exploratory Data Analysis

01_eda.ipynb
Initial exploration of raw sensor data to:

Identify degradation trends

Detect constant or non-informative sensors

Analyze lifecycle patterns

04_eda_processed.ipynb
Validation of engineered features and transformed dataset.

2. Feature Engineering

02_feature_engineering.ipynb

Raw sensor readings alone do not sufficiently capture gradual degradation patterns. Therefore, time-aware features were engineered.

Rolling Mean

Rolling averages smooth short-term noise and highlight long-term degradation behavior.

This helps the model detect slow deterioration rather than reacting to random fluctuations.

Rolling Slope (Trend Feature)

Rolling slope is computed using linear regression over recent cycles.

This captures:

Direction of change

Rate of degradation

Slope encodes degradation velocity and significantly improves lifecycle modeling.

Delta Features

Delta = difference between current value and previous cycle.

This captures short-term instability and sudden sensor shifts that may indicate accelerating degradation.

Lifecycle Progress Feature

Normalized cycle progression:

cycle / max_cycle_per_engine

This explicitly informs the model about the engine’s stage in its lifecycle.

3. Model Training

03_model_test.ipynb and src/train_model.py

Models implemented:

Random Forest (baseline)

XGBoost (final model)

Random Forest training time: approximately 5 minutes depending on hardware.
Rolling feature computation may also take several minutes due to grouped operations.

Please allow time for feature engineering and training steps to complete.

Model Performance
Model	RMSE	MAE	R²
Random Forest	13.09	9.56	0.80
XGBoost	12.07	8.74	0.83

Key Observations:

XGBoost reduced RMSE by approximately 8% over Random Forest.

The final model explains 83% of variance in RUL.

Average prediction error is approximately 9 cycles.

Evaluation Strategy

Model evaluation includes:

RMSE

MAE

R²

Correlation analysis

Actual vs Predicted scatter plots

Engine-wise degradation visualization

Residual diagnostics

This ensures model behavior is interpreted and validated rather than evaluated solely by aggregate metrics.

How to Reproduce
Step 1: Install Dependencies
pip install -r requirements.txt
Step 2: Download Dataset

Place raw CMAPSS files inside:

data/raw/
Step 3:Run Preprocessing and train models
, propocessing in src/preprocessing/save_preprocess.py
, model training in src/train_model/train_model.py

  in that order
Step 4: Run Notebooks in Order

01_eda.ipynb

02_eda_processed.ipnb

03_feature_engineering.ipynb

04_model_test.ipynb

, train models directly:

python src/train_model.py

Key Takeaways:

Time-aware feature engineering is critical for degradation modeling.

Rolling slope features significantly improve predictive accuracy.

Gradient boosting methods outperform bagging methods in lifecycle modeling.

Diagnostic analysis is essential for predictive maintenance systems.

Conclusion

This project demonstrates a complete Data Science pipeline for predictive maintenance, combining:

Statistical trend analysis

Temporal feature engineering

Model benchmarking

Performance diagnostics

Lifecycle interpretation

It reflects practical understanding of time-series degradation modeling and predictive analytics.
