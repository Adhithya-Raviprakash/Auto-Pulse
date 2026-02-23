📌 Predictive Maintenance Model for Vehicle Component Failure using Sensor Data
🚗 Overview

This project builds an end-to-end predictive maintenance system that estimates:

🔴 Failure Probability of a vehicle component

⏳ Remaining Useful Life (RUL)

📊 Component Risk Level

The model uses time-series sensor readings (temperature, vibration, pressure, RPM, etc.) to predict potential failures before they occur.

This solution simulates real-world predictive maintenance systems used in:

Tesla

Ford

Bosch

Fleet vehicle monitoring systems

EV battery management systems

🎯 Business Problem

Unexpected vehicle component failure leads to:

High maintenance cost

Warranty claims

Downtime in fleet vehicles

Safety risks

This project predicts failures in advance, enabling preventive maintenance instead of reactive repair.

📊 Dataset

NASA Turbofan Engine Dataset (CMAPSS) {Current Choice}

Multivariate time-series sensor data

Multiple engines with degradation cycles

Failure labels and Remaining Useful Life (RUL)


predictive-maintenance-vehicle/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md