# 🔧 Predictive Maintenance — NASA CMAPSS Turbofan Engine Dataset

> **End-to-end machine learning pipeline for Remaining Useful Life (RUL) prediction of aircraft engines using multivariate time-series sensor data.**

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [How to Reproduce](#how-to-reproduce)
- [Key Takeaways](#key-takeaways)

---

## Project Overview

This project implements a complete **predictive maintenance pipeline** using the [NASA CMAPSS](https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) turbofan engine degradation dataset.

The core objective is to predict the **Remaining Useful Life (RUL)** of aircraft engines before failure, enabling proactive maintenance decisions.

**Beyond just training models, this project emphasizes:**

- Degradation pattern analysis
- Time-aware feature engineering
- Model benchmarking & comparison
- Statistical evaluation & residual diagnostics
- Engine-wise lifecycle interpretation

---

## Dataset

**Source:** NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)

Each engine record contains:
- Multiple operational cycles from healthy state to failure
- Multivariate sensor readings per cycle
- Operating condition settings

**Target Variable:** `RUL` — the number of remaining cycles before engine failure

### ⚠️ Data Not Included

Raw and processed CSV files are **not included** in this repository due to storage limitations.

To reproduce results:

1. Download the NASA CMAPSS dataset from the [NASA data portal](https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)
2. Place raw files inside `data/raw/`
3. Run preprocessing and feature engineering scripts to generate processed data in `data/processed/`

---

## Project Structure

```
├── data/
│   ├── raw/                   # Raw CMAPSS files (not included)
│   └── processed/             # Engineered features (generated locally)
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis (raw)
│   ├── 02_feature_engineering.ipynb  # Feature engineering
│   ├── 03_model_test.ipynb           # Model training & evaluation
│   └── 04_eda_processed.ipynb        # EDA on processed features
│
├── src/
│   ├── preprocessing/
│   │   └── save_preprocess.py        # Preprocessing pipeline
│   └── train_model/
│       └── train_model.py            # Model training script
│
├── requirements.txt
└── README.md
```

---

## Workflow

### 1. Exploratory Data Analysis

**`01_eda.ipynb`** — Raw sensor exploration:
- Identify degradation trends over engine cycles
- Detect constant or non-informative sensor channels
- Analyze engine lifecycle distributions

**`04_eda_processed.ipynb`** — Validation of engineered features and transformed dataset

---

### 2. Feature Engineering

**`02_feature_engineering.ipynb`**

Raw sensor readings alone don't sufficiently capture gradual degradation patterns. Four categories of time-aware features were engineered:

| Feature Type | Description | Purpose |
|---|---|---|
| **Rolling Mean** | Smoothed averages over recent cycles | Suppress noise; surface long-term trends |
| **Rolling Slope** | Linear regression slope over a rolling window | Capture degradation velocity and direction |
| **Delta Features** | Cycle-over-cycle sensor difference | Detect sudden shifts and short-term instability |
| **Lifecycle Progress** | `cycle / max_cycle_per_engine` (normalized) | Inform model of engine's current lifecycle stage |

> **Rolling slope features** are particularly impactful — encoding both the direction and rate of degradation significantly improves RUL prediction across long lifecycle spans.

---

### 3. Model Training & Evaluation

**`03_model_test.ipynb`** | **`src/train_model/train_model.py`**

Two models were trained and benchmarked:

- **Random Forest** — ensemble baseline
- **XGBoost** — gradient boosting final model

> ⏱️ *Random Forest training takes approximately 5 minutes. Rolling feature computation may also take several minutes due to grouped time-series operations. Please allow time for these steps to complete.*

---

## Model Performance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | 13.09 | 9.56 | 0.80 |
| **XGBoost** | **12.07** | **8.74** | **0.83** |

**Key observations:**
- XGBoost reduced RMSE by ~8% over Random Forest
- The final model explains **83% of variance** in RUL predictions
- Average prediction error is approximately **9 engine cycles**

**Evaluation methods used:**
- RMSE, MAE, R² metrics
- Actual vs. predicted scatter plots
- Engine-wise degradation curve visualization
- Residual diagnostics and correlation analysis

---

## How to Reproduce

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Download Dataset

Download the NASA CMAPSS dataset and place the raw files in:

```
data/raw/
```

### Step 3 — Preprocess & Train

Run in the following order:

```bash
# 1. Preprocess raw data
python src/preprocessing/save_preprocess.py

# 2. Train models
python src/train_model/train_model.py
```

### Step 4 — Run Notebooks

Explore the full analysis in sequence:

```
01_eda.ipynb
04_eda_processed.ipynb
02_feature_engineering.ipynb
03_model_test.ipynb
```

---

## Key Takeaways

- **Time-aware feature engineering is essential** — raw sensor data alone is insufficient for degradation modeling
- **Rolling slope features** significantly improve prediction by encoding the rate of change, not just current state
- **Gradient boosting outperforms bagging** for lifecycle regression tasks on this dataset
- **Diagnostic evaluation matters** — aggregate metrics alone don't reveal where and why models fail across different engine lifecycles

---

## Conclusion

This project demonstrates a production-style Data Science pipeline for predictive maintenance, combining:

- Statistical trend analysis
- Temporal feature engineering
- Comparative model benchmarking
- Residual & lifecycle diagnostics

It reflects a practical understanding of time-series degradation modeling and how thoughtful feature design drives predictive accuracy in industrial maintenance applications.

---

*Dataset: NASA CMAPSS Jet Engine Simulated Data — publicly available via the NASA data portal.*
