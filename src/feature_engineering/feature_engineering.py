#Feature Engineering for CMAPSS

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import linregress

# 1. RUL Capping

def cap_rul(df, threshold=125):
    df['RUL'] = df['RUL'].clip(upper=threshold)
    return df

# 2. Delta Features (Rate of Change)

def add_delta_features(df):
    sensor_cols = [col for col in df.columns if 'sensor_' in col]

    for col in sensor_cols:
        df[f'{col}_delta'] = df.groupby('engine_id')[col].diff()

    return df


# 3. EWMA(Exponential Weighted Moving Average) Features

def add_ewma_features(df, alpha=0.3):
    sensor_cols = [col for col in df.columns if 'sensor_' in col]

    for col in sensor_cols:
        df[f'{col}_ewma'] = (
            df.groupby('engine_id')[col]
            .transform(lambda x: x.ewm(alpha=alpha).mean())
        )

    return df

# 4. Operating Condition Clustering

def add_operating_condition_cluster(df, n_clusters=6):
    op_cols = ['setting_1', 'setting_2','setting_3']#op_setting_3 is removed as its dead/constant column removed in preprocessing

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['op_cluster'] = kmeans.fit_predict(df[op_cols])

    return df

# FULL PIPELINE

def feature_engineering_pipeline(df):

    df = cap_rul(df)
    df = add_delta_features(df)
    df = add_ewma_features(df)
    df = add_operating_condition_cluster(df)

    return df