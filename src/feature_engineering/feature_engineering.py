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

# 3. Degradation Trend (Slope Feature)

def add_slope_features(df, window=10):
    sensor_cols = [col for col in df.columns if 'sensor_' in col]

    for col in sensor_cols:
        slope_feature = []

        for engine in df['engine_id'].unique():
            engine_df = df[df['engine_id'] == engine]

            values = engine_df[col].values
            cycles = engine_df['cycle'].values

            slopes = []

            for i in range(len(values)):
                start = max(0, i - window)
                y = values[start:i+1]
                x = cycles[start:i+1]

                if len(y) > 1:
                    slope = linregress(x, y).slope
                else:
                    slope = 0

                slopes.append(slope)

            slope_feature.extend(slopes)

        df[f'{col}_slope'] = slope_feature

    return df


# 4. EWMA(Exponential Weighted Moving Average) Features

def add_ewma_features(df, alpha=0.3):
    sensor_cols = [col for col in df.columns if 'sensor_' in col]

    for col in sensor_cols:
        df[f'{col}_ewma'] = (
            df.groupby('engine_id')[col]
            .transform(lambda x: x.ewm(alpha=alpha).mean())
        )

    return df

# 5. Operating Condition Clustering

def add_operating_condition_cluster(df, n_clusters=6):
    op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['op_cluster'] = kmeans.fit_predict(df[op_cols])

    return df

# FULL PIPELINE

def feature_engineering_pipeline(df):

    df = cap_rul(df)
    df = add_delta_features(df)
    df = add_slope_features(df)
    df = add_ewma_features(df)
    df = add_operating_condition_cluster(df)

    return df