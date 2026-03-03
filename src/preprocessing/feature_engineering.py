#Feature Engineering for CMAPSS

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import linregress

#2.Delta Feature (Rate Of Change)
def add_delta_features(df):
    
    df = df.copy()
    exclude_cols = ["engine_id", "cycle", "RUL"]
    sensor_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(exclude_cols)

    new_features = []

    for col in sensor_cols:
        delta = df.groupby("engine_id")[col].diff()
        new_features.append(delta.rename(f"{col}_delta"))

    df = pd.concat([df] + new_features, axis=1)
    return df

# 3. Operating Condition Clustering

def add_operating_condition_cluster(df, n_clusters=6):
    op_cols = ['setting_1', 'setting_2','setting_3']#op_setting_3 is removed as its dead/constant column removed in preprocessing

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['op_cluster'] = kmeans.fit_predict(df[op_cols])

    return df
