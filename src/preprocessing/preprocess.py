
import numpy as np

def drop_constant_sensors(df):
    sensor_cols = [col for col in df.columns if "sensor_" in col]
    
    for col in sensor_cols:
        if df[col].std() == 0:
            df.drop(columns=[col], inplace=True)
    
    return df


def normalize_sensors(df):
    sensor_cols = [col for col in df.columns if "sensor_" in col]
    
    df[sensor_cols] = (df[sensor_cols] - df[sensor_cols].mean()) / df[sensor_cols].std()
    
    return df