import numpy as np
import pandas as pd

def add_rolling_features(df, window=20):

    exclude_cols = ["engine_id", "cycle", "RUL"]
    sensor_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(exclude_cols)

    for col in sensor_cols:
        rolling_group = df.groupby("engine_id")[col].rolling(window, min_periods=1)
        
        df[f"{col}_rolling_mean"] = rolling_group.mean().reset_index(level=0, drop=True)
        df[f"{col}_rolling_std"]  = rolling_group.std().reset_index(level=0, drop=True)

    return df


def add_rolling_slope(df, window=20):
    
    df = df.copy()
    exclude_cols = ["engine_id", "cycle", "RUL"]
    sensor_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(exclude_cols)

    for col in sensor_cols:
        
        slopes = []

        for engine_id, group in df.groupby("engine_id"):
            
            values = group[col].values
            engine_slopes = []
            
            for i in range(len(values)):
                
                start = max(0, i - window + 1)
                y = values[start:i+1]
                
                if len(y) < 2:
                    engine_slopes.append(0)
                else:
                    x = np.arange(len(y))
                    slope = np.polyfit(x, y, 1)[0]
                    engine_slopes.append(slope)
            
            slopes.extend(engine_slopes)

        df[f"{col}_slope"] = slopes

    return df