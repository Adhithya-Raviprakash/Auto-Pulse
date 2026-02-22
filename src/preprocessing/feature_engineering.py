def add_rolling_features(df, window=5):
    sensor_cols = [col for col in df.columns if "sensor_" in col]
    
    for col in sensor_cols:
        df[f"{col}_rolling_mean"] = (
            df.groupby("unit")[col]
            .rolling(window)
            .mean()
            .reset_index(0, drop=True)
        )
        
        df[f"{col}_rolling_std"] = (
            df.groupby("unit")[col]
            .rolling(window)
            .std()
            .reset_index(0, drop=True)
        )
    
    df.fillna(0, inplace=True)
    
    return df