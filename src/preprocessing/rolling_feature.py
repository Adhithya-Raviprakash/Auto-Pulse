def add_rolling_features(df, window=5):

    exclude_cols = ["engine_id", "cycle", "RUL"]
    sensor_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(exclude_cols)

    for col in sensor_cols:
        rolling_group = df.groupby("engine_id")[col].rolling(window, min_periods=1)
        
        df[f"{col}_rolling_mean"] = rolling_group.mean().reset_index(level=0, drop=True)
        df[f"{col}_rolling_std"]  = rolling_group.std().reset_index(level=0, drop=True)

    return df