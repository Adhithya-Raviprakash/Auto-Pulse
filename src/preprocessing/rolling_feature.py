def add_rolling_features(df, window=5):
    
    sensor_cols = [
        "T2", "T24", "T30", "T50",
        "P2", "P15",
        "Nc", "Nf",
        "Ps30", "phi",
        "NRf", "NRc",
        "BPR", "farB",
        "htBleed", "W31", "W32"
    ]
    
    for col in sensor_cols:
        df[f"{col}_rolling_mean"] = (
            df.groupby("engine_id")[col]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        df[f"{col}_rolling_std"] = (
            df.groupby("engine_id")[col]
            .rolling(window)
            .std()
            .reset_index(level=0, drop=True)
        )
    
    df.fillna(0, inplace=True)
    
    return df