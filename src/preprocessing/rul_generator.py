
def add_rul_column(df):
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    
    df = df.merge(max_cycles, on="unit")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    
    df.drop(columns=["max_cycle"], inplace=True)
    
    return df