
def add_rul_column(df):

    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycles, on="engine_id", how="left")

    df["RUL"] = df["max_cycle"] - df["cycle"]

    df.drop("max_cycle", axis=1, inplace=True)

    return df