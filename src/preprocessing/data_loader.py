import pandas as pd
import os

# Load single txt file

def load_txt_file(filepath):
    return pd.read_csv(filepath, sep=r"\s+", header=None)


# Add column names (CMAPSS standard)

def add_column_names(df):

    df = df.iloc[:, :26]

    index_names = ['engine_id', 'cycle']

    operational_settings = ['setting_1', 'setting_2', 'setting_3']

    sensor_names = [
        'T2', 'T24', 'T30', 'T50',
        'P2', 'P15', 'P30',
        'Nf', 'Nc', 'epr', 'Ps30',
        'phi', 'NRf', 'NRc',
        'BPR', 'farB', 'htBleed',
        'Nf_dmd', 'PCNfR_dmd',
        'W31', 'W32'
    ]

    df.columns = index_names + operational_settings + sensor_names

    return df

# Load FD001 TRAIN

def load_train(base_path, filename="train_FD001.txt"):

    filepath = os.path.join(base_path, filename)
    df = load_txt_file(filepath)
    df = add_column_names(df)

    # Generate RUL
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)

    df["RUL"] = df["RUL"].clip(upper=125)# Clipping RUL

    return df

# Load FD001 TEST + RUL

def load_test(base_path,
              test_file="test_FD001.txt",
              rul_file="RUL_FD001.txt"):

    # Load test
    test_path = os.path.join(base_path, test_file)
    test_df = load_txt_file(test_path)
    test_df = add_column_names(test_df)

    # Load RUL
    rul_path = os.path.join(base_path, rul_file)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)
    rul_df.columns = ["RUL"]
    rul_df["engine_id"] = rul_df.index + 1

    # Compute max cycle in test
    max_cycle = test_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    test_df = test_df.merge(max_cycle, on="engine_id")
    test_df = test_df.merge(rul_df, on="engine_id")

    test_df["RUL"] = test_df["RUL"] + (test_df["max_cycle"] - test_df["cycle"])
    test_df["RUL"] = test_df["RUL"].clip(upper=125)

    test_df.drop(columns=["max_cycle"], inplace=True)

    return test_df