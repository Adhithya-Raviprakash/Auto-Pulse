
import pandas as pd
import os

def load_txt_file(filepath):
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None
    )
    return df

def add_column_names(df):

    # Keep only first 26 columns (CMAPSS standard format)
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

    col_names = index_names + operational_settings + sensor_names

    # Assign new column names
    df.columns = col_names

    return df

def load_multiple_files(base_path, file_list):
    all_data = []
    
    for file in file_list:
        full_path = os.path.join(base_path, file)
        df = load_txt_file(full_path)
        df["dataset"] = file.replace(".txt", "")
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)