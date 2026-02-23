
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

    columns = ['engine_id', 'cycle']
    columns += [f'op_setting_{i}' for i in range(1, 4)]
    columns += [f'sensor_{i}' for i in range(1, 22)]

    df.columns = columns

    return df

def load_multiple_files(base_path, file_list):
    all_data = []
    
    for file in file_list:
        full_path = os.path.join(base_path, file)
        df = load_txt_file(full_path)
        df["dataset"] = file.replace(".txt", "")
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)