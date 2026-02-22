
import pandas as pd
import os

COLUMN_NAMES = (
    ["unit", "cycle"] +
    [f"setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

def load_txt_file(filepath):
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None
    )
    
    df = df.iloc[:, :26]  # remove extra blank columns
    df.columns = COLUMN_NAMES
    
    return df


def load_multiple_files(base_path, file_list):
    all_data = []
    
    for file in file_list:
        full_path = os.path.join(base_path, file)
        df = load_txt_file(full_path)
        df["dataset"] = file.replace(".txt", "")
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)