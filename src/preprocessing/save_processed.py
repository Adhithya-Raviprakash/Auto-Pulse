
from config import DATA_PATH, TRAIN_FILES
from data_loader import load_multiple_files,add_column_names
from rul_generator import add_rul_column
from preprocess import drop_constant_sensors
from rolling_feature import add_rolling_features

def run_pipeline():
    print("Loading data...")
    df = load_multiple_files(DATA_PATH, TRAIN_FILES)
    
    print("Adding column Names...")
    df = add_column_names(df)
    
    print("Generating RUL...")
    df = add_rul_column(df)
    
    print("Dropping constant sensors...")
    df = drop_constant_sensors(df)
    
    print("Adding rolling features...")
    df = add_rolling_features(df)
    
    print("Saving processed file...")
    df.to_csv("data/processed/train_processed.csv", index=False)
    
    print("Done.")


if __name__ == "__main__":
    run_pipeline()