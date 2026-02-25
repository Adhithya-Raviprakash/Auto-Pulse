
from config import DATA_PATH
from rolling_feature import add_rolling_features
from data_loader import load_train, load_test
import feature_engineering

def run_pipeline(base_path):

    print("Loading train...")
    train_df = load_train(base_path)

    print("Loading test...")
    test_df = load_test(base_path)

    print("Dropping constant sensors...")
    constant_cols = train_df.nunique()[train_df.nunique() == 1].index

    train_df.drop(columns=constant_cols, inplace=True)
    test_df.drop(columns=constant_cols, inplace=True)

    print("Adding rolling features...")
    train_df = add_rolling_features(train_df)
    test_df  = add_rolling_features(test_df)

    print("Adding delta features...")
    train_df = feature_engineering.add_delta_features(train_df)
    test_df  = feature_engineering.add_delta_features(test_df)

    print("Dropping NaNs...")
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    
    print("Saving processed CSV files...")

    train_df.to_csv("data/processed/fd001_train_processed.csv", index=False)
    test_df.to_csv("data/processed/fd001_test_processed.csv", index=False)

    print("Pipeline complete.")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
 
    
if __name__ == "__main__":
   run_pipeline(DATA_PATH)