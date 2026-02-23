# Build Feature Engineered Dataset

import pandas as pd
from feature_engineering import feature_engineering_pipeline


def main():

    print("Loading basic processed dataset...")

    df = pd.read_csv("data/processed/train_processed.csv")

    print("Applying feature engineering...")

    df = feature_engineering_pipeline(df)

    print("Dropping NaN values...")
    df = df.dropna()

    print("Saving feature engineered dataset...")

    df.to_csv(
        "data/processed/train_feature_engineered.csv",
        index=False
    )

    print("Feature engineered dataset saved successfully!")
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()