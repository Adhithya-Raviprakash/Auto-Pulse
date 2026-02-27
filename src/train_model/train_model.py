
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# NASA Scoring Function

def nasa_score(y_true, y_pred):
    score = 0
    for true, pred in zip(y_true, y_pred):
        diff = pred - true
        if diff < 0:
            score += np.exp(-diff / 13) - 1
        else:
            score += np.exp(diff / 10) - 1
    return score

# Main Training Function

def train_and_save_models(train_path, test_path):

    print("Loading Data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    drop_cols = ['engine_id', 'cycle']

    X_train = train_df.drop(drop_cols + ['RUL'], axis=1)
    y_train = train_df['RUL']

    X_test = test_df.drop(drop_cols + ['RUL'], axis=1)
    y_test = test_df['RUL']


    # RANDOM FOREST
    
    print("\nTraining Random Forest...")
    
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_nasa = nasa_score(y_test.values, rf_pred)

    print("\nRandom Forest Results")
    print("RMSE:", rf_rmse)
    print("MAE:", rf_mae)
    print("R2:", rf_r2)
    print("NASA Score:", rf_nasa)

    # Save RF model
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    print("Random Forest model saved as random_forest_model.pkl")

    # XGBOOST
    
    print("\nTraining XGBoost...")

    xgb_model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,
    reg_alpha=0.5,
    random_state=42
    )
    
    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)

    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_nasa = nasa_score(y_test.values, xgb_pred)

    print("\nXGBoost Results")
    print("RMSE:", xgb_rmse)
    print("MAE:", xgb_mae)
    print("R2:", xgb_r2)
    print("NASA Score:", xgb_nasa)

    # Save XGB model
    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    print("XGBoost model saved as xgboost_model.pkl")

    print("\nTraining Complete ✅")


# Run Script

if __name__ == "__main__":
    train_and_save_models(
        "data/processed/fd001_train_processed.csv",
        "data/processed/fd001_test_processed.csv"
    )