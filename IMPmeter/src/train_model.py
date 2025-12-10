import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from config import MODEL_DIR, MODEL_PATH
from feature_engineering import prepare_for_training


# ----------------------------------------------------------
# Train & evaluate a single model
# ----------------------------------------------------------

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n📌 Model: {name}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")

    return {
        "name": name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ----------------------------------------------------------
# Master training function
# ----------------------------------------------------------

def train_all_models():
    print("[INFO] Preparing data...")
    preprocessor, X_train, X_test, y_train, y_test = prepare_for_training()

    print("[INFO] Initializing models...")

    models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    ),
    "SVR_RBF": SVR(kernel="rbf", C=2.0, epsilon=0.2),
    "XGBoost": XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=3,
        random_state=42
    ),
    # "CatBoost": CatBoostRegressor(...)   # removed
}


    results = []

    print("\n[INFO] Training models...\n")

    for name, model in models.items():
        print(f"--- Training {name} ---")
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])
        res = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        results.append(res)

    # ------------------------------------------------------
    # Select best model (based on RMSE)
    # ------------------------------------------------------
    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values(by="rmse").iloc[0]

    best_model_name = best_row["name"]
    best_model = best_row["model"]

    print("\n🏆 BEST MODEL:", best_model_name)
    print(best_row)

    # ------------------------------------------------------
    # Save best model pipeline
    # ------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"\n[OK] Best model saved at {MODEL_PATH}")

    return best_model, results_df


# Run everything
if __name__ == "__main__":
    best_model, results = train_all_models()
    print("\nAll done!")
