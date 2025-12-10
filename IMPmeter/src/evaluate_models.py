import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from feature_engineering import prepare_for_training
from config import MODEL_PATH, FINAL_DATA_PATH, TARGET_COL


# =========================================================
# LOAD MODEL + DATA
# =========================================================
def load_model_and_data():
    print("[INFO] Loading best model...")
    model = joblib.load(MODEL_PATH)

    print("[INFO] Loading dataset...")
    df = pd.read_csv(FINAL_DATA_PATH)

    return model, df


# =========================================================
# EVALUATE MODEL
# =========================================================
def evaluate(model, df):
    # prepare split exactly the same way as training
    _, X_train, X_test, y_train, y_test = prepare_for_training()

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n==============================")
    print("📌 MODEL PERFORMANCE")
    print("==============================")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")

    return X_test, y_test, preds


# =========================================================
# PLOT 1: RESIDUAL DISTRIBUTION
# =========================================================
def plot_residuals(y_test, preds):
    residuals = y_test - preds

    plt.figure(figsize=(10,5))
    sns.histplot(residuals, bins=40, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.show()


# =========================================================
# PLOT 2: ACTUAL VS PREDICTED
# =========================================================
def plot_actual_vs_predicted(y_test, preds):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, preds, alpha=0.3)
    plt.plot([0,100],[0,100], color="red")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Actual vs Predicted Impulse Buy Score")
    plt.show()


# =========================================================
# PLOT 3: FEATURE IMPORTANCE (Tree Models Only)
# =========================================================
def plot_feature_importance(model, X_test):
    # extract model from pipeline
    try:
        final_model = model.named_steps["model"]
    except:
        print("[INFO] This model has no feature importance.")
        return

    if not hasattr(final_model, "feature_importances_"):
        print("[INFO] Model does not support feature importance.")
        return

    # get preprocessor to extract feature names
    preprocessor = model.named_steps["preprocess"]
    cat_features = preprocessor.transformers_[1][2]
    num_features = preprocessor.transformers_[0][2]

    # OneHotEncoder feature names
    try:
        ohe = preprocessor.transformers_[1][1].named_steps["encoder"]
        ohe_names = list(ohe.get_feature_names_out(cat_features))
    except:
        ohe_names = list(cat_features)

    feature_names = list(num_features) + ohe_names

    importances = final_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:20]  # top 20

    plt.figure(figsize=(10,6))
    sns.barplot(
        x=importances[sorted_idx],
        y=np.array(feature_names)[sorted_idx],
        orient="h"
    )
    plt.title("Top 20 Feature Importances")
    plt.show()


# =========================================================
# RUN EVERYTHING
# =========================================================
def run_evaluation():
    model, df = load_model_and_data()
    X_test, y_test, preds = evaluate(model, df)

    print("\n[INFO] Plotting residuals...")
    plot_residuals(y_test, preds)

    print("\n[INFO] Plotting Actual vs Predicted...")
    plot_actual_vs_predicted(y_test, preds)

    print("\n[INFO] Plotting Feature Importance...")
    plot_feature_importance(model, X_test)

    print("\n✔ Evaluation Complete.")


if __name__ == "__main__":
    run_evaluation()
