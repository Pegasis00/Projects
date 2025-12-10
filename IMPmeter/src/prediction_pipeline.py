import joblib
import pandas as pd
import numpy as np
import sys
import os

# Make sure src is discoverable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # /src
IMPMETER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(IMPMETER_DIR)
sys.path.append(CURRENT_DIR)

from feature_engineering import prepare_single_input
from config import MODEL_PATH

# =====================================================
# LOAD MODEL ONE TIME (efficient for Streamlit)
# =====================================================
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


# =====================================================
# MAIN PREDICTION FUNCTION
# =====================================================
def predict_user_score(input_dict: dict):
    """
    1) Takes raw input from UI
    2) Converts into DataFrame
    3) Performs same feature engineering as training
    4) Runs ML model prediction
    """

    df = pd.DataFrame([input_dict])

    # FE for single row
    X = prepare_single_input(df)

    # Predict score
    pred = model.predict(X)[0]

    return float(pred)
