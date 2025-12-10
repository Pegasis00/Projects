import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import FINAL_DATA_PATH, TARGET_COL


# =====================================================
# 1. DEFINE FEATURES
# =====================================================

NUM_FEATURES = [
    "age",
    "monthly_income",
    "account_age_days",
    "total_sessions",
    "num_product_page_visits",
    "avg_time_on_product",
    "late_night_session_ratio",
    "total_purchases",
    "total_spent",
    "avg_purchase_value",
    "avg_discount_used",
    "impulse_purchase_ratio",
    "avg_minutes_to_purchase",
    "stress_level",
    "saving_habit_score",
]

CAT_FEATURES = [
    "gender",
    "city",
    "default_payment_method",
    "device_preference",
    "mood_last_week",
    "persona"
]


# =====================================================
# 2. PREPROCESSORS
# =====================================================

def get_preprocessor():
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
        ]
    )

    return preprocessor


# =====================================================
# 3. TRAINING PIPELINE PREPARATION
# =====================================================

def prepare_for_training(test_size=0.2, random_state=42):
    df = pd.read_csv(FINAL_DATA_PATH)

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET_COL]

    preprocessor = get_preprocessor()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return preprocessor, X_train, X_test, y_train, y_test


# =====================================================
# 4. SINGLE INPUT PROCESSING (FOR STREAMLIT)
# =====================================================

def prepare_single_input(df: pd.DataFrame):
    """
    Ensures:
    - All numeric features are float
    - All categorical features are strings
    - Missing columns are added
    - Order matches training dataset
    """

    # Add missing columns
    for col in NUM_FEATURES + CAT_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Numeric → float
    df[NUM_FEATURES] = df[NUM_FEATURES].astype(float)

    # Categorical → string
    df[CAT_FEATURES] = df[CAT_FEATURES].astype(str)

    # Enforce correct order
    df = df[NUM_FEATURES + CAT_FEATURES]

    return df

