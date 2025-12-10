import os

# -------------------
# GLOBAL CONFIG
# -------------------

RANDOM_SEED = 42

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FINAL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "final_user_dataset.csv")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# Target column
TARGET_COL = "impulse_buy_score"

# -------------------
# FEATURE GROUPS
# -------------------

NUMERIC_FEATURES = [
    "age",
    "monthly_income",
    "account_age_days",
    "avg_time_on_product",
    "total_sessions",
    "num_product_page_visits",
    "num_cart_visits",
    "num_checkout_visits",
    "late_night_session_ratio",
    "total_purchases",
    "total_spent",
    "avg_purchase_value",
    "avg_discount_used",
    "impulse_purchase_ratio",
    "past_impulse_purchases",
    "avg_minutes_to_purchase",
    "stress_level",
    "saving_habit_score"
]

CATEGORICAL_FEATURES = [
    "gender",
    "city",
    "default_payment_method",
    "mood_last_week",
    "device_preference"
]
