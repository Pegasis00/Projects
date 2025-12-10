import os
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================
RAW_DIR = "data/raw_a2_fast"
OUTPUT_DIR = "data/processed"
FINAL_OUTPUT_PATH = f"{OUTPUT_DIR}/final_user_dataset.csv"


# =========================================================
# 1. LOAD RAW DATA
# =========================================================
def load_raw():
    user_df = pd.read_csv(f"{RAW_DIR}/user_profile.csv")
    browsing_df = pd.read_csv(f"{RAW_DIR}/browsing_logs.csv")
    trans_df = pd.read_csv(f"{RAW_DIR}/transactions.csv")
    survey_df = pd.read_csv(f"{RAW_DIR}/psychology_survey.csv")
    return user_df, browsing_df, trans_df, survey_df


# =========================================================
# 2. AGGREGATE BROWSING LOGS
# =========================================================
def aggregate_browsing_logs(df: pd.DataFrame):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_late_night"] = df["hour"].isin([22,23,0,1,2,3]).astype(int)

    agg = df.groupby("user_id").agg(
        total_sessions=("session_id", "count"),
        num_product_page_visits=("page_type", lambda x: (x == "product").sum()),
        num_cart_visits=("page_type", lambda x: (x == "cart").sum()),
        num_checkout_visits=("page_type", lambda x: (x == "checkout").sum()),
        avg_time_on_product=("time_spent_seconds", lambda x: x.mean()),
        late_night_session_ratio=("is_late_night", "mean"),
        device_preference=("device_type", lambda x: x.mode()[0] if len(x.mode()) else "mobile"),
    )

    agg["avg_time_on_product"] = agg["avg_time_on_product"].fillna(0)

    return agg.reset_index()


# =========================================================
# 3. AGGREGATE TRANSACTIONS
# =========================================================
def aggregate_transactions(df: pd.DataFrame):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["first_view_time"] = pd.to_datetime(df["first_view_time"])

    df["minutes_to_purchase"] = (
        df["timestamp"] - df["first_view_time"]
    ).dt.total_seconds() / 60

    df["minutes_to_purchase"] = df["minutes_to_purchase"].clip(lower=0)

    agg = df.groupby("user_id").agg(
        total_purchases=("transaction_id", "count"),
        total_spent=("amount", "sum"),
        avg_purchase_value=("amount", "mean"),
        avg_discount_used=("discount_applied", "mean"),
        impulse_purchase_ratio=("is_impulse_purchase", "mean"),
        past_impulse_purchases=("is_impulse_purchase", "sum"),
        avg_minutes_to_purchase=("minutes_to_purchase", "mean"),
    )

    return agg.reset_index()


# =========================================================
# 4. MERGE ALL DATASETS
# =========================================================
def merge_all(user_df, browsing_agg, trans_agg, survey_df):

    merged = (
        user_df
        .merge(browsing_agg, on="user_id", how="left")
        .merge(trans_agg, on="user_id", how="left")
        .merge(survey_df, on="user_id", how="left")
    )

    # fill missing for users with no transactions
    trans_fill = [
        "total_purchases", "total_spent", "avg_purchase_value",
        "avg_discount_used", "impulse_purchase_ratio",
        "past_impulse_purchases", "avg_minutes_to_purchase"
    ]

    merged[trans_fill] = merged[trans_fill].fillna(0)

    # fill missing browsing
    merged["avg_time_on_product"] = merged["avg_time_on_product"].fillna(0)
    merged["device_preference"] = merged["device_preference"].fillna("mobile")

    return merged


# =========================================================
# 5. TARGET V2 — REALISTIC IMPULSE SCORE
# =========================================================
def compute_target(df: pd.DataFrame) -> pd.DataFrame:

    # Normalize core signals
    impulse_ratio = df["impulse_purchase_ratio"].clip(0, 1)
    discount_norm = (df["avg_discount_used"] / 50).clip(0, 1)
    late_night = df["late_night_session_ratio"].clip(0, 1)
    stress = (df["stress_level"] / 10).clip(0, 1)

    revisit_ratio = (
        df["num_product_page_visits"] / df["total_sessions"].clip(1)
    ).clip(0, 1)

    # time pressure = buy quickly → more impulsive
    time_pressure = np.exp(-df["avg_minutes_to_purchase"].clip(1, 500) / 200)

    # persona effects
    persona_map = {
        "impulse_buyer": 1.25,
        "deal_hunter": 1.10,
        "steady_buyer": 0.90,
        "window_shopper": 0.70,
        "premium_user": 0.80,
        "cautious_low_income": 1.05
    }
    persona_factor = df["persona"].map(persona_map)

    # mood effects
    mood_map = {
        "Happy": -0.05,
        "Neutral": 0.0,
        "Sad": 0.10,
        "Anxious": 0.20
    }
    mood_factor = df["mood_last_week"].map(mood_map)

    # nonlinear impulse score
    score = (
        40 * (impulse_ratio ** 1.8)
        + 20 * np.sqrt(discount_norm)
        + 15 * (late_night ** 1.5)
        + 12 * (stress ** 2)
        + 10 * revisit_ratio
        + 10 * time_pressure
        + 8 * persona_factor
        + 5 * mood_factor
        + np.random.normal(0, 5, len(df))   # realistic noise
    )

    df["impulse_buy_score"] = np.clip(score, 0, 100)
    return df


# =========================================================
# 6. CREATE FINAL DATASET
# =========================================================
def create_final_dataset():
    print("[INFO] Loading raw data...")
    user_df, browsing_df, trans_df, survey_df = load_raw()

    print("[INFO] Aggregating browsing logs...")
    browsing_agg = aggregate_browsing_logs(browsing_df)

    print("[INFO] Aggregating transactions...")
    trans_agg = aggregate_transactions(trans_df)

    print("[INFO] Merging datasets...")
    merged = merge_all(user_df, browsing_agg, trans_agg, survey_df)

    print("[INFO] Computing Target V2...")
    final_df = compute_target(merged)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)

    print(f"[SUCCESS] Final dataset stored at: {FINAL_OUTPUT_PATH}")
    print(f"[SHAPE] {final_df.shape}")
    print(final_df.head())

    return final_df







if __name__ == "__main__":
    create_final_dataset()
