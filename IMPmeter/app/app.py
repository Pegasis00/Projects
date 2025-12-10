import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))       
IMPMETER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(IMPMETER_DIR, "src")

sys.path.append(IMPMETER_DIR)
sys.path.append(SRC_DIR)

from prediction_pipeline import predict_user_score




# =====================================================
# STREAMLIT PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Impulse Buy Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(90deg, #ff5f6d, #ffc371);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-heading {
            font-size: 20px;
            font-weight: 500;
            color: #444;
            margin-top: -15px;
        }
        .score-box {
            padding: 20px;
            border-radius: 10px;
            background: #ffffff10;
            backdrop-filter: blur(10px);
            border: 1px solid #ffffff30;
            text-align: center;
        }
        .risk-high {color: #ff2e2e; font-weight: 700; font-size: 20px;}
        .risk-medium {color: #ffa600; font-weight: 700; font-size: 20px;}
        .risk-low {color: #00c853; font-weight: 700; font-size: 20px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>Impulse Buy Risk Predictor 🔮</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-heading'>AI-powered scoring based on user behavior, psychology, and spending patterns.</p>", unsafe_allow_html=True)

st.write("---")

# =====================================================
# SIDEBAR FORM
# =====================================================
st.sidebar.header("Enter User Details")

with st.sidebar.form("user_form"):
    st.subheader("👤 Profile")

    age = st.slider("Age", 18, 60, 28)
    monthly_income = st.number_input("Monthly Income (₹)", 5000, 300000, 45000)
    account_age_days = st.number_input("Account Age (Days)", 1, 2000, 365)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    city = st.selectbox("City", ["Mumbai","Pune","Delhi","Bengaluru","Hyderabad","Chennai","Kolkata"])
    payment = st.selectbox("Default Payment Method", ["UPI","Card","COD","Wallet"])

    st.subheader("📱 Browsing Behavior")
    total_sessions = st.number_input("Total Sessions (90 days)", 0, 200, 40)
    num_product_page_visits = st.number_input("Product Page Visits", 0, 200, 25)
    late_night_session_ratio = st.slider("Late Night Session Ratio", 0.0, 1.0, 0.2)
    device_preference = st.selectbox("Primary Device", ["mobile","desktop","tablet"])
    avg_time_on_product = st.number_input("Avg Time on Product (sec)", 1, 600, 50)

    st.subheader("🛍️ Purchases")
    total_purchases = st.number_input("Total Purchases", 0, 50, 6)
    total_spent = st.number_input("Total Spent (₹)", 0, 500000, 8000)
    avg_purchase_value = st.number_input("Avg Purchase Value (₹)", 0, 50000, 1200)
    avg_discount_used = st.number_input("Avg Discount Used (%)", 0, 50, 12)
    impulse_purchase_ratio = st.slider("Impulse Purchase Ratio", 0.0, 1.0, 0.25)
    avg_minutes_to_purchase = st.number_input("Avg Minutes To Purchase", 1, 500, 18)

    st.subheader("🧠 Psychology")
    stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
    mood = st.selectbox("Mood Last Week", ["Happy","Neutral","Sad","Anxious"])
    saving_habit_score = st.slider("Saving Habit Score (1–5)", 1, 5, 3)

    submitted = st.form_submit_button("Predict 🔮")


# =====================================================
# PREDICTION + DISPLAY
# =====================================================
if submitted:

    user_input = {
        "age": age,
        "monthly_income": monthly_income,
        "account_age_days": account_age_days,
        "gender": gender,
        "city": city,
        "default_payment_method": payment,
        "total_sessions": total_sessions,
        "num_product_page_visits": num_product_page_visits,
        "late_night_session_ratio": late_night_session_ratio,
        "device_preference": device_preference,
        "avg_time_on_product": avg_time_on_product,
        "total_purchases": total_purchases,
        "total_spent": total_spent,
        "avg_purchase_value": avg_purchase_value,
        "avg_discount_used": avg_discount_used,
        "impulse_purchase_ratio": impulse_purchase_ratio,
        "avg_minutes_to_purchase": avg_minutes_to_purchase,
        "stress_level": stress_level,
        "mood_last_week": mood,
        "saving_habit_score": saving_habit_score,
    }

    score = predict_user_score(user_input)

    st.markdown("## 🎯 Predicted Impulse Buy Score")
    st.markdown(f"<div class='score-box'><h1>{score:.2f} / 100</h1></div>", unsafe_allow_html=True)

    if score >= 70:
        st.markdown("<p class='risk-high'>⚠ High Impulsiveness — This user is likely to make spontaneous purchases.</p>", unsafe_allow_html=True)
    elif score >= 40:
        st.markdown("<p class='risk-medium'>🟡 Medium Impulsiveness — Mixed buying behavior.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='risk-low'>🟢 Low Impulsiveness — Controlled & rational shopper.</p>", unsafe_allow_html=True)


st.write("---")
st.caption("Built with ❤️ by Pegasus · Powered by Machine Learning")
