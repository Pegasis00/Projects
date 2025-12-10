# 🛒 AI-Powered Impulse Buy Risk Predictor  
### Predict how impulsively a user will shop — using behavioral, psychological & transactional features.

This project is a complete end-to-end Machine Learning pipeline + Streamlit web application that predicts a user's **Impulse Buy Risk Score (0–100)**.

It uses:
- Browsing behavior  
- Purchase history  
- Discounts used  
- Time to purchase  
- Psychology + mood data  
- Synthetic realistic datasets  
- Gradient-boosted ML models (XGBoost / LightGBM)  

---

## 🚀 Features  
### ✔ Synthetic Dataset Generator  
Creates highly realistic online-shopping behavioral data.

### ✔ Target V2 Scoring  
A nonlinear scoring system designed to mimic real impulsive buying behavior.

### ✔ Feature Engineering Pipeline  
Numerical scaling + OneHot encoding + full preprocessing pipeline.

### ✔ Multiple Models Trained  
- Linear Regression  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost (optional)

Best-performing model is auto-saved as:

IMP_Pred_project/
├── app/
│ ├── app.py # Streamlit Frontend
│
├── src/
│ ├── prediction_pipeline.py # Model load + single prediction
│ ├── feature_engineering.py # Preprocessing logic
│ ├── train_model.py # Train + save best model
│ ├── evaluate_models.py # Evaluation + visualizations
│ ├── data_merge.py # Merge raw → final dataset
│
├── models/
│ ├── best_model.pkl
│
├── data/
│ ├── processed/
│ ├── final_user_dataset.csv
│
├── requirements.txt
├── README.md


---

## 🚀 Deployment (Streamlit Cloud)

1. Push this folder structure to GitHub  
2. Go to **https://share.streamlit.io/**  
3. Click **Deploy → Connect GitHub**  
4. Choose repo & branch  
5. Set app file:



6. Deploy 🎉  

---

## ⚙ Local Development

### Install dependencies:
```bash
pip install -r requirements.txt
python -m src.train_model

streamlit run app/app.py
"# Projects" 
