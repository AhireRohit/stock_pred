import streamlit as st
import pandas as pd
import sys
import tensorflow.keras as keras
sys.modules['keras'] = keras

import numpy as np
import pickle
from datetime import datetime
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K
import warnings
# ──────────────────────────
# 1) Load & cache artifacts
# ──────────────────────────
@st.cache(allow_output_mutation=True)
def load_artifacts():
    with open("dual_lstm_model.pkl","rb")   as f: model      = pickle.load(f)
    with open("scaler_X_raw.pkl","rb")      as f: scaler_Xr  = pickle.load(f)
    with open("scaler_X_eng.pkl","rb")      as f: scaler_Xe  = pickle.load(f)
    with open("scaler_y.pkl","rb")          as f: scaler_y   = pickle.load(f)
    with open("df_feature_history.pkl","rb")as f: df_feat    = pickle.load(f)
    with open("trend_thresholds.pkl","rb")  as f: (up_th, down_th) = pickle.load(f)
    return model, scaler_Xr, scaler_Xe, scaler_y, df_feat, up_th, down_th

model, scaler_X_raw, scaler_X_eng, scaler_y, df_feat, up_th, down_th = load_artifacts()


# ──────────────────────────
# 2) Helpers from your notebook
# ──────────────────────────
raw_features = ["A1","A2","B1","B2","C1","C2"]
eng_features = [
    "A2_A1_diff","B2_B1_diff","C2_C1_diff",
    "ret_1","ret_5","volatility",
    "price_norm","Y2_lag1","expiry_pressure","client_score"
]

def make_one_row(user_dict):
    r = pd.Series(user_dict).copy()
    # engineered diffs
    r["A2_A1_diff"] = r["A2"] - r["A1"]
    r["B2_B1_diff"] = r["B2"] - r["B1"]
    r["C2_C1_diff"] = r["C2"] - r["C1"]
    # expiry pressure
    days = (pd.to_datetime(r["Expiry_date"]).normalize()
          - pd.to_datetime(r["Date"]).normalize()).days
    r["expiry_pressure"] = 1.0/(1+max(days,0))
    # placeholders
    for c in ["ret_1","ret_5","volatility","price_norm","Y2_lag1","client_score"]:
        r[c] = 0.0
    return r

def pct_to_trend(pct):
    if pct > up_th:    return "Bullish"
    if pct < down_th:  return "Bearish"
    return "Consolidation"

def predict_30min_ahead(user_dict, df_hist, time_steps=6):
    # build & name row
    row = make_one_row(user_dict)
    ts  = pd.to_datetime(f"{user_dict['Date']} {user_dict['Time']}")
    row.name = ts

    # get last (time_steps−1) bars < ts
    hist = df_hist.sort_index()
    prev = hist[hist.index < ts].iloc[-(time_steps-1):]
    if len(prev) < time_steps-1:
        st.error(f"Not enough history rows (need {time_steps-1}, got {len(prev)})")
        return None, None

    seq = pd.concat([prev, row.to_frame().T])

    # raw branch
    Xr = scaler_X_raw.transform(seq[raw_features])\
           .reshape(1, time_steps, len(raw_features))
    # engineered branch
    Xe = scaler_X_eng.transform(seq[eng_features])\
           .reshape(1, time_steps, len(eng_features))

    # forward + invert
    pct_s = model.predict([Xr, Xe])[0,0]
    pct   = scaler_y.inverse_transform([[pct_s]])[0,0] * 100

    return pct_to_trend(pct), round(pct,3)


# ──────────────────────────
# 3) Streamlit UI
# ──────────────────────────
st.title("📈 30‑Min Ahead Market Trend Predictor")

expiry = st.date_input("Expiry Date")
date   = st.date_input("Date")
time   = st.time_input("Time")
price  = st.number_input("PRICE", value=0)
A1,A2  = st.number_input("A1", value=0), st.number_input("A2", value=0)
B1,B2  = st.number_input("B1", value=0), st.number_input("B2", value=0)
C1,C2  = st.number_input("C1", value=0), st.number_input("C2", value=0)

if st.button("Submit"):
    user_in = {
      "Expiry_date": expiry.strftime("%Y-%m-%d"),
      "Date"       : date.strftime("%Y-%m-%d"),
      "Time"       : time.strftime("%H:%M:%S"),
      "PRICE"      : price,
      "A1": A1, "A2": A2, "B1": B1, "B2": B2, "C1": C1, "C2": C2
    }
    trend, pct = predict_30min_ahead(user_in, df_feat, time_steps=6)
    if trend is not None:
        st.metric("Predicted Trend", trend)
        st.metric("Expected % Change", f"{pct:+.3f}%")
