import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Fraud Detection", layout="centered")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    # Load Keras model
    model = load_model("model.keras")

    # Load pickles
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    with open("threshold.pkl", "rb") as f:
        threshold = pickle.load(f)

    return model, scaler, encoders, features, threshold

model, scaler, encoders, features, threshold = load_artifacts()

numeric_features = features["numeric"]
categorical_features = features["categorical"]

st.title("Credit Card Fraud Detection")
st.markdown("Enter transaction details below to get a fraud probability and prediction.")

with st.form("input_form"):
    st.subheader("Numeric features")
    numeric_inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(numeric_features):
        with cols[i % 2]:
            # default value 0.0 to avoid errors
            val = st.number_input(f"{col}", value=float(0.0), format="%.6f", step=1.0)
            numeric_inputs[col] = val

    st.subheader("Categorical features")
    cat_inputs = {}
    for col in categorical_features:
        # provide a text input so user can directly paste categories (same as your Colab approach)
        val = st.text_input(f"{col}", value="")
        cat_inputs[col] = val if val != "" else ""

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Prepare numeric
        X_num = np.array([ [numeric_inputs[c] for c in numeric_features] ], dtype=float)
        X_num_scaled = scaler.transform(X_num)

        # Prepare categorical (label-encoding fallback if unseen)
        cat_encoded = []
        for col in categorical_features:
            val = str(cat_inputs[col])
            le = encoders.get(col)
            if le is None:
                # if for some reason encoder missing, fallback to 0
                cat_encoded.append(0)
                continue

            # If unseen value, fallback to first known class (same as training fallback)
            if val in le.classes_:
                enc_val = int(le.transform([val])[0])
            else:
                enc_val = int(le.transform([le.classes_[0]])[0])
            cat_encoded.append(enc_val)

        final_input = np.hstack([X_num_scaled, np.array([cat_encoded])]).astype(float)

        # Predict
        prob = float(model.predict(final_input, verbose=0).ravel()[0])
        pred = 1 if prob > threshold else 0

        st.success("Prediction complete")
        st.write(f"**Fraud probability:** {prob:.4f}")
        st.write(f"**Prediction:** {'FRAUD' if pred==1 else 'LEGIT'}")
        st.write(f"**Threshold used:** {threshold}")

        # Optionally show a simple interpretation
        st.info("Tip: Lower threshold → more fraud caught (higher false positives); Higher threshold → fewer false alarms.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
