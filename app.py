import streamlit as st
import numpy as np
from joblib import load
import pickle
model = load("model.pkl")
scaler = load("scaler.pkl")


# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival.")

# -----------------------------
# USER INPUTS
# -----------------------------
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])

sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0   # same encoding used in training

age = st.number_input("Age", 0, 100, 25)

sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)

parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)

fare = st.number_input("Fare", 0.0, 600.0, 50.0)

embarked = st.selectbox("Embarked", ["S", "C", "Q"])
if embarked == "S":
    embarked = 0
elif embarked == "C":
    embarked = 1
else:
    embarked = 2

# -----------------------------
# CREATE INPUT ARRAY
# ⚠️ ORDER MUST MATCH TRAINING
# -----------------------------
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# scale input
input_scaled = scaler.transform(input_data)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict Survival"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Passenger likely SURVIVED (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Passenger likely DID NOT survive (Probability: {probability:.2f})")
