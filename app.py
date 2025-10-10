import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load('artifacts/xgb_model.pkl')
scaler = joblib.load('artifacts/scaler.pkl')

# Streamlit App
st.title("🔥 Calories Burnt Predictor 🔥")
st.write("Predict the number of calories burnt during a workout based on personal data and workout stats.")

# User Input
st.sidebar.header("Enter Your Details")
age = st.sidebar.number_input("Age (years)", min_value=10, max_value=100, value=25)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=175)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
body_temp = st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=36.6)
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=50, max_value=200, value=80)

# Encode Gender
gender_encoded = 0 if gender.lower() == "male" else 1

# Prepare input dataframe
input_data = pd.DataFrame([{
    'Age': age,
    'Height': height,
    'Gender': gender_encoded,
    'Body_Temp': body_temp,
    'Heart_Rate': heart_rate
}])

# Match scaler features
input_data = input_data[scaler.feature_names_in_]

# Scale features
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Calories"):
    predicted_calories = model.predict(input_scaled)[0]
    st.success(f"🔥 Estimated Calories Burnt: {predicted_calories:.2f} kcal 🔥")
