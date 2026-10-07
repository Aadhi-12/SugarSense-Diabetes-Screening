import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("Diabetes_Disease_prediction.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="SugarSense - Diabetes Screening",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 SugarSense")
st.subheader("AI-Powered Early Diabetes Risk Prediction and Screening Tool")

st.info("Enter the patient's clinical measurements and click Predict Diabetes Risk.")

# Sidebar inputs
st.sidebar.header("Patient Information")

pregnancies = st.sidebar.number_input(
    "Pregnancies", min_value=0, max_value=20, value=1
)

glucose = st.sidebar.number_input(
    "Glucose (mg/dL)", min_value=0.0, value=120.0
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure (mm Hg)", min_value=0.0, value=70.0
)

skin_thickness = st.sidebar.number_input(
    "Skin Thickness (mm)", min_value=0.0, value=20.0
)

insulin = st.sidebar.number_input(
    "Insulin (μU/ml)", min_value=0.0, value=80.0
)

bmi = st.sidebar.number_input(
    "BMI", min_value=0.0, value=25.0
)

diabetes_pedigree = st.sidebar.number_input(
    "Diabetes Pedigree Function", min_value=0.0, value=0.47
)

age = st.sidebar.number_input(
    "Age", min_value=1, max_value=120, value=30
)

if st.sidebar.button("🔍 Predict Diabetes Risk"):

    input_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    input_scaled = scaler.transform(input_data.to_numpy())

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.header("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Higher Diabetes Risk")
    else:
        st.success("✅ Lower Diabetes Risk")

    st.metric(
        "Diabetes Probability",
        f"{probability * 100:.2f}%"
    )

st.markdown("---")

st.subheader("About SugarSense")

st.write(
    "SugarSense is an educational machine-learning application "
    "designed for early diabetes risk screening."
)

st.warning(
    "⚠️ This application is for educational and screening purposes only. "
    "It is NOT a medical diagnosis and should not replace professional medical advice."
)
