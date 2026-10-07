import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
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

st.info(
    "Enter the patient's clinical measurements and click Predict Diabetes Risk."
)

# Sidebar
st.sidebar.header("Patient Information")

pregnancies = st.sidebar.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.sidebar.number_input(
    "Glucose (mg/dL)",
    min_value=0.0,
    value=120.0
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure (mm Hg)",
    min_value=0.0,
    value=70.0
)

skin_thickness = st.sidebar.number_input(
    "Skin Thickness (mm)",
    min_value=0.0,
    value=20.0
)

insulin = st.sidebar.number_input(
    "Insulin (μU/ml)",
    min_value=0.0,
    value=80.0
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

diabetes_pedigree = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.47
)

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

# Prediction
if st.sidebar.button("🔍 Predict Diabetes Risk"):

    # Create input DataFrame
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

    # Load original training data
    training_data = pd.read_csv("diabetes_screening_data.csv")

    # Columns where zero represents missing/invalid data
    problematic_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    # Replace zero values with missing values
    for col in problematic_columns:
        training_data[col] = training_data[col].replace(0, pd.NA)
        input_data[col] = input_data[col].replace(0, pd.NA)

    # Use the same median handling as the Colab notebook
    for col in problematic_columns:
        median_value = training_data[col].median()
        input_data[col] = input_data[col].fillna(median_value)

    # Feature engineering
    # 1. Obesity flag
    input_data["ObesityFlag"] = (
        input_data["BMI"] >= 30
    ).astype(int)

    # 2. High glucose flag
    input_data["HighGlucoseFlag"] = (
        input_data["Glucose"] >= 140
    ).astype(int)

    # 3. Age group
    input_data["AgeGroup"] = pd.cut(
        input_data["Age"],
        bins=[0, 30, 45, 60, 100],
        labels=[
            "Young",
            "Adult",
            "Middle_Age",
            "Senior"
        ]
    )

    # 4. BMI category
    input_data["BMI_Category"] = pd.cut(
        input_data["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[
            "Underweight",
            "Normal",
            "Overweight",
            "Obese"
        ]
    )

    # 5. Glucose category
    input_data["GlucoseCategory"] = pd.cut(
        input_data["Glucose"],
        bins=[0, 100, 126, 200],
        labels=[
            "Normal",
            "Prediabetes",
            "High"
        ]
    )

    # One-hot encode categorical features
    categorical_columns = [
        "AgeGroup",
        "BMI_Category",
        "GlucoseCategory"
    ]

    input_encoded = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        drop_first=True
    )

    # Use the exact feature names expected by the saved scaler
    expected_columns = scaler.feature_names_in_

    input_encoded = input_encoded.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # Scale input using the saved scaler
    input_scaled = scaler.transform(input_encoded)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.header("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Higher Diabetes Risk")
    else:
        st.success("✅ Lower Diabetes Risk")

    st.metric(
        "Diabetes Probability",
        f"{probability * 100:.2f}%"
    )

# About section
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
