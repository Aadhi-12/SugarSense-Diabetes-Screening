import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL, SCALER AND COLUMNS
# =========================================================

model = joblib.load("Diabetes_Disease_prediction.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SugarSense - Diabetes Screening",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🩺 SugarSense")

st.subheader(
    "AI-Powered Early Diabetes Risk Prediction and Screening Tool"
)

st.info(
    "Enter the patient's clinical measurements in the sidebar "
    "and click **Predict Diabetes Risk**."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("👤 Patient Information")
st.sidebar.subheader("Clinical Measurements")


pregnancies = st.sidebar.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)


glucose = st.sidebar.number_input(
    "Glucose (mg/dL)",
    min_value=0.0,
    max_value=300.0,
    value=120.0,
    step=1.0
)


blood_pressure = st.sidebar.number_input(
    "Blood Pressure (mm Hg)",
    min_value=0.0,
    max_value=150.0,
    value=70.0,
    step=1.0
)


skin_thickness = st.sidebar.number_input(
    "Skin Thickness (mm)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=1.0
)


insulin = st.sidebar.number_input(
    "Insulin (μU/ml)",
    min_value=0.0,
    max_value=900.0,
    value=80.0,
    step=1.0
)


bmi = st.sidebar.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)


diabetes_pedigree = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.47,
    step=0.01
)


age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30,
    step=1
)


st.sidebar.markdown("---")


predict_button = st.sidebar.button(
    "🔍 Predict Diabetes Risk",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # =====================================================
    # INPUT DATA
    # =====================================================

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


    # =====================================================
    # SCALE INPUT
    # =====================================================

    input_scaled = scaler.transform(
        input_data.to_numpy()
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(
        input_scaled
    )[0]


    probability = model.predict_proba(
        input_scaled
    )[0]


    non_diabetes_probability = probability[0] * 100
    diabetes_probability = probability[1] * 100


    # =====================================================
    # RISK LEVEL
    # =====================================================

    if diabetes_probability < 30:

        risk_level = "LOW RISK"

    elif diabetes_probability < 70:

        risk_level = "MODERATE RISK"

    else:

        risk_level = "HIGH RISK"


    # =====================================================
    # RESULTS
    # =====================================================

    st.markdown("---")

    st.header("🎯 Prediction Results")


    if prediction == 1:

        if diabetes_probability >= 70:

            st.error(
                "### 🔴 HIGH RISK - Higher Diabetes Risk"
            )

        else:

            st.warning(
                "### 🟡 MODERATE RISK - Diabetes Risk"
            )

    else:

        if diabetes_probability < 30:

            st.success(
                "### 🟢 LOW RISK - Lower Diabetes Risk"
            )

        else:

            st.warning(
                "### 🟡 MODERATE RISK - Please Monitor"
            )


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.subheader("📊 Probability Breakdown")


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Non-Diabetic Probability",
        f"{non_diabetes_probability:.1f}%"
    )


    col2.metric(
        "Diabetes Probability",
        f"{diabetes_probability:.1f}%"
    )


    col3.metric(
        "Risk Level",
        risk_level
    )


    # =====================================================
    # PROGRESS BAR
    # =====================================================

    st.subheader("📈 Diabetes Risk Level")


    st.progress(
        int(diabetes_probability)
    )


    st.caption(
        "0% = lower model-estimated risk | "
        "100% = higher model-estimated risk"
    )


    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.subheader("🔎 Result Interpretation")


    if diabetes_probability < 30:

        st.success(
            "The model estimates a relatively low probability "
            "of diabetes for the entered measurements."
        )

    elif diabetes_probability < 70:

        st.warning(
            "The model estimates an intermediate probability "
            "of diabetes. Some measurements may require "
            "additional monitoring."
        )

    else:

        st.error(
            "The model estimates a relatively high probability "
            "of diabetes for the entered measurements. "
            "Professional medical evaluation is recommended."
        )


    # =====================================================
    # RISK FACTORS
    # =====================================================

    st.markdown("---")

    st.subheader("⚠️ Risk Factor Analysis")


    risk_factors = []
    positive_factors = []


    if glucose >= 126:

        risk_factors.append(
            f"🔴 High glucose level: {glucose:.0f} mg/dL"
        )

    elif glucose >= 100:

        risk_factors.append(
            f"🟡 Elevated glucose level: {glucose:.0f} mg/dL"
        )

    else:

        positive_factors.append(
            f"🟢 Glucose is below 100 mg/dL: {glucose:.0f}"
        )


    if blood_pressure > 80:

        risk_factors.append(
            f"🔴 Blood pressure is above 80 mm Hg: "
            f"{blood_pressure:.0f} mm Hg"
        )

    else:

        positive_factors.append(
            f"🟢 Blood pressure is not above 80 mm Hg: "
            f"{blood_pressure:.0f} mm Hg"
        )


    if bmi >= 30:

        risk_factors.append(
            f"🔴 BMI indicates obesity: {bmi:.1f}"
        )

    elif bmi >= 25:

        risk_factors.append(
            f"🟡 BMI is in the overweight range: {bmi:.1f}"
        )

    elif bmi >= 18.5:

        positive_factors.append(
            f"🟢 BMI is in the normal range: {bmi:.1f}"
        )

    else:

        risk_factors.append(
            f"🟡 BMI is below the normal range: {bmi:.1f}"
        )


    if age > 45:

        risk_factors.append(
            f"🟡 Age-related risk factor: {age} years"
        )

    else:

        positive_factors.append(
            f"🟢 Age is below 45 years: {age}"
        )


    if pregnancies >= 6:

        risk_factors.append(
            f"🟡 Higher number of pregnancies: {pregnancies}"
        )


    if risk_factors:

        st.warning("**Identified Risk Factors:**")

        for factor in risk_factors:

            st.markdown(
                f"- {factor}"
            )


    if positive_factors:

        st.success("**Positive Indicators:**")

        for factor in positive_factors:

            st.markdown(
                f"- {factor}"
            )


    # =====================================================
    # CLINICAL ANALYSIS
    # =====================================================

    st.markdown("---")

    st.subheader("🧪 Clinical Measurement Analysis")


    analysis_data = pd.DataFrame({

        "Measurement": [
            "Glucose",
            "Blood Pressure",
            "BMI",
            "Age",
            "Pregnancies",
            "Diabetes Pedigree"
        ],

        "Value": [
            f"{glucose:.1f} mg/dL",
            f"{blood_pressure:.1f} mm Hg",
            f"{bmi:.1f}",
            f"{age} years",
            pregnancies,
            f"{diabetes_pedigree:.3f}"
        ],

        "Category": [

            (
                "High"
                if glucose >= 126
                else "Elevated"
                if glucose >= 100
                else "Normal"
            ),

            (
                "Above 80"
                if blood_pressure > 80
                else "Not above 80"
            ),

            (
                "Obese"
                if bmi >= 30
                else "Overweight"
                if bmi >= 25
                else "Normal"
                if bmi >= 18.5
                else "Underweight"
            ),

            (
                "Higher age group"
                if age > 45
                else "Lower age group"
            ),

            (
                "Higher"
                if pregnancies >= 6
                else "Not high"
            ),

            "Model Input"
        ]
    })


    st.dataframe(
        analysis_data,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    st.markdown("---")

    st.subheader("💡 Recommendations")


    if prediction == 1:

        st.error("""
**Recommended next steps:**

- Discuss the result with a qualified healthcare professional.
- Consider appropriate diabetes screening/testing.
- Continue monitoring relevant health measurements.
- Maintain a balanced diet and healthy lifestyle.
        """)

    else:

        st.success("""
**Maintain healthy practices:**

- Continue regular health check-ups.
- Maintain a balanced diet.
- Exercise regularly according to your health condition.
- Maintain a healthy body weight.
- Continue monitoring important health measurements.
        """)


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.markdown("---")

    st.subheader("🤖 Model Information")


    model_col1, model_col2, model_col3 = st.columns(3)


    model_col1.metric(
        "Model",
        "Random Forest"
    )


    model_col2.metric(
        "Accuracy",
        "76.62%"
    )


    model_col3.metric(
        "Features",
        "8"
    )


    st.info(
        "The model uses the original clinical measurements "
        "for diabetes risk prediction."
    )


    # =====================================================
    # PATIENT INPUT SUMMARY
    # =====================================================

    st.markdown("---")

    st.subheader("📋 Patient Input Summary")


    summary_data = pd.DataFrame({

        "Measurement": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree Function",
            "Age"
        ],

        "Value": [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            diabetes_pedigree,
            bmi,
            age
        ]
    })


    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# INITIAL PAGE
# =========================================================

else:

    st.markdown("---")

    st.info(
        "👈 Enter patient information in the sidebar "
        "and click **Predict Diabetes Risk**."
    )


    st.subheader("📌 About SugarSense")


    st.write(
        "SugarSense is an educational machine-learning application "
        "designed for early diabetes risk screening."
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Model",
        "Random Forest"
    )


    col2.metric(
        "Accuracy",
        "76.62%"
    )


    col3.metric(
        "Dataset",
        "768 Patients"
    )


    st.markdown("---")

    st.subheader("⚠️ Medical Disclaimer")


    st.warning(
        "This application is for educational and screening purposes only. "
        "It is NOT a medical diagnosis and should not replace advice, "
        "examination, or testing by a qualified healthcare professional."
    )
