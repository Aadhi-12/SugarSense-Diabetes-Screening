import streamlit as st
import pandas as pd
import joblib

# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

model = joblib.load("Diabetes_Disease_prediction.pkl")
scaler = joblib.load("scaler.pkl")

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
st.subheader("AI-Powered Early Diabetes Risk Prediction and Screening Tool")

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
    # CREATE INPUT DATA
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
    # LOAD TRAINING DATA
    # =====================================================

    training_data = pd.read_csv(
        "diabetes_screening_data.csv"
    )

    # =====================================================
    # HANDLE ZERO / MISSING VALUES
    # =====================================================

    problematic_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for col in problematic_columns:

        training_data[col] = training_data[col].replace(
            0, pd.NA
        )

        input_data[col] = input_data[col].replace(
            0, pd.NA
        )

    # Use training-data median for missing values
    for col in problematic_columns:

        median_value = training_data[col].median()

        input_data[col] = input_data[col].fillna(
            median_value
        )

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

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

    # =====================================================
    # ONE-HOT ENCODING
    # =====================================================

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

    # =====================================================
    # MATCH EXACT MODEL FEATURES
    # =====================================================

    expected_columns = scaler.feature_names_in_

    input_encoded = input_encoded.reindex(
        columns=expected_columns,
        fill_value=0
    )

    # =====================================================
    # SCALE DATA
    # =====================================================

    input_scaled = scaler.transform(
        input_encoded
    )

    # =====================================================
    # MODEL PREDICTION
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
    # DETERMINE RISK LEVEL
    # =====================================================

    if diabetes_probability < 30:

        risk_level = "LOW RISK"
        risk_message = "Diabetes Not Detected"
        risk_icon = "🟢"

    elif diabetes_probability < 70:

        risk_level = "MODERATE RISK"
        risk_message = "Please Monitor"
        risk_icon = "🟡"

    else:

        risk_level = "HIGH RISK"
        risk_message = "Higher Diabetes Risk"
        risk_icon = "🔴"

    # =====================================================
    # PREDICTION RESULTS
    # =====================================================

    st.markdown("---")
    st.header("🎯 Prediction Results")

    if diabetes_probability < 30:

        st.success(
            f"### {risk_icon} {risk_level} - {risk_message}"
        )

    elif diabetes_probability < 70:

        st.warning(
            f"### {risk_icon} {risk_level} - {risk_message}"
        )

    else:

        st.error(
            f"### {risk_icon} {risk_level} - {risk_message}"
        )

    # =====================================================
    # PROBABILITY BREAKDOWN
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
    # PROBABILITY BAR
    # =====================================================

    st.write("### Diabetes Risk Level")

    st.progress(
        min(
            max(
                int(diabetes_probability),
                0
            ),
            100
        )
    )

    st.caption(
        "0% = lower model-estimated risk | "
        "100% = higher model-estimated risk"
    )

    # =====================================================
    # DETAILED INTERPRETATION
    # =====================================================

    st.subheader("🔎 Result Interpretation")

    if diabetes_probability < 30:

        st.write(
            "The model estimates a relatively low probability "
            "of diabetes for the entered measurements."
        )

    elif diabetes_probability < 70:

        st.write(
            "The model estimates an intermediate probability "
            "of diabetes. Some measurements may warrant "
            "additional attention or monitoring."
        )

    else:

        st.write(
            "The model estimates a relatively high probability "
            "of diabetes for the entered measurements. "
            "Professional medical evaluation is recommended."
        )

    # =====================================================
    # RISK FACTOR ANALYSIS
    # =====================================================

    st.markdown("---")
    st.subheader("⚠️ Risk Factor Analysis")

    risk_factors = []
    positive_factors = []

    # Glucose
    if glucose >= 126:

        risk_factors.append(
            f"🔴 Glucose is high ({glucose:.0f} mg/dL)"
        )

    elif glucose >= 100:

        risk_factors.append(
            f"🟡 Glucose is elevated ({glucose:.0f} mg/dL)"
        )

    else:

        positive_factors.append(
            f"🟢 Glucose is below 100 mg/dL ({glucose:.0f})"
        )

    # Blood pressure
    if blood_pressure > 80:

        risk_factors.append(
            f"🔴 Blood pressure is above 80 mm Hg "
            f"({blood_pressure:.0f} mm Hg)"
        )

    else:

        positive_factors.append(
            f"🟢 Blood pressure is not above 80 mm Hg "
            f"({blood_pressure:.0f} mm Hg)"
        )

    # BMI
    if bmi >= 30:

        risk_factors.append(
            f"🔴 BMI indicates obesity ({bmi:.1f})"
        )

    elif bmi >= 25:

        risk_factors.append(
            f"🟡 BMI is in the overweight range ({bmi:.1f})"
        )

    elif bmi >= 18.5:

        positive_factors.append(
            f"🟢 BMI is in the normal range ({bmi:.1f})"
        )

    else:

        risk_factors.append(
            f"🟡 BMI is below the normal range ({bmi:.1f})"
        )

    # Age
    if age > 45:

        risk_factors.append(
            f"🟡 Age-related risk factor ({age} years)"
        )

    else:

        positive_factors.append(
            f"🟢 Age is below 45 years ({age})"
        )

    # Pregnancies
    if pregnancies >= 6:

        risk_factors.append(
            f"🟡 Higher number of pregnancies ({pregnancies})"
        )

    # Display risk factors
    if risk_factors:

        st.warning("**Identified Risk Factors:**")

        for factor in risk_factors:

            st.markdown(
                f"- {factor}"
            )

    else:

        st.success(
            "No major risk indicators identified "
            "from the entered measurements."
        )

    # Display positive factors
    if positive_factors:

        st.success("**Positive Indicators:**")

        for factor in positive_factors:

            st.markdown(
                f"- {factor}"
            )

    # =====================================================
    # PATIENT MEASUREMENT CATEGORIES
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
            f"{pregnancies}",
            f"{diabetes_pedigree:.3f}"
        ],

        "Assessment": [

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

            "Model input"
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

        st.error(
            """
            **Suggested next steps:**

            - Consider discussing the result with a qualified healthcare professional.
            - Consider appropriate diabetes screening or testing.
            - Continue monitoring relevant health measurements.
            - Maintain a balanced diet and healthy lifestyle.
            """
        )

    else:

        st.success(
            """
            **Maintain healthy practices:**

            - Continue regular health check-ups.
            - Maintain a balanced diet.
            - Exercise regularly according to your health condition.
            - Maintain a healthy body weight.
            - Continue monitoring important health measurements.
            """
        )

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
        "Model Accuracy",
        "76.62%"
    )

    model_col3.metric(
        "Input Features",
        "18"
    )

    st.info(
        "The model uses the original clinical measurements "
        "together with engineered and one-hot-encoded features "
        "before prediction."
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
            bmi,
            diabetes_pedigree,
            age
        ]
    })

    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # FEATURE PROCESSING DETAILS
    # =====================================================

    with st.expander("🔬 View Feature Processing Details"):

        st.write(
            "The entered measurements are transformed into "
            "the same feature structure used during model training."
        )

        st.markdown(
            """
            **Engineered features:**

            - Obesity Flag → BMI ≥ 30
            - High Glucose Flag → Glucose ≥ 140 mg/dL
            - Age Group
            - BMI Category
            - Glucose Category

            These categorical features are one-hot encoded before "
            "being passed through the saved scaler and Random Forest model.
            """
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
        "768 Samples"
    )

    st.markdown("---")

    st.subheader("⚠️ Medical Disclaimer")

    st.warning(
        "This application is for educational and screening purposes only. "
        "It is NOT a medical diagnosis and should not replace advice, "
        "examination, or testing by a qualified healthcare professional."
    )
