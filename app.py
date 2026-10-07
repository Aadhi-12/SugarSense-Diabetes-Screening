import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SugarSense - Diabetes Screening",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD SAVED FILES
# =========================================================

@st.cache_resource
def load_files():

    model = joblib.load("Diabetes_Disease_prediction.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")

    return model, scaler, columns


try:

    model, scaler, columns = load_files()

except Exception as e:

    st.error("❌ Unable to load the model files.")
    st.write(e)
    st.stop()


# =========================================================
# PAGE STYLE
# =========================================================

st.markdown("""
<style>

.main {
    padding: 1rem 2rem;
}

h1 {
    color: #2c3e50;
}

</style>
""", unsafe_allow_html=True)


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
    min_value=21,
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
    # ORIGINAL 8 FEATURES
    # =====================================================

    feature_values = {

        "Pregnancies": pregnancies,

        "Glucose": glucose,

        "BloodPressure": blood_pressure,

        "SkinThickness": skin_thickness,

        "Insulin": insulin,

        "BMI": bmi,

        "DiabetesPedigreeFunction": diabetes_pedigree,

        "Age": age
    }


    # =====================================================
    # ENGINEERED FEATURES
    # =====================================================

    # Obesity flag
    obesity = int(bmi >= 30)

    # Glucose × BMI
    glucose_bmi = glucose * bmi

    # Age × Glucose
    age_glucose = age * glucose

    # Additional useful engineered features
    high_glucose = int(glucose >= 126)

    high_bmi = int(bmi >= 30)


    feature_values["Obesity"] = obesity

    feature_values["Glucose_BMI"] = glucose_bmi

    feature_values["Age_Glucose"] = age_glucose

    feature_values["High_Glucose"] = high_glucose

    feature_values["High_BMI"] = high_bmi


    # =====================================================
    # CREATE INPUT USING SAVED COLUMN ORDER
    # =====================================================

    final_values = []

    unknown_features = []


    for column in columns:

        # Exact match
        if column in feature_values:

            final_values.append(
                feature_values[column]
            )

            continue


        # -------------------------------------------------
        # Handle different possible column naming styles
        # -------------------------------------------------

        normalized = (
            str(column)
            .lower()
            .replace("_", "")
            .replace(" ", "")
            .replace("-", "")
        )


        if normalized in [
            "pregnancies",
            "pregnancy"
        ]:

            value = pregnancies


        elif normalized == "glucose":

            value = glucose


        elif normalized in [
            "bloodpressure",
            "bp"
        ]:

            value = blood_pressure


        elif normalized in [
            "skinthickness",
            "skin"
        ]:

            value = skin_thickness


        elif normalized == "insulin":

            value = insulin


        elif normalized == "bmi":

            value = bmi


        elif normalized in [
            "diabetespedigreefunction",
            "diabetespedigree",
            "dpf"
        ]:

            value = diabetes_pedigree


        elif normalized == "age":

            value = age


        elif normalized in [
            "obesity",
            "obesityflag",
            "isobese"
        ]:

            value = obesity


        elif normalized in [
            "glucosebmi",
            "glucosebminteraction",
            "glucosebmiproduct"
        ]:

            value = glucose_bmi


        elif normalized in [
            "ageglucose",
            "ageglucoseinteraction",
            "ageglucoseproduct"
        ]:

            value = age_glucose


        elif normalized in [
            "highglucose",
            "highglucoseflag"
        ]:

            value = high_glucose


        elif normalized in [
            "highbmi",
            "highbmiflag"
        ]:

            value = high_bmi


        else:

            unknown_features.append(column)

            value = 0


        final_values.append(value)


    # =====================================================
    # INPUT DATAFRAME
    # =====================================================

    input_data = pd.DataFrame(
        [final_values],
        columns=columns
    )


    # =====================================================
    # CHECK FEATURE COUNT
    # =====================================================

    if len(columns) != scaler.n_features_in_:

        st.error(
            f"❌ Feature mismatch between columns.pkl "
            f"and scaler.pkl."
        )

        st.write(
            f"columns.pkl: {len(columns)} features"
        )

        st.write(
            f"scaler.pkl: {scaler.n_features_in_} features"
        )

        st.stop()


    # =====================================================
    # CHECK UNKNOWN FEATURES
    # =====================================================

    if unknown_features:

        st.warning(
            "⚠️ Additional saved features detected: "
            + ", ".join(
                map(str, unknown_features)
            )
        )


    # =====================================================
    # CHECK ZERO VALUES
    # =====================================================

    missing_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]


    missing_input = False


    for col in missing_columns:

        if col in input_data.columns:

            if input_data[col].iloc[0] == 0:

                missing_input = True


    if missing_input:

        st.warning(
            "⚠️ Some clinical measurements are zero. "
            "Please enter valid values before prediction."
        )

        st.stop()


    # =====================================================
    # SCALE
    # =====================================================

    try:

        input_scaled = scaler.transform(
            input_data
        )

    except Exception as e:

        st.error(
            "❌ Error while processing the input."
        )

        st.code(str(e))

        st.stop()


    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        prediction = model.predict(
            input_scaled
        )[0]


        probability = model.predict_proba(
            input_scaled
        )[0]


        non_diabetes_probability = (
            probability[0] * 100
        )


        diabetes_probability = (
            probability[1] * 100
        )


    except Exception as e:

        st.error(
            "❌ Error during prediction."
        )

        st.code(str(e))

        st.stop()


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


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Prediction",
        "Diabetes Risk"
        if prediction == 1
        else "No Diabetes Risk"
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
    # RESULT MESSAGE
    # =====================================================

    if diabetes_probability >= 70:

        st.error(
            "### 🔴 HIGH RISK"
        )

        st.write(
            "The model estimates a relatively high "
            "probability of diabetes."
        )


    elif diabetes_probability >= 30:

        st.warning(
            "### 🟡 MODERATE RISK"
        )

        st.write(
            "The model estimates an intermediate "
            "probability of diabetes."
        )


    else:

        st.success(
            "### 🟢 LOW RISK"
        )

        st.write(
            "The model estimates a relatively low "
            "probability of diabetes."
        )


    # =====================================================
    # PROBABILITY BREAKDOWN
    # =====================================================

    st.subheader(
        "📊 Probability Breakdown"
    )


    pcol1, pcol2 = st.columns(2)


    pcol1.metric(
        "Non-Diabetic",
        f"{non_diabetes_probability:.1f}%"
    )


    pcol2.metric(
        "Diabetes",
        f"{diabetes_probability:.1f}%"
    )


    # =====================================================
    # PROGRESS BAR
    # =====================================================

    st.subheader(
        "📈 Diabetes Risk"
    )


    st.progress(
        min(
            int(diabetes_probability),
            100
        )
    )


    # =====================================================
    # RISK FACTORS
    # =====================================================

    st.markdown("---")

    st.subheader(
        "⚠️ Risk Factor Analysis"
    )


    risk_factors = []


    if glucose >= 126:

        risk_factors.append(
            f"🔴 High glucose: {glucose:.0f} mg/dL"
        )

    elif glucose >= 100:

        risk_factors.append(
            f"🟡 Elevated glucose: {glucose:.0f} mg/dL"
        )


    if blood_pressure > 80:

        risk_factors.append(
            f"🟡 Elevated blood pressure: "
            f"{blood_pressure:.0f} mm Hg"
        )


    if bmi >= 30:

        risk_factors.append(
            f"🔴 BMI indicates obesity: {bmi:.1f}"
        )

    elif bmi >= 25:

        risk_factors.append(
            f"🟡 BMI is in overweight range: {bmi:.1f}"
        )


    if age > 45:

        risk_factors.append(
            f"🟡 Age-related risk factor: {age} years"
        )


    if pregnancies >= 6:

        risk_factors.append(
            f"🟡 Higher number of pregnancies: "
            f"{pregnancies}"
        )


    if risk_factors:

        for factor in risk_factors:

            st.markdown(
                f"- {factor}"
            )

    else:

        st.success(
            "No major risk factors identified "
            "from the entered measurements."
        )


    # =====================================================
    # PATIENT SUMMARY
    # =====================================================

    st.markdown("---")

    st.subheader(
        "📋 Patient Input Summary"
    )


    summary = pd.DataFrame({

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
        summary,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.markdown("---")

    st.subheader(
        "🤖 Model Information"
    )


    mcol1, mcol2 = st.columns(2)


    mcol1.metric(
        "Features Used",
        len(columns)
    )


    mcol2.metric(
        "Model",
        type(model).__name__
    )


    # =====================================================
    # MEDICAL DISCLAIMER
    # =====================================================

    st.markdown("---")

    st.warning("""
⚠️ **Medical Disclaimer**

This application is intended for educational and screening
purposes only. It is NOT a medical diagnosis and should not
replace examination, laboratory testing, or advice from a
qualified healthcare professional.
""")


# =========================================================
# INITIAL PAGE
# =========================================================

else:

    st.markdown("---")

    st.info(
        "👈 Enter patient information in the sidebar "
        "and click **Predict Diabetes Risk**."
    )


    st.subheader(
        "📌 About SugarSense"
    )


    st.write("""
SugarSense is an educational machine-learning application
for early diabetes risk screening.

The application uses clinical measurements including glucose,
blood pressure, BMI, age, insulin and other patient information
to estimate diabetes risk.
""")


    st.markdown("---")


    st.subheader(
        "📊 Saved Model Information"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Saved Features",
        len(columns)
    )


    c2.metric(
        "Model",
        type(model).__name__
    )


    st.warning("""
⚠️ **Medical Disclaimer**

This application is for educational and screening purposes only.
It is NOT a medical diagnosis.
""")
