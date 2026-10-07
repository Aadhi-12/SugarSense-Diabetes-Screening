import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SugarSense - Diabetes Prediction",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding: 0rem 1rem;
}

h1 {
    color: #1f77b4;
}

.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL, SCALER AND COLUMNS
# =========================================================

@st.cache_resource
def load_model():

    try:

        model = joblib.load(
            "Diabetes_Disease_prediction.pkl"
        )

        scaler = joblib.load(
            "scaler.pkl"
        )

        columns = joblib.load(
            "columns.pkl"
        )

        return model, scaler, columns

    except FileNotFoundError:

        return None, None, None


model, scaler, columns = load_model()


# =========================================================
# CHECK MODEL FILES
# =========================================================

if model is None:

    st.error("❌ Model files not found!")

    st.info("""
    Make sure these files are in the same folder as app.py:

    1. Diabetes_Disease_prediction.pkl
    2. scaler.pkl
    3. columns.pkl
    """)

    st.stop()


# =========================================================
# CHECK FEATURES
# =========================================================

if len(columns) != 11:

    st.error(
        f"❌ Feature mismatch detected! "
        f"Your saved columns.pkl contains {len(columns)} features."
    )

    st.warning("""
    Your model should use 11 features:

    8 original features +
    3 engineered features
    """)

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title("🏥 SugarSense")

st.markdown(
    "### AI-Powered Early Diabetes Risk Prediction and Screening Tool"
)

st.info(
    "Enter the patient's clinical measurements in the sidebar "
    "and click **Predict Diabetes Risk**."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Patient Information")

st.sidebar.subheader("Clinical Measurements")


# ---------------------------------------------------------
# 1. Pregnancies
# ---------------------------------------------------------

pregnancies = st.sidebar.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)


# ---------------------------------------------------------
# 2. Glucose
# ---------------------------------------------------------

glucose = st.sidebar.number_input(
    "Glucose (mg/dL)",
    min_value=0.0,
    max_value=300.0,
    value=120.0,
    step=1.0
)


# ---------------------------------------------------------
# 3. Blood Pressure
# ---------------------------------------------------------

bp = st.sidebar.number_input(
    "Blood Pressure (mm Hg)",
    min_value=0.0,
    max_value=150.0,
    value=70.0,
    step=1.0
)


# ---------------------------------------------------------
# 4. Skin Thickness
# ---------------------------------------------------------

skin = st.sidebar.number_input(
    "Skin Thickness (mm)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=1.0
)


# ---------------------------------------------------------
# 5. Insulin
# ---------------------------------------------------------

insulin = st.sidebar.number_input(
    "Insulin (μU/ml)",
    min_value=0.0,
    max_value=900.0,
    value=80.0,
    step=1.0
)


# ---------------------------------------------------------
# 6. BMI
# ---------------------------------------------------------

bmi = st.sidebar.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)


# ---------------------------------------------------------
# 7. Diabetes Pedigree Function
# ---------------------------------------------------------

dpf = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.47,
    step=0.01
)


# ---------------------------------------------------------
# 8. Age
# ---------------------------------------------------------

age = st.sidebar.number_input(
    "Age",
    min_value=21,
    max_value=120,
    value=30,
    step=1
)


# =========================================================
# PREDICT BUTTON
# =========================================================

st.sidebar.markdown("---")

predict_btn = st.sidebar.button(
    "🔮 Predict Diabetes Risk",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_btn:

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    # 1. Obesity flag

    obesity = int(
        bmi >= 30
    )


    # 2. Glucose × BMI interaction

    glucose_bmi = glucose * bmi


    # 3. Age × Glucose interaction

    age_glucose = age * glucose


    # =====================================================
    # CREATE INPUT DATAFRAME
    # =====================================================

    input_values = [[
        pregnancies,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age,
        obesity,
        glucose_bmi,
        age_glucose
    ]]


    input_data = pd.DataFrame(
        input_values,
        columns=columns
    )


    # =====================================================
    # HANDLE ZERO VALUES
    # =====================================================

    missing_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]


    for col in missing_columns:

        if input_data[col].iloc[0] == 0:

            input_data[col] = pd.NA


    # =====================================================
    # CHECK FOR MISSING VALUES
    # =====================================================

    if input_data.isna().any().any():

        st.warning(
            "⚠️ Some measurements are zero/missing. "
            "Your current saved scaler/model does not contain "
            "an imputer, so please enter valid clinical values."
        )

        st.stop()


    # =====================================================
    # SCALE INPUT
    # =====================================================

    input_scaled = scaler.transform(
        input_data
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(
        input_scaled
    )[0]


    # =====================================================
    # PREDICTION PROBABILITY
    # =====================================================

    try:

        probability = model.predict_proba(
            input_scaled
        )[0]

        prob_negative = probability[0] * 100

        prob_positive = probability[1] * 100

    except:

        prob_positive = (
            100
            if prediction == 1
            else 0
        )

        prob_negative = (
            100 - prob_positive
        )


    # =====================================================
    # RESULTS
    # =====================================================

    st.markdown("---")

    st.header("🎯 Prediction Results")


    col1, col2 = st.columns([2, 1])


    # =====================================================
    # PREDICTION MESSAGE
    # =====================================================

    with col1:

        if prediction == 1:

            if prob_positive >= 70:

                st.error(
                    "### 🔴 HIGH RISK - Diabetes Detected"
                )

            else:

                st.warning(
                    "### ⚠️ MODERATE RISK - Diabetes Risk"
                )

        else:

            if prob_positive < 30:

                st.success(
                    "### 🟢 LOW RISK - Diabetes Not Detected"
                )

            else:

                st.warning(
                    "### ⚠️ MODERATE RISK - Please Monitor"
                )


        # =================================================
        # PROBABILITY
        # =================================================

        st.subheader(
            "Probability Breakdown"
        )


        pcol1, pcol2 = st.columns(2)


        pcol1.metric(
            "Non-Diabetic Probability",
            f"{prob_negative:.1f}%"
        )


        pcol2.metric(
            "Diabetic Probability",
            f"{prob_positive:.1f}%"
        )


    # =====================================================
    # GAUGE
    # =====================================================

    with col2:

        fig = go.Figure(
            go.Indicator(

                mode="gauge+number",

                value=prob_positive,

                title={
                    "text": "Diabetes Risk"
                },

                number={
                    "suffix": "%"
                },

                gauge={

                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "darkblue"
                    },

                    "steps": [

                        {
                            "range": [0, 30],
                            "color": "lightgreen"
                        },

                        {
                            "range": [30, 70],
                            "color": "yellow"
                        },

                        {
                            "range": [70, 100],
                            "color": "red"
                        }

                    ]
                }
            )
        )


        fig.update_layout(

            height=300,

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # RISK FACTOR ANALYSIS
    # =====================================================

    st.markdown("---")

    st.subheader(
        "⚠️ Risk Factor Analysis"
    )


    risk_factors = []

    positive_factors = []


    # -----------------------------------------------------
    # Glucose
    # -----------------------------------------------------

    if glucose >= 126:

        risk_factors.append(
            "🔴 High Glucose level (≥126 mg/dL)"
        )

    elif glucose >= 100:

        risk_factors.append(
            "🟡 Elevated Glucose level"
        )

    else:

        positive_factors.append(
            "🟢 Glucose level is below the elevated range"
        )


    # -----------------------------------------------------
    # Blood Pressure
    # -----------------------------------------------------

    if bp > 80:

        risk_factors.append(
            "🔴 High Blood Pressure (>80 mm Hg)"
        )

    else:

        positive_factors.append(
            "🟢 Blood Pressure is not above 80 mm Hg"
        )


    # -----------------------------------------------------
    # BMI
    # -----------------------------------------------------

    if bmi >= 30:

        risk_factors.append(
            "🔴 BMI indicates obesity (≥30)"
        )

    elif 18.5 <= bmi < 25:

        positive_factors.append(
            "🟢 BMI is in the healthy range"
        )


    # -----------------------------------------------------
    # Age
    # -----------------------------------------------------

    if age > 45:

        risk_factors.append(
            "🟡 Age-related risk factor (>45)"
        )


    # -----------------------------------------------------
    # Pregnancies
    # -----------------------------------------------------

    if pregnancies >= 6:

        risk_factors.append(
            "🟡 Higher number of pregnancies"
        )


    # =====================================================
    # DISPLAY RISK FACTORS
    # =====================================================

    if risk_factors:

        st.warning(
            "**Identified Risk Factors:**"
        )

        for factor in risk_factors:

            st.markdown(
                f"- {factor}"
            )


    # =====================================================
    # DISPLAY POSITIVE FACTORS
    # =====================================================

    if positive_factors:

        st.success(
            "**Positive Indicators:**"
        )

        for factor in positive_factors:

            st.markdown(
                f"- {factor}"
            )


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    st.markdown("---")

    st.subheader(
        "💡 Recommendations"
    )


    if prediction == 1:

        st.error("""
**Recommended Actions:**

- Consult a qualified healthcare professional.
- Consider appropriate diabetes screening.
- Monitor blood glucose regularly.
- Maintain a balanced diet and healthy lifestyle.
        """)

    else:

        st.success("""
**Maintain Healthy Practices:**

- Continue regular health check-ups.
- Maintain a balanced diet.
- Exercise regularly.
- Maintain a healthy body weight.
        """)


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.markdown("---")

    st.subheader(
        "📋 Patient Input Summary"
    )


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

            bp,

            skin,

            insulin,

            bmi,

            dpf,

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


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Model",
        "Logistic Regression",
        "Accuracy: 77.2%"
    )


    col2.metric(
        "Features",
        "11"
    )


    col3.metric(
        "Dataset",
        "768 Samples"
    )


    st.markdown("---")


    st.subheader(
        "📌 About SugarSense"
    )


    st.write("""
    SugarSense is an educational machine-learning application
    designed for early diabetes risk screening.

    The model uses clinical measurements such as glucose,
    blood pressure, BMI, age and other patient information
    to estimate diabetes risk.
    """)


    st.warning("""
    ⚠️ **Medical Disclaimer**

    This application is for educational and screening purposes only.
    It is NOT a medical diagnosis and should not replace advice,
    examination or testing by a qualified healthcare professional.
    """)
