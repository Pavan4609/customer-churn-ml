import streamlit as st
import requests


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered customer retention analysis using Machine Learning</div>',
    unsafe_allow_html=True
)


# ---------------- CUSTOMER INFORMATION ----------------

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:

    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col3:

    device = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


# ---------------- BILLING INFORMATION ----------------

st.markdown(
    '<div class="section-title">💳 Billing Information</div>',
    unsafe_allow_html=True
)

col4, col5, col6 = st.columns(3)

with col4:

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col5:

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with col6:

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=600.0
    )


# ---------------- PREDICT BUTTON ----------------

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)


# ---------------- PREDICTION ----------------

if predict_button:

    data = {

        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,

        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,

        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,

        "StreamingTV": tv,
        "StreamingMovies": movies,

        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,

        "MonthlyCharges": monthly,
        "TotalCharges": total
    }


    try:

        response = requests.post(
    "https://customer-churn-ml-qcdw.onrender.com/predict",
    json=data
        )


        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]

            probability = result["churn_probability"]


            # ---------------- RESULT ----------------

            st.markdown(
                '<div class="section-title">📈 Prediction Result</div>',
                unsafe_allow_html=True
            )


            if prediction == 1:

                st.error(
                    "⚠️ Customer may churn"
                )

            else:

                st.success(
                    "✅ Customer likely to stay"
                )


            # ---------------- METRICS ----------------

            col7, col8, col9 = st.columns(3)


            with col7:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.0f}%"
                )


            with col8:

                st.metric(
                    "Prediction",
                    "Churn" if prediction == 1 else "Stay"
                )


            with col9:

                st.metric(
                    "Contract",
                    contract
                )


            # ---------------- PROBABILITY BAR ----------------

            st.write("### Churn Risk")

            st.progress(
                probability
            )


            # ---------------- CUSTOMER SUMMARY ----------------

            st.write("### Customer Summary")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:

                st.write(
                    f"**Tenure:** {tenure} months"
                )

                st.write(
                    f"**Internet:** {internet}"
                )

                st.write(
                    f"**Contract:** {contract}"
                )


            with summary_col2:

                st.write(
                    f"**Monthly Charges:** ₹{monthly:.2f}"
                )

                st.write(
                    f"**Total Charges:** ₹{total:.2f}"
                )

                st.write(
                    f"**Payment Method:** {payment}"
                )


        else:

            st.error(
                "API request failed."
            )


    except requests.exceptions.ConnectionError:

       st.error(
    "❌ Cannot connect to the FastAPI server. "
    "Please try again later."
)


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "Customer Churn Prediction • Machine Learning • "
    "Python • Scikit-learn • FastAPI • Streamlit"
)