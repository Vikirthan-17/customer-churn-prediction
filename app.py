import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/logistic_regression_balanced.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details below to predict whether the customer is likely to churn")

gender= st.selectbox("Gender", ["Female", "Male"])
senior_citizen=st.selectbox("Senior Citizen", [0,1], index=0)
partner= st.selectbox("Partner", ["Yes", "No"])
dependents= st.selectbox("Dependents", ["Yes", "No"])
tenure= st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

phone_service= st.selectbox("Phone Service", ["Yes", "No"])
if phone_service == "No":
    multiple_line_options = ["No phone service"]
else:
    multiple_line_options = ["No", "Yes"]
multiple_lines= st.selectbox("Multiple Lines", multiple_line_options)
internet_service= st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
if internet_service == "No":
    internet_options= ["No internet service"]
else:
    internet_options= ["No", "Yes"]
online_security= st.selectbox("Online Security", ["No internet service", "Yes", "No"])
online_backup= st.selectbox("Online Backup", ["No internet service", "Yes", "No"])
device_protection= st.selectbox("Device Protection", ["No internet service", "Yes", "No"])
tech_support= st.selectbox("Tech Support", ["No internet service", "Yes", "No"])
streaming_tv= st.selectbox("Streaming TV", ["No internet service", "Yes", "No"])
streaming_movies= st.selectbox("Streaming Movies", ["No internet service", "Yes", "No"])
contract= st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing= st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method= st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges= st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges= st.number_input("Total Charges", min_value=0.0, value=1000.0)

def encode_binary(value):

    return 1 if value == "Yes" else 0

gender_encoded = 1 if gender == "Male" else 0

partner_encoded = encode_binary(partner)

dependents_encoded = encode_binary(dependents)

phone_service_encoded = encode_binary(phone_service)

paperless_billing_encoded = encode_binary(paperless_billing)

multiple_lines_map = {"No": 0, "No phone service": 1, "Yes": 2}

internet_service_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

three_option_map = {"No": 0, "No internet service": 1, "Yes": 2}

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}

payment_method_map = {

    "Bank transfer (automatic)": 0,

    "Credit card (automatic)": 1,

    "Electronic check": 2,

    "Mailed check": 3

}

input_data = pd.DataFrame([[

    gender_encoded,

    senior_citizen,

    partner_encoded,

    dependents_encoded,

    tenure,

    phone_service_encoded,

    multiple_lines_map[multiple_lines],

    internet_service_map[internet_service],

    three_option_map[online_security],

    three_option_map[online_backup],

    three_option_map[device_protection],

    three_option_map[tech_support],

    three_option_map[streaming_tv],

    three_option_map[streaming_movies],

    contract_map[contract],

    paperless_billing_encoded,

    payment_method_map[payment_method],

    monthly_charges,

    total_charges

]], columns=[

    "gender",

    "SeniorCitizen",

    "Partner",

    "Dependents",

    "tenure",

    "PhoneService",

    "MultipleLines",

    "InternetService",

    "OnlineSecurity",

    "OnlineBackup",

    "DeviceProtection",

    "TechSupport",

    "StreamingTV",

    "StreamingMovies",

    "Contract",

    "PaperlessBilling",

    "PaymentMethod",

    "MonthlyCharges",

    "TotalCharges"

])

if st.button("Predict Churn"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    prediction_probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:

        st.error(f"This customer is likely to churn. Churn probability: {prediction_probability:.2f}")

    else:

        st.success(f"This customer is not likely to churn. Churn probability: {prediction_probability:.2f}")