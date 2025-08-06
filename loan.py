import streamlit as st
import pandas as pd
import pickle

# Load the trained model and columns
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("x_columns.pkl", "rb") as f:
    X_columns = pickle.load(f)

st.title("Loan Prediction App")

# User input
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", ["1.0", "0.0"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Prepare input as dictionary
input_dict = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': float(credit_history),
    'Property_Area': property_area
}

input_df = pd.DataFrame([input_dict])

# One-hot encode input
input_encoded = pd.get_dummies(input_df)

# Align input with training features
input_encoded = input_encoded.reindex(columns=X_columns, fill_value=0)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    if prediction == 'Y':
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved.")
