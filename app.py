import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

st.title("🛡️ Fraud Detection for NBFC & UPI")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['txn_hour'] = pd.to_datetime(df['timestamp']).dt.hour
    features = df[['amount', 'txn_hour']]
    df['fraud_score'] = model.predict_proba(features)[:,1]
    df['fraud_flag'] = (df['fraud_score'] > 0.7).astype(int)

    st.subheader("🔍 Fraudulent Transactions")
    st.dataframe(df[df['fraud_flag'] == 1])

    st.download_button("Download Report", df.to_csv(index=False), file_name="fraud_report.csv")
