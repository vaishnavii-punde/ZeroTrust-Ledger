import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ZeroTrust Ledger", layout="wide")

# 🎯 Title
st.title("📘 ZeroTrust Ledger – Internal Fraud Detection")

# 📤 Upload CSV
uploaded_file = st.file_uploader("📁 Upload your ledger CSV file", type=["csv"])

if uploaded_file:
    # 🧾 Read file
    df_new = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Ledger Data")
    st.dataframe(df_new.head())

    # 💾 Load trained model
    with open("models/fraud_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    # 🔄 Preprocessing function
    def preprocess_input(df):
        for col in ['Vendor', 'Description', 'Department', 'Status']:
            df[col] = df[col].astype('category').cat.codes
        df = df.drop(columns=['Entry_ID', 'Date'], errors='ignore')
        return df

    # 🧹 Preprocess
    df_processed = preprocess_input(df_new.copy())

    # 🎯 Predict
    predictions = loaded_model.predict(df_processed)

    # 📌 Show results
    df_new['Is_Suspicious_Prediction'] = predictions
    st.subheader("🚨 Suspicion Prediction Results")
    st.dataframe(df_new[['Amount', 'Vendor', 'Description', 'Is_Suspicious_Prediction']])

    # 📊 Suspicious vs Normal Transaction Count
    st.subheader("📌 Suspicious vs. Normal Transaction Count")
    suspicion_counts = df_new['Is_Suspicious_Prediction'].value_counts()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=suspicion_counts.index, y=suspicion_counts.values, palette='Set2', ax=ax1)
    ax1.set_xticklabels(['Normal', 'Suspicious'])
    ax1.set_ylabel("Number of Transactions")
    st.pyplot(fig1)

    # 💰 Transaction Amount by Suspicion Type
    st.subheader("💰 Transaction Amount by Suspicion Type")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Is_Suspicious_Prediction', y='Amount', data=df_new, palette='coolwarm', ax=ax2)
    ax2.set_xticklabels(['Normal', 'Suspicious'])
    ax2.set_ylabel("Amount")
    st.pyplot(fig2)

    # 📤 Download suspicious transactions
    st.subheader("⬇️ Download Suspicious Transactions")
    suspicious_df = df_new[df_new['Is_Suspicious_Prediction'] == 1]

    if not suspicious_df.empty:
        csv = suspicious_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Suspicious Records as CSV",
            data=csv,
            file_name="suspicious_transactions.csv",
            mime='text/csv'
        )
    else:
        st.info("✅ No suspicious records detected.")

    # 🧮 Summary statistics
    st.subheader("📊 Summary")
    total = len(df_new)
    suspicious = df_new['Is_Suspicious_Prediction'].sum()
    percent = round((suspicious / total) * 100, 2)

    st.markdown(f"""
    - **Total Transactions:** {total}  
    - **Suspicious Transactions:** {suspicious}  
    - **Suspicion Rate:** {percent}%  
    """)

else:
    st.info("👆 Upload a valid ledger `.csv` file to begin.")
