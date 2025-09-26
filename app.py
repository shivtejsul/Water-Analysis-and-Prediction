# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Water Analysis & Prediction", layout="wide")

# App Title
st.title("💧 Water Analysis & Prediction - Home")

st.markdown("### Upload your dataset to get started!")

# Upload Dataset
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Save dataset in session state
    st.session_state["dataset"] = df  

    st.success("✅ Dataset Uploaded Successfully!")
    st.dataframe(df.head())

# Info Section
st.markdown("""
### ℹ About the Project
This web application allows users to:
- Upload water quality datasets
- Perform interactive visual analysis
- Predict water quality using ML models
- Provide feedback for improvement
""")
