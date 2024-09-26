import streamlit as st
import pandas as pd
from predictive_model import train_predictive_model
import joblib

st.title("Predictive Maintenance with IoT Data")

# Upload CSV file for sensor data
uploaded_file = st.file_uploader("Upload sensor data CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:", data.head())

    if st.button("Train Model"):
        model = train_predictive_model('data/sensor_data.csv')
        joblib.dump(model, "predictive_model.pkl")
        st.write("Model trained successfully!")

