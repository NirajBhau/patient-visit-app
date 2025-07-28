import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle

# Load the trained model
try:
    pipeline = joblib.load('patient_visit_model.joblib')
except FileNotFoundError:
    st.error("Model file 'patient_visit_model.joblib' not found. Please upload it.")
    st.stop()

# Load the list of patient names
try:
    with open('patient_names.pkl', 'rb') as f:
        patient_names = pickle.load(f)
except FileNotFoundError:
    st.error("File 'patient_names.pkl' not found.")
    st.stop()

st.title('Patient Visit Prediction App')

st.write("Enter the details to predict patient visit count.")

# Patient Name selection
selected_patient = st.selectbox('Select Patient Name', patient_names)

# Date input
selected_date = st.date_input('Select Date')

if st.button('Predict Visit Count'):
    input_data = pd.DataFrame({
        'PATIENT NAME': [selected_patient],
        'year': [selected_date.year],
        'month': [selected_date.month],
        'day_of_week': [selected_date.weekday()],
        'week_of_year': [selected_date.isocalendar().week],
        'quarter': [(selected_date.month - 1) // 3 + 1]
    })

    predicted_visit_count = pipeline.predict(input_data)

    st.subheader('Predicted Visit Count:')
    st.write(f"The predicted visit count for {selected_patient} on {selected_date.strftime('%Y-%m-%d')} is: {max(0, round(predicted_visit_count[0]))}")
