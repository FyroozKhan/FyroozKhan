import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, preprocessor, and important features list
model = joblib.load('insurance_model.pkl')
preprocessor = joblib.load('insurance_preprocessor.pkl')
important_features = joblib.load('important_features.pkl')

st.title("Insurance Charges Predictor (Optimized Model)")

st.write("Enter the details below to predict insurance charges:")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", options=['male', 'female'])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", options=['yes', 'no'])
region = st.selectbox("Region", options=['northeast', 'northwest', 'southeast', 'southwest'])

if st.button("Predict Charges"):
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    input_processed = preprocessor.transform(input_df)
    input_df_transformed = pd.DataFrame(input_processed, columns=preprocessor.get_feature_names_out())

    # Create same interaction terms
    if 'cat__smoker_yes' in input_df_transformed.columns:
        input_df_transformed['bmi_smoker'] = input_df_transformed['num__bmi'] * input_df_transformed['cat__smoker_yes']
        input_df_transformed['age_smoker'] = input_df_transformed['num__age'] * input_df_transformed['cat__smoker_yes']
    else:
        input_df_transformed['bmi_smoker'] = 0
        input_df_transformed['age_smoker'] = 0

    input_selected = input_df_transformed[important_features]
    prediction = model.predict(input_selected)[0]

    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")
