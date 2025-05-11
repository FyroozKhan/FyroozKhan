
#  Implementing baseline  Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# # Load dataset
# data = pd.read_csv("healthdata.csv")


# # Define features and target variable
# X = data.drop('charges', axis=1)
# y = data['charges']

# # Preprocessing pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), ['age', 'bmi', 'children']),
#         ('cat', OneHotEncoder(drop='first'), ['sex', 'smoker', 'region'])
#     ]
# )

# # Apply preprocessing to features
# X_processed = preprocessor.fit_transform(X)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# print(f"Training set size: {X_train.shape}")
# print(f"Testing set size: {X_test.shape}")

# model = LinearRegression()
# model.fit(X_train, y_train)

# # Display model coefficients and intercept
# print("Model Coefficients:", model.coef_)
# print("Model Intercept:", model.intercept_)

# # Make predictions on test set
# y_pred = model.predict(X_test)

# # Evaluate performance using R² and RMSE
# r2 = r2_score(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)

# print(f"R² Score: {r2:.3f}")
# print(f"RMSE: ${rmse:,.2f}")

# Plot actual vs predicted charges
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.6, color='skyblue')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Reference line (y=x)
# plt.xlabel('Actual Charges')
# plt.ylabel('Predicted Charges')
# plt.title('Actual vs Predicted Charges')
# plt.show()

# Building the Web Application 
# Streamlit Web App

# joblib.dump(model, 'insurance_model.pkl')
# joblib.dump(preprocessor, 'insurance_preprocessor.pkl')
 
# Load the trained model and preprocessor
model = joblib.load('insurance_model.pkl')
preprocessor = joblib.load('insurance_preprocessor.pkl')
 
st.title("Insurance Charges Predictor")
 
st.write("Enter the details below to predict insurance charges:")
 
# User input widgets
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", options=['male', 'female'])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.selectbox("Smoker", options=['yes', 'no'])
region = st.selectbox("Region", options=['northeast', 'northwest', 'southeast', 'southwest'])
 
if st.button("Predict Charges"):
    # Prepare input data as a DataFrame
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
 
    # Preprocess input
    input_processed = preprocessor.transform(input_df)
 
    # Predict
    prediction = model.predict(input_processed)[0]
 
    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")
