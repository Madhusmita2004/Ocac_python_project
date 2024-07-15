import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('house_price_model.pkl')

# Define a function to make predictions
def predict_price(total_sqft, bhk, balcony,):
    features = np.array([[total_sqft, bhk, balcony]])
    prediction = model.predict(features)
    return prediction[0]

# Title of the app
st.title('Real Estate House Price Prediction')

# Input fields for user to enter data
st.header('Enter the details of the house:')
total_sqft = st.number_input('Total Square Feet', min_value=0.0, value=0.0)
bhk = st.number_input('BHK', min_value=2, step=1)
balcony = st.number_input('Balcony', min_value=0, step=1)

# Add a button to make the prediction
if st.button('Predict'):
    price = predict_price(total_sqft, bhk, balcony,)
    st.subheader(f'Predicted House Price: {price:,.2f}')

# Instructions on how to run the app
st.write("Save this script as `app.py` and run it with the following command:")
st.code("streamlit run app.py", language="bash")
