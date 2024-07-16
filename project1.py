import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generate some dummy data for demonstration
X, y = make_regression(n_samples=100, n_features=3, noise=0.1)
# Create a RandomForestRegressor model
rf_model = RandomForestRegressor()
# Fit the model
rf_model.fit(X, y)

# Save the trained model to a file
joblib.dump(rf_model, 'house_price_model.pkl')


import streamlit as st
import joblib
import numpy as np

# Load the model
model_path = 'house_price_model.pkl'
# Check if the model file exists
try:
    model = joblib.load(model_path)
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error(f"Model file not found. Please ensure {model_path} is in the current directory.")
except ModuleNotFoundError:
    st.error("Required module not found. Please install the necessary packages using pip.")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Define a function to make predictions
def predict_price(total_sqft, bhk, balcony):
    if 'model' in globals():
        features = np.array([[total_sqft, bhk, balcony]])
        prediction = model.predict(features)
        return prediction[0]
    else:
        return None

# Title of the app
st.title('Real Estate House Price Prediction')

# Input fields for user to enter data
st.header('Enter the details of the house:')
total_sqft = st.number_input('Total Square Feet', min_value=0.0, value=0.0)
bhk = st.number_input('BHK', min_value=1, step=1)
balcony = st.number_input('Balcony', min_value=1, step=1)

# Add a button to make the prediction
if st.button('Predict'):
    if 'model' in globals():
        price = predict_price(total_sqft, bhk, balcony)
        if price is not None:
            st.subheader(f'Predicted House Price: {price:,.2f}')
        else:
            st.error("Model is not loaded. Please ensure the model file is available.")
    else:
        st.error("Model is not loaded. Please ensure the model file is available.")

# Instructions on how to run the app
st.write("Save this script as `app.py` and run it with the following command:")
st.code("streamlit run app.py", language="bash")
