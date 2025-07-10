import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# --- CONFIGURATION: REPLACE THESE URLs WITH YOURS ---
MODEL_URL = "https://github.com/Kumaravijay/used-car-price-predictor/blob/main/LinearRegressionModel.pkl"
CSV_URL = "https://github.com/Kumaravijay/used-car-price-predictor/blob/main/Cleaned_Car_data.csv"


@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    with open("LinearRegressionModel.pkl", "wb") as f:
        f.write(response.content)
    with open("LinearRegressionModel.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    return df

# --- LOAD MODEL AND DATA ---
model = load_model()
car = load_data()

# --- STREAMLIT UI ---
st.title("Car Price Prediction App 🚗")

companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

company = st.selectbox('Select Company', companies)
car_model = st.selectbox('Select Car Model', car_models)
year = st.selectbox('Select Year', years)
fuel_type = st.selectbox('Select Fuel Type', fuel_types)
kilo_driven = st.number_input('Kilometers Driven', min_value=0, value=10000, step=1000)

if st.button('Predict Price'):
    input_df = pd.DataFrame([[car_model, company, year, kilo_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    prediction = model.predict(input_df)
    st.success(f"Predicted Car Price: ₹ {np.round(prediction[0], 2)}")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
