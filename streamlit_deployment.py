import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Sidebar for user input
st.title("Car Price Prediction App")

companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

company = st.selectbox('Select Company', companies)
car_model = st.selectbox('Select Car Model', car_models)
year = st.selectbox('Select Year', years)
fuel_type = st.selectbox('Select Fuel Type', fuel_types)
kilo_driven = st.number_input('Kilometers Driven', min_value=0, step=1000)

if st.button('Predict Price'):
    input_df = pd.DataFrame([[car_model, company, year, kilo_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    prediction = model.predict(input_df)
    st.success(f"Predicted Car Price: ₹ {np.round(prediction[0], 2)}")
