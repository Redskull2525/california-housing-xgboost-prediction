import streamlit as st
import numpy as np
import joblib
import xgboost  # important so pickle can find XGBRegressor

# Load model
@st.cache_resource
def load_model():
    return joblib.load("xgboost.pkl")

model = load_model()

st.title("🏠 California Housing Price Prediction")

st.sidebar.header("Input Features")

med_inc = st.sidebar.number_input("Median Income", value=3.0)
house_age = st.sidebar.number_input("House Age", value=20)
ave_rooms = st.sidebar.number_input("Average Rooms", value=5.0)
ave_bedrooms = st.sidebar.number_input("Average Bedrooms", value=1.0)
population = st.sidebar.number_input("Population", value=1000)
ave_occup = st.sidebar.number_input("Average Occupancy", value=3.0)
latitude = st.sidebar.number_input("Latitude", value=34.0)
longitude = st.sidebar.number_input("Longitude", value=-118.0)

features = np.array([[med_inc, house_age, ave_rooms, ave_bedrooms,
                      population, ave_occup, latitude, longitude]])

if st.button("Predict Price"):
    prediction = model.predict(features)[0] * 100000
    st.success(f"Predicted House Price: ${prediction:,.2f}")
