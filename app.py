import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="California Housing Price Predictor",
                   page_icon="🏠",
                   layout="wide")

# Load model
with open("xgboost.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🏠 California Housing Price Prediction")
st.write("Predict house prices using an XGBoost Machine Learning model.")

# Sidebar
st.sidebar.header("Input Housing Features")

med_inc = st.sidebar.number_input("Median Income", min_value=0.0, value=3.0)
house_age = st.sidebar.number_input("House Age", min_value=0, value=20)
ave_rooms = st.sidebar.number_input("Average Rooms", min_value=0.0, value=5.0)
ave_bedrooms = st.sidebar.number_input("Average Bedrooms", min_value=0.0, value=1.0)
population = st.sidebar.number_input("Population", min_value=0, value=1000)
ave_occup = st.sidebar.number_input("Average Occupancy", min_value=0.0, value=3.0)
latitude = st.sidebar.number_input("Latitude", value=34.0)
longitude = st.sidebar.number_input("Longitude", value=-118.0)

# Input array
features = np.array([[med_inc, house_age, ave_rooms, ave_bedrooms,
                      population, ave_occup, latitude, longitude]])

# Prediction
if st.button("Predict House Price"):

    prediction = model.predict(features)

    st.subheader("Predicted House Price")
    st.success(f"${prediction[0]*100000:,.2f}")

# Info section
st.markdown("---")
st.write("### About the Model")
st.write("""
This application uses an **XGBoost Regressor** trained on the California Housing dataset.

Features used:
- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude
""")
