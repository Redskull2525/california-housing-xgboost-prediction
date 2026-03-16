import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="California House Price Prediction", layout="wide")

# Title
st.title("🏠 California House Price Prediction App")
st.write("This app predicts house prices using an XGBoost Machine Learning model.")

# Load model
model = pickle.load(open("xgboost.pkl", "rb"))

# Sidebar
st.sidebar.header("Input House Features")

medinc = st.sidebar.number_input("Median Income (MedInc)", min_value=0.0, value=3.0)
houseage = st.sidebar.number_input("House Age (HouseAge)", min_value=0.0, value=20.0)
averooms = st.sidebar.number_input("Average Rooms (AveRooms)", min_value=0.0, value=5.0)
avebedrooms = st.sidebar.number_input("Average Bedrooms (AveBedrms)", min_value=0.0, value=1.0)
population = st.sidebar.number_input("Population", min_value=0.0, value=1000.0)
aveoccup = st.sidebar.number_input("Average Occupancy (AveOccup)", min_value=0.0, value=3.0)
latitude = st.sidebar.number_input("Latitude", value=34.0)
longitude = st.sidebar.number_input("Longitude", value=-118.0)

# Input array
input_data = np.array([[medinc, houseage, averooms, avebedrooms, population, aveoccup, latitude, longitude]])

# Prediction button
if st.button("Predict House Price"):
    prediction = model.predict(input_data)

    st.subheader("Predicted House Price")
    st.success(f"${prediction[0]*100000:,.2f}")
