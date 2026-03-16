import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="California Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    with open("xgboost.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = joblib.load("xgboost.pkl")


# Title
st.title("🏠 California Housing Price Prediction")
st.write("Predict the median house price using an XGBoost Machine Learning model.")

# Sidebar inputs
st.sidebar.header("Enter Housing Features")

med_inc = st.sidebar.number_input("Median Income", min_value=0.0, value=3.0)
house_age = st.sidebar.number_input("House Age", min_value=0, value=20)
ave_rooms = st.sidebar.number_input("Average Rooms", min_value=0.0, value=5.0)
ave_bedrooms = st.sidebar.number_input("Average Bedrooms", min_value=0.0, value=1.0)
population = st.sidebar.number_input("Population", min_value=0, value=1000)
ave_occup = st.sidebar.number_input("Average Occupancy", min_value=0.0, value=3.0)
latitude = st.sidebar.number_input("Latitude", value=34.0)
longitude = st.sidebar.number_input("Longitude", value=-118.0)

# Create feature array
features = np.array([[med_inc, house_age, ave_rooms, ave_bedrooms,
                      population, ave_occup, latitude, longitude]])

# Prediction button
if st.button("Predict House Price"):

    prediction = model.predict(features)

    st.subheader("Predicted Median House Price")

    # Multiply because dataset values are in 100k
    price = prediction[0] * 100000

    st.success(f"Estimated House Price: ${price:,.2f}")

# Footer
st.markdown("---")
st.write("### About the Model")
st.write("""
This application uses an **XGBoost Regressor** trained on the California Housing dataset.

Features used in prediction:
- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude
""")
