import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="California House Price Prediction", layout="wide")

# -------- SIDEBAR -------- #
st.sidebar.title("👨‍💻 Developer")

st.sidebar.markdown("""
**Abhishek Shelke**

M.Sc Computer Science  
ASM's CSIT, Pimpri  

**Interests**
- Data Science
- Machine Learning
- AI

GitHub  
https://github.com/Redskull2525

LinkedIn  
https://www.linkedin.com/in/abhishek-s-b98895249
""")

# -------- TITLE -------- #
st.title("🏠 California Housing Price Prediction")
st.write("Predict house prices using an XGBoost Machine Learning model.")

# -------- LOAD MODEL -------- #
with open("xgboost.pkl", "rb") as file:
    model = pickle.load(file)

# -------- INPUT SECTION (CENTER) -------- #
st.subheader("Enter Housing Features")

col1, col2 = st.columns(2)

with col1:
    medinc = st.number_input("Median Income (MedInc)", value=3.0)
    houseage = st.number_input("House Age", value=20.0)
    averooms = st.number_input("Average Rooms", value=5.0)
    avebedrooms = st.number_input("Average Bedrooms", value=1.0)

with col2:
    population = st.number_input("Population", value=1000.0)
    aveoccup = st.number_input("Average Occupancy", value=3.0)
    latitude = st.number_input("Latitude", value=34.0)
    longitude = st.number_input("Longitude", value=-118.0)

# -------- PREDICTION -------- #
input_data = np.array([[medinc, houseage, averooms, avebedrooms,
                        population, aveoccup, latitude, longitude]])

if st.button("Predict House Price"):
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ${prediction[0]*100000:,.2f}")
