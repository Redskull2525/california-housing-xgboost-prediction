import streamlit as st
import pickle
import numpy as np

# Page configuration
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

**GitHub**  
https://github.com/Redskull2525

**LinkedIn**  
https://www.linkedin.com/in/abhishek-s-b98895249
""")

# -------- INPUT FEATURES -------- #
st.sidebar.header("Input Housing Features")

medinc = st.sidebar.number_input("Median Income (MedInc)", value=3.0)
houseage = st.sidebar.number_input("House Age (HouseAge)", value=20.0)
averooms = st.sidebar.number_input("Average Rooms (AveRooms)", value=5.0)
avebedrooms = st.sidebar.number_input("Average Bedrooms (AveBedrms)", value=1.0)
population = st.sidebar.number_input("Population", value=1000.0)
aveoccup = st.sidebar.number_input("Average Occupancy (AveOccup)", value=3.0)
latitude = st.sidebar.number_input("Latitude", value=34.0)
longitude = st.sidebar.number_input("Longitude", value=-118.0)

# -------- TITLE -------- #
st.title("🏠 California House Price Prediction App")
st.write("Predict house prices using an XGBoost Machine Learning model.")

# -------- LOAD MODEL -------- #
with open("xgboost.pkl", "rb") as file:
    model = pickle.load(file)

# -------- PREDICTION -------- #
input_data = np.array([[medinc, houseage, averooms, avebedrooms,
                        population, aveoccup, latitude, longitude]])

if st.button("Predict House Price"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ${prediction[0]*100000:,.2f}")
