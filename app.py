import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import requests
from streamlit_lottie import st_lottie

# -------- PAGE CONFIG -------- #
st.set_page_config(page_title="California Housing Price Prediction", layout="wide")

# -------- CUSTOM CSS -------- #
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color: white;
}

[data-testid="stSidebar"]{
background: linear-gradient(180deg,#141E30,#243B55);
}

h1,h2,h3{
text-align:center;
color:white;
}

.stButton>button {
background-color:#00c6ff;
color:white;
border-radius:10px;
height:3em;
width:100%;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# -------- SIDEBAR -------- #
st.sidebar.title("👨‍💻 Developer")

st.sidebar.markdown("""
**Abhishek Shelke**

M.Sc Computer Science  
ASM's CSIT, Pimpri  

### Interests
- Data Science
- Machine Learning
- Artificial Intelligence

### GitHub
https://github.com/Redskull2525

### LinkedIn
https://www.linkedin.com/in/abhishek-s-b98895249
""")

# -------- LOAD MODEL -------- #
model = pickle.load(open("xgboost.pkl", "rb"))

# -------- TITLE -------- #
st.title("🏠 California Housing Price Prediction")
st.write("Predict house prices using an **XGBoost Machine Learning model**.")

# -------- LOTTIE ANIMATION -------- #
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

st_lottie(lottie_ai, height=300)

# -------- INPUT SECTION -------- #
st.subheader("Enter Housing Features")

col1, col2 = st.columns(2)

with col1:
    medinc = st.number_input("Median Income", value=3.0)
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

    price = prediction[0] * 100000

    st.markdown(f"""
    <div style="
    padding:20px;
    border-radius:10px;
    background:linear-gradient(135deg,#36d1dc,#5b86e5);
    text-align:center;
    font-size:30px;
    color:white;">
    
    Predicted House Price <br>
    <b>${price:,.2f}</b>

    </div>
    """, unsafe_allow_html=True)

# -------- FEATURE IMPORTANCE -------- #
if st.checkbox("Show Feature Importance"):

    features = ["MedInc","HouseAge","AveRooms","AveBedrms",
                "Population","AveOccup","Latitude","Longitude"]

    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_title("Feature Importance")

    st.pyplot(fig)

# -------- FOOTER -------- #
st.markdown("---")
st.write("Built with ❤️ using Streamlit and XGBoost")
