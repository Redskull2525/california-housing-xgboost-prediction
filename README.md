
# 🏠 California Housing Price Prediction using XGBoost

A **Machine Learning web application** that predicts California housing prices using the **XGBoost regression algorithm**. The model is deployed using **Streamlit** and allows users to input housing features to estimate the predicted house value.

🔗 **Live App:**
[https://california-housing-xgboost-prediction-abhishek.streamlit.app/](https://california-housing-xgboost-prediction-abhishek.streamlit.app/)

---

# 📌 Project Overview

The goal of this project is to build a **machine learning regression model** that predicts the **median house value** in California based on several socioeconomic and geographic features.

The application uses the **California Housing Dataset**, which contains **20,640 housing samples and 8 numerical features** related to housing districts. ([inria.github.io][1])

Users can enter housing parameters through the **Streamlit sidebar interface**, and the model instantly predicts the expected house price.

---

# ⚙️ Technologies Used

* Python
* Scikit-learn
* XGBoost
* Streamlit
* NumPy
* Pandas
* Pickle (for model serialization)

---

# 🧠 Machine Learning Model

The project uses **XGBoost Regressor**, a powerful gradient boosting algorithm designed for high performance and efficiency in regression tasks. ([GeeksforGeeks][2])

XGBoost builds multiple decision trees sequentially where each new tree learns from the errors of the previous trees, improving prediction accuracy. ([mbrenndoerfer.com][3])

Model parameters used:

```
n_estimators = 100
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
max_depth = 3
random_state = 42
```

---

# 📊 Dataset Information

The **California Housing Dataset** includes socioeconomic and geographic data for housing districts in California.

### Input Features

| Feature    | Description                 |
| ---------- | --------------------------- |
| MedInc     | Median income of households |
| HouseAge   | Median age of houses        |
| AveRooms   | Average number of rooms     |
| AveBedrms  | Average number of bedrooms  |
| Population | Total population of block   |
| AveOccup   | Average occupancy           |
| Latitude   | Geographic latitude         |
| Longitude  | Geographic longitude        |

These features are used to predict the **median house value** of a district. ([Hugging Face][4])

---

# 🚀 How the Application Works

1. The trained **XGBoost model** is saved using **pickle**.
2. The model is loaded inside a **Streamlit web application**.
3. Users input housing features using the **sidebar controls**.
4. The model predicts the **estimated house price** instantly.

---

# 📂 Project Structure

```
california-housing-xgboost-prediction
│
├── app.py                 # Streamlit application
├── xgboost.pkl            # Trained ML model
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

# ▶️ Run the Project Locally

### 1️⃣ Clone the repository

```
git clone https://github.com/Redskull2525/california-housing-xgboost-prediction.git
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit app

```
streamlit run app.py
```

---

# 🌐 Deployment

This project is deployed using **Streamlit Cloud**, which allows easy deployment of machine learning applications directly from GitHub.

---

# 📈 Future Improvements

* Add **feature importance visualization**
* Add **data visualization dashboards**
* Improve **UI/UX design**
* Add **map visualization of housing locations**

---

# 👨‍💻 Author

**Abhishek Shelke**

🔗 GitHub:
[https://github.com/Redskull2525](https://github.com/Redskull2525)

🔗 LinkedIn:
[https://www.linkedin.com/in/abhishek-s-b98895249](https://www.linkedin.com/in/abhishek-s-b98895249)

---

⭐ If you like this project, consider **starring the repository**!

---
