🏋️ Calories Burnt Prediction using Machine Learning
📘 Overview

This project predicts the number of calories burnt during exercise based on user parameters such as age, gender, height, weight, duration, heart rate, and body temperature.

The model helps estimate energy expenditure, which can be used in fitness tracking apps, wearable device analytics, and personalized training recommendations.

🎯 Objectives

Analyze and preprocess exercise and calorie data.

Identify key factors affecting calorie burn.

Train multiple regression models and compare performance.

Deploy the best-performing model (XGBoost) for predictions.

📂 Project Structure
Calories_Prediction/
│
├── data/
│   ├── calories.csv
│
├── notebooks/
│   └── calories_prediction.ipynb       # EDA + model training
│
│
├── artifacts/
│   ├── calories_xgb_model.pkl          # saved XGBoost model
│   └── calories_scaler.pkl             # saved StandardScaler
│
├── app.py                              # Optional Flask/Streamlit app
├── requirements.txt
└── README.md

📊 Dataset Information

Two datasets were used and merged on the User_ID column:

Dataset	Description
calories.csv	Contains user IDs and corresponding calories burnt.
exercise.csv	Includes Age, Gender, Height, Weight, Duration, Heart Rate, and Body Temperature.
🔢 Sample Columns

User_ID

Gender (male, female)

Age

Height (cm)

Weight (kg)

Duration (min)

Heart_Rate

Body_Temp (°C)

Calories (kcal)

🧠 Model Development Workflow
1️⃣ Data Preprocessing

Merged calories.csv and exercise.csv using User_ID.

Encoded Gender (male → 0, female → 1).

Removed duplicates and missing values.

Normalized numerical features using StandardScaler.

2️⃣ Exploratory Data Analysis (EDA)

Visualized feature relationships using Seaborn scatterplots.

Checked data distribution and feature correlation.

Identified outliers and key influencing parameters.

3️⃣ Model Training & Evaluation

Trained multiple regression models using train-test split (90%-10%):

Model	Train MAE	Validation MAE
Linear Regression	17.89	18.00
Lasso	17.91	17.99
Ridge	17.89	18.00
Random Forest Regressor	3.99	10.49
XGBoost Regressor	7.89	10.12 ✅

📈 Best Model: XGBoost Regressor (balanced bias–variance trade-off)

💾 Model Saving

Both the trained model and scaler were saved using joblib:

import joblib

joblib.dump(best_model, 'calories_xgb_model.pkl')
joblib.dump(scaler, 'calories_scaler.pkl')

🔍 Model Testing
Load and Predict:
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('calories_xgb_model.pkl')
scaler = joblib.load('calories_scaler.pkl')

# Example input (Age, Height, Weight, Duration)
sample = np.array([[25, 175, 70, 30]])
sample_scaled = scaler.transform(sample)

# Predict calories burnt
pred = model.predict(sample_scaled)
print(f"🔥 Predicted Calories Burnt: {pred[0]:.2f}")

🧰 Tech Stack

Python

NumPy, Pandas

Matplotlib, Seaborn

Scikit-Learn

XGBoost

Joblib

(Optional) Flask / Streamlit (for UI)

📈 Results Visualization

Example plot of predicted vs actual calories:

plt.scatter(Y_val, val_preds, color='blue', alpha=0.6)
plt.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 'r--')
plt.xlabel('Actual Calories Burnt')
plt.ylabel('Predicted Calories Burnt')
plt.title('XGBoost Predictions vs Actual')
plt.show()

🧩 Key Learnings

Gained hands-on experience in regression modeling.

Learned to compare and interpret performance metrics.

Understood how feature scaling affects model stability.

Implemented model saving and loading for deployment readiness.

🚀 Future Improvements

Build a Streamlit web app for real-time calorie predictions.

Add more user parameters (steps, activity type, intensity).

Deploy using Render, HuggingFace Spaces, or Streamlit Cloud.

👨‍💻 Author

Suparn Nayak
Machine Learning Enthusiast | Developer | Problem Solver

📫 Reach me at: LinkedIn
 | GitHub
