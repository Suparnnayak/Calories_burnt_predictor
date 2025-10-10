import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load('artifacts/xgb_model.pkl')
scaler = joblib.load('artifacts/scaler.pkl')

# New input data
new_data = pd.DataFrame([{
    'Age': 30,
    'Height': 170,
    'Gender': 0,       # 0=male
    'Body_Temp': 37,
    'Heart_Rate': 120   # a typical workout heart rate
}])


# Ensure columns match what the scaler expects
new_data = new_data[scaler.feature_names_in_]

# Scale features
new_data_scaled = scaler.transform(new_data)

# Predict
predicted_calories = model.predict(new_data_scaled)
print("Predicted Calories Burnt:", predicted_calories[0])
