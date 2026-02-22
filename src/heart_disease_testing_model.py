import joblib
import pandas as pd

# Load saved pipeline model
model = joblib.load("../models/heart_disease_model.pkl")

THRESHOLD = 0.30   # Use optimized threshold

# Feature names (MUST match training dataset exactly)
feature_names = [
    "Age",
    "Weight",
    "Height",
    "BMI",
    "Hypertension",
    "Diabetes",
    "Hyperlipidemia",
    "Family_History",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Blood_Sugar_Fasting",
    "Cholesterol_Total",
    "Gender",
    "Smoking",
    "Alcohol_Intake",
    "Physical_Activity",
    "Diet",
    "Stress_Level"
]

# Example patient input
input_data = {
    "Age": 58,
    "Weight": 82,
    "Height": 168,
    "BMI": 29.0,
    "Hypertension": 1,
    "Diabetes": 0,
    "Hyperlipidemia": 1,
    "Family_History": 1,
    "Systolic_BP": 150,
    "Diastolic_BP": 95,
    "Heart_Rate": 88,
    "Blood_Sugar_Fasting": 130,
    "Cholesterol_Total": 240,
    "Gender": "Male",
    "Smoking": "Former",
    "Alcohol_Intake": "Moderate",
    "Physical_Activity": "Sedentary",
    "Diet": "Average",
    "Stress_Level": "High"
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# Get probabilities
probabilities = model.predict_proba(input_df)

prob_no_disease = probabilities[0][0]
prob_disease = probabilities[0][1]

# Apply custom threshold
prediction = 1 if prob_disease > THRESHOLD else 0

# Print result
if prediction == 0:
    print("Prediction: NO HEART DISEASE")
    print(f"Confidence: {prob_no_disease * 100:.2f}%")
    print(f"Heart Disease Risk Probability: {prob_disease * 100:.2f}%")
else:
    print("Prediction: HEART DISEASE DETECTED")
    print(f"Confidence: {prob_disease * 100:.2f}%")
    print(f"No Disease Probability: {prob_no_disease * 100:.2f}%")