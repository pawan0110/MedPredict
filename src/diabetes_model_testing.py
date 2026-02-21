import joblib
import pandas as pd

# Load saved dictionary
data = joblib.load("../models/diabetes_model.pkl")

model = data["model"]
THRESHOLD = 0.45

# Feature names (must match training exactly)
feature_names = [
    "gender",
    "age",
    "location",
    "race:AfricanAmerican",
    "race:Asian",
    "race:Caucasian",
    "race:Hispanic",
    "race:Other",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "hbA1c_level",
    "blood_glucose_level"
]

# Example patient input
input_data = {
    "gender": "Male",
    "age": 52,
    "location": "Urban",
    "race:AfricanAmerican": 0,
    "race:Asian": 0,
    "race:Caucasian": 1,
    "race:Hispanic": 0,
    "race:Other": 0,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "former",
    "bmi": 31.2,
    "hbA1c_level": 6.5,
    "blood_glucose_level": 140
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# Get probabilities
probabilities = model.predict_proba(input_df)

prob_no_diabetes = probabilities[0][0]
prob_diabetes = probabilities[0][1]

# Apply custom threshold
prediction = 1 if prob_diabetes > THRESHOLD else 0

# Print result
if prediction == 0:
    print("Prediction: NON-DIABETIC")
    print(f"Confidence: {prob_no_diabetes * 100:.2f}% (No Diabetes)")
    print(f"Diabetes Risk Probability: {prob_diabetes * 100:.2f}%")
else:
    print("Prediction: DIABETIC")
    print(f"Confidence: {prob_diabetes * 100:.2f}% (Diabetes)")
    print(f"Non-Diabetes Probability: {prob_no_diabetes * 100:.2f}%")