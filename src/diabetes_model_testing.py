import joblib
import pandas as pd

# Load models
rf_model = joblib.load("../models/diabetes_rf_pipeline.pkl")


feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Example patient input
input_data = {
    "Pregnancies": 1,
    "Glucose": 85,
    "BloodPressure": 66,
    "SkinThickness": 29,
    "Insulin": 0,
    "BMI": 26.6,
    "DiabetesPedigreeFunction": 0.351,
    "Age": 31
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

rf_pred = rf_model.predict(input_df)
rf_prob = rf_model.predict_proba(input_df)

prob_no_diabetes = rf_prob[0][0]
prob_diabetes = rf_prob[0][1]

if rf_pred[0] == 0:
    print("Prediction: NON-DIABETIC")
    print(f"Confidence: {prob_no_diabetes * 100:.2f}% (No Diabetes)")
    print(f"Diabetes Risk: {prob_diabetes * 100:.2f}%")
else:
    print("Prediction: DIABETIC")
    print(f"Confidence: {prob_diabetes * 100:.2f}% (Diabetes)")
    print(f"Non-Diabetes Probability: {prob_no_diabetes * 100:.2f}%")
