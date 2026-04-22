import joblib
import pandas as pd
import numpy as np

# Load saved model and scaler
model = joblib.load(r'C:\Users\ARYAN - LAPTOP\MedPredict\models\breast_cancer_model.pkl')
scaler = joblib.load(r'C:\Users\ARYAN - LAPTOP\MedPredict\models\breast_cancer_scaler.pkl')

# Feature names
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Example patient input
input_data = {
    'radius_mean': 17.99,
    'texture_mean': 10.38,
    'perimeter_mean': 122.80,
    'area_mean': 1001.0,
    'smoothness_mean': 0.11840,
    'compactness_mean': 0.27760,
    'concavity_mean': 0.30010,
    'concave points_mean': 0.14710,
    'symmetry_mean': 0.24190,
    'fractal_dimension_mean': 0.07871,
    'radius_se': 1.095,
    'texture_se': 0.9053,
    'perimeter_se': 8.589,
    'area_se': 153.40,
    'smoothness_se': 0.006399,
    'compactness_se': 0.04904,
    'concavity_se': 0.05373,
    'concave points_se': 0.01587,
    'symmetry_se': 0.03003,
    'fractal_dimension_se': 0.006193,
    'radius_worst': 25.38,
    'texture_worst': 17.33,
    'perimeter_worst': 184.60,
    'area_worst': 2019.0,
    'smoothness_worst': 0.1622,
    'compactness_worst': 0.6656,
    'concavity_worst': 0.7119,
    'concave points_worst': 0.2654,
    'symmetry_worst': 0.4601,
    'fractal_dimension_worst': 0.11890
}

# Make prediction
input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)

# Print result
if prediction[0] == 1:
    print("Prediction: MALIGNANT (Cancer Detected)")
    print(f"Confidence: {probability[0][1] * 100:.2f}%")
else:
    print("Prediction: BENIGN (No Cancer)")
    print(f"Confidence: {probability[0][0] * 100:.2f}%")