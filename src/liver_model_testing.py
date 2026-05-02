import joblib
import pandas as pd

# Load saved model
model = joblib.load("../models/liver_model.pkl")
print(model)

THRESHOLD = 0.30   # adjust if needed

# Feature names (must match training exactly, EXCEPT target column)
feature_names = [
    "Age_of_the_patient",
    "Gender_of_the_patient",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkphos_Alkaline_Phosphotase",
    "Sgpt_Alamine_Aminotransferase",
    "Sgot_Aspartate_Aminotransferase",
    "Total_Protiens",
    "ALB_Albumin",
    "A/G_Ratio_Albumin_and_Globulin_Ratio"
]

# Example input (⚠️ match encoding used in training)
input_data = {
    "Age_of_the_patient": 45,
    "Gender_of_the_patient": 1,   # e.g., Male=1, Female=0 (adjust if needed)
    "Total_Bilirubin": 1.2,
    "Direct_Bilirubin": 0.3,
    "Alkphos_Alkaline_Phosphotase": 200,
    "Sgpt_Alamine_Aminotransferase": 35,
    "Sgot_Aspartate_Aminotransferase": 40,
    "Total_Protiens": 6.5,
    "ALB_Albumin": 3.2,
    "A/G_Ratio_Albumin_and_Globulin_Ratio": 0.9
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# Predict probabilities
probabilities = model.predict_proba(input_df)

# ⚠️ Adjust depending on your target encoding
prob_class0 = probabilities[0][0]
prob_class1 = probabilities[0][1]

# Apply threshold
prediction = 1 if prob_class1 > THRESHOLD else 0

# Output
if prediction == 0:
    print("Prediction: NO DISEASE")
    print(f"Confidence: {prob_class0 * 100:.2f}%")
else:
    print("Prediction: DISEASE DETECTED")
    print(f"Confidence: {prob_class1 * 100:.2f}%")