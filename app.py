from flask import Flask, render_template, request
import joblib
import pandas as pd
import os 

app = Flask(__name__)

# =====================================
# Load Models
# =====================================

# ---- Diabetes Model (dictionary saved) ----
diabetes_data = joblib.load("models/diabetes_model.pkl")
diabetes_model = diabetes_data["model"]
DIABETES_THRESHOLD = 0.45

# ---- Heart Disease Model (pipeline saved) ----
heart_model = joblib.load("models/heart_disease_model.pkl")
HEART_THRESHOLD = 0.30


# =====================================
# Home Route
# =====================================
@app.route("/")
def home():
    return render_template("home.html")


# =====================================
# Diabetes Page Route
# =====================================
@app.route("/diabetes")
def diabetes_page():
    return render_template("diabetes.html")


# =====================================
# Heart Page Route
# =====================================
@app.route("/heart")
def heart_page():
    return render_template("heart.html")


# =====================================
# Diabetes Prediction Route
# =====================================
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        # -------- Numeric Inputs --------
        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        bmi = float(request.form["bmi"])
        hbA1c_level = float(request.form["hbA1c"])
        blood_glucose_level = float(request.form["glucose"])

        # -------- Manual Encoding --------
        gender = 1 if request.form["gender"] == "Male" else 0
        location = 1 if request.form["location"] == "Urban" else 0

        smoking_map = {"never": 0, "former": 1, "current": 2}
        smoking_history = smoking_map[request.form["smoking_history"]]

        selected_race = request.form["race"]

        race_dict = {
            "race:AfricanAmerican": 0,
            "race:Asian": 0,
            "race:Caucasian": 0,
            "race:Hispanic": 0,
            "race:Other": 0
        }

        race_dict[f"race:{selected_race}"] = 1

        # -------- Final Input --------
        input_data = {
            "gender": gender,
            "age": age,
            "location": location,
            **race_dict,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "hbA1c_level": hbA1c_level,
            "blood_glucose_level": blood_glucose_level
        }

        input_df = pd.DataFrame([input_data])

        # -------- Prediction --------
        prob = diabetes_model.predict_proba(input_df)[0][1]
        prediction = 1 if prob > DIABETES_THRESHOLD else 0

        result = "Diabetic" if prediction else "Not Diabetic"

        # -------- Risk Level --------
        if prob < 0.30:
            risk_level = "Low Risk"
        elif prob < 0.60:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return render_template(
            "result.html",
            disease="Diabetes",
            prediction=result,
            probability=round(prob * 100, 2),
            risk_level=risk_level
        )

    except Exception as e:
        return f"Diabetes Prediction Error: {e}"


# =====================================
# Heart Disease Prediction Route
# =====================================
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    try:
        # -------- Input Data (Pipeline handles encoding) --------
        input_data = {
            "Age": float(request.form["age"]),
            "Weight": float(request.form["weight"]),
            "Height": float(request.form["height"]),
            "BMI": float(request.form["bmi"]),
            "Hypertension": int(request.form["hypertension"]),
            "Diabetes": int(request.form["diabetes"]),
            "Hyperlipidemia": int(request.form["hyperlipidemia"]),
            "Family_History": int(request.form["family_history"]),
            "Systolic_BP": float(request.form["systolic_bp"]),
            "Diastolic_BP": float(request.form["diastolic_bp"]),
            "Heart_Rate": float(request.form["heart_rate"]),
            "Blood_Sugar_Fasting": float(request.form["blood_sugar"]),
            "Cholesterol_Total": float(request.form["cholesterol"]),
            "Gender": request.form["gender"],
            "Smoking": request.form["smoking"],
            "Alcohol_Intake": request.form["alcohol"],
            "Physical_Activity": request.form["activity"],
            "Diet": request.form["diet"],
            "Stress_Level": request.form["stress"]
        }

        input_df = pd.DataFrame([input_data])

        # -------- Prediction --------
        prob = heart_model.predict_proba(input_df)[0][1]
        prediction = 1 if prob > HEART_THRESHOLD else 0

        result = "Heart Disease Detected" if prediction else "No Heart Disease"

        # -------- Risk Level --------
        if prob < 0.30:
            risk_level = "Low Risk"
        elif prob < 0.60:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return render_template(
            "result.html",
            disease="Heart Disease",
            prediction=result,
            probability=round(prob * 100, 2),
            risk_level=risk_level
        )

    except Exception as e:
        return f"Heart Prediction Error: {e}"
    





    

    # =====================================
# Breast Cancer Model
# =====================================
breast_cancer_model = joblib.load("models/breast_cancer_model.pkl")
breast_cancer_scaler = joblib.load("models/breast_cancer_scaler.pkl")

# =====================================
# Breast Cancer Page Route
# =====================================
@app.route("/breast-cancer")
def breast_cancer_page():
    return render_template("breast_cancer.html")

# =====================================
# Breast Cancer Predict Route
# =====================================
@app.route("/predict-breast-cancer", methods=["POST"])
def predict_breast_cancer():
    try:
        # Get form data
        features = [
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

        # Collect input
        input_data = {f: float(request.form[f]) for f in features}
        input_df = pd.DataFrame([input_data])

        # Scale and predict
        input_scaled = breast_cancer_scaler.transform(input_df)
        prediction = breast_cancer_model.predict(input_scaled)
        probability = breast_cancer_model.predict_proba(input_scaled)

        # Result
        if prediction[0] == 1:
           result = "Malignant - Cancer Detected"
           probability_val = round(probability[0][1] * 100, 2)
           risk_level = "High Risk"
        else:
           result = "Benign - No Cancer"
           probability_val = round(probability[0][0] * 100, 2)
           risk_level = "Low Risk"

        return render_template("result.html",
                      disease="Breast Cancer",
                      prediction=result,
                      probability=probability_val,
                      risk_level=risk_level)
    except Exception as e:
        return render_template("breast_cancer.html",
                             error=str(e))


# =====================================
# Run Application
# =====================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

