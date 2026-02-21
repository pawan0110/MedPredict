from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ==============================
# Load Model + Threshold
# ==============================
model_data = joblib.load("models/diabetes_model.pkl")

model = model_data["model"]
THRESHOLD = 0.45


# ==============================
# Home Route
# ==============================
@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# Prediction Route
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --------------------------
        # 1️⃣ Basic Numeric Inputs
        # --------------------------
        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        bmi = float(request.form["bmi"])
        hbA1c_level = float(request.form["hbA1c"])
        blood_glucose_level = float(request.form["glucose"])

        # --------------------------
        # 2️⃣ Encode Gender (Manual)
        # --------------------------
        gender = 1 if request.form["gender"] == "Male" else 0

        # --------------------------
        # 3️⃣ Encode Location
        # --------------------------
        location = 1 if request.form["location"] == "Urban" else 0

        # --------------------------
        # 4️⃣ Encode Smoking
        # --------------------------
        smoking_map = {
            "never": 0,
            "former": 1,
            "current": 2
        }
        smoking_history = smoking_map[request.form["smoking_history"]]

        # --------------------------
        # 5️⃣ One-Hot Encode Race
        # --------------------------
        selected_race = request.form["race"]

        race_dict = {
            "race:AfricanAmerican": 0,
            "race:Asian": 0,
            "race:Caucasian": 0,
            "race:Hispanic": 0,
            "race:Other": 0
        }

        race_dict[f"race:{selected_race}"] = 1

        # --------------------------
        # 6️⃣ Final Input Dictionary
        # --------------------------
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

        # --------------------------
        # 7️⃣ Predict
        # --------------------------
        prob = model.predict_proba(input_df)[0][1]
        prediction = 1 if prob > THRESHOLD else 0

        result = "Diabetic" if prediction else "Not Diabetic"

        # Risk interpretation
        if prob < 0.30:
            risk_level = "Low Risk"
        elif prob < 0.60:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return render_template(
            "result.html",
            prediction=result,
            probability=round(prob * 100, 2),
            risk_level=risk_level
        )

    except Exception as e:
        return f"Error occurred: {e}"


# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)