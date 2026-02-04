from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/diabetes_rf_pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
         # Get form values (must match input names in HTML)
        pregnancies = float(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        blood_pressure = float(request.form["BloodPressure"])
        skin_thickness = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        dpf = float(request.form["DiabetesPedigreeFunction"])
        age = float(request.form["Age"])    

         # Arrange input in same order as training data
        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]])   

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = "Daibetic" if prediction == 1 else "Not daibetic"

        return render_template(
            "result.html",
            prediction=result,
            probability=round(probability*100,2)
        )  
                            
    except Exception as e:
        return f"Error occured: {e}"
    

if __name__ == "__main__":
    app.run(debug=True)
   