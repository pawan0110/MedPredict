from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv
import json
from datetime import datetime

# =====================================
# LOAD ENVIRONMENT VARIABLES
# =====================================

load_dotenv()

app = Flask(__name__)

# =====================================
# API CLIENTS
# =====================================

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# Store conversations
conversations = {}

# =====================================
# LOAD MODELS
# =====================================

# ---- Diabetes Model ----
diabetes_data = joblib.load("models/diabetes_model.pkl")
diabetes_model = diabetes_data["model"]
DIABETES_THRESHOLD = 0.45

# ---- Heart Disease Model ----
heart_model = joblib.load("models/heart_disease_model.pkl")
HEART_THRESHOLD = 0.30

# ---- Breast Cancer Model ----
breast_cancer_model = joblib.load("models/best_breast_cancer_model.pkl")
breast_cancer_scaler = joblib.load("models/breast_cancer_scaler.pkl")

# ---- Kidney Disease Model ----
kidney_model = joblib.load("models/best_model_Random_Forest.pkl")
kidney_num_imputer = joblib.load("models/preprocessor_num_imputer.pkl")
kidney_cat_imputer = joblib.load("models/preprocessor_cat_imputer.pkl")
kidney_le_target = joblib.load("models/preprocessor_label_encoder.pkl")

kidney_num_cols = list(kidney_num_imputer.feature_names_in_)
kidney_cat_cols = list(kidney_cat_imputer.feature_names_in_)

# ---- Liver Disease Model ----
liver_model = joblib.load("models/liver_model.pkl")
LIVER_THRESHOLD = 0.30

# =====================================
# HOME ROUTE
# =====================================

@app.route("/")
def home():
    return render_template("home.html")

# =====================================
# CHATBOT PAGE
# =====================================

@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")

# =====================================
# WEB SEARCH FUNCTION
# =====================================

def web_search(query):
    try:
        response = tavily_client.search(query)

        results = response.get("results", [])

        if not results:
            return "No reliable medical information found."

        summary = "\n\n".join([
            f"{i+1}. {r.get('content', 'No content available')} "
            f"(Source: {r.get('url', 'Unknown')})"
            for i, r in enumerate(results[:3])
        ])

        return f"Search Results:\n\n{summary}"

    except Exception as e:
        return f"Web search error: {str(e)}"

# =====================================
# CHATBOT ROUTE
# =====================================
@app.route("/chat", methods=["POST"])
def chat():

    try:
        data = request.get_json()

        thread_id = data.get("threadId", "default")
        user_message = data.get("message", "")

        # Create conversation memory
        if thread_id not in conversations:

            current_datetime = datetime.now().strftime(
                "%A, %B %d, %Y %I:%M %p"
            )

            system_prompt = f"""
You are an AI-powered Medical Assistant Chatbot integrated into a Health Disease Prediction System.

Current date and time: {current_datetime}

Your purpose is to:
- Provide simple and helpful medical-related information.
- Explain diseases, symptoms, medical terms, lab reports, and health parameters in easy language.
- Help users understand AI-based disease prediction results and risk percentages.
- Suggest healthy lifestyle habits, diet improvements, exercise, sleep, hydration, stress management, and preventive care.
- Support users with general wellness guidance.
- Encourage users to consult qualified healthcare professionals when necessary.

Important Rules:
- You are NOT a licensed doctor.
- You are only an AI medical assistant.
- Never provide guaranteed diagnoses or cures.
- Never prescribe medicines, dosages, or dangerous treatments.
- Do not provide emergency treatment instructions.
- Avoid fear-inducing responses.
- If symptoms appear severe or emergency-related, advise immediate medical attention.
- If information is uncertain, clearly mention that.

Response Style:
- Keep responses short, clear, and easy to read.
- Use headings and bullet points.
- Avoid large paragraphs.
- Use simple language instead of complex medical jargon.
- Make responses mobile-friendly and user-friendly.
- Keep bullet points short.
- Highlight important information clearly.

Formatting Rules:
- Structure responses using sections such as:
  - Meaning / Explanation
  - Possible Causes
  - What You Can Do
  - Healthy Lifestyle Tips
  - Important Note
- Use markdown formatting when appropriate.
- For prediction results:
  - Explain the risk level clearly.
  - Mention that predictions are AI-generated estimates, not confirmed diagnoses.
  - Suggest healthy lifestyle improvements.
  - Recommend medical consultation.

Capabilities:
- Answer general medical and wellness questions.
- Explain terms like HbA1c, hemoglobin, cholesterol, BMI, blood pressure, glucose, etc.
- Explain disease risk percentages from prediction models.
- Provide general food and lifestyle suggestions.
- Respond to greetings and casual conversation politely.
- Provide current date and time when asked.

Always include a medical disclaimer for important health-related responses:
"Please consult a qualified healthcare professional for proper medical advice and diagnosis.""
"""

            conversations[thread_id] = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]

        # Add user message
        conversations[thread_id].append({
            "role": "user",
            "content": user_message
        })

        # Limit memory
        if len(conversations[thread_id]) > 20:
            conversations[thread_id] = conversations[thread_id][-20:]

        # =====================================
        # NORMAL AI RESPONSE
        # =====================================

        completion = groq_client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=conversations[thread_id],

            temperature=0.2,

            max_tokens=400
        )

        assistant_reply = completion.choices[0].message.content

        conversations[thread_id].append({
            "role": "assistant",
            "content": assistant_reply
        })

        return jsonify({
            "message": assistant_reply
        })

    except Exception as e:

        print("CHAT ERROR:", str(e))

        return jsonify({
            "message": f"Sorry, an error occurred: {str(e)}"
        })
        

# =====================================
# DIABETES PAGE
# =====================================

@app.route("/diabetes")
def diabetes_page():
    return render_template("diabetes.html")

# =====================================
# DIABETES PREDICTION
# =====================================

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():

    try:

        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        bmi = float(request.form["bmi"])
        hbA1c_level = float(request.form["hbA1c"])
        blood_glucose_level = float(request.form["glucose"])

        gender = 1 if request.form["gender"] == "Male" else 0
        location = 1 if request.form["location"] == "Urban" else 0

        smoking_map = {
            "never": 0,
            "former": 1,
            "current": 2
        }

        smoking_history = smoking_map[
            request.form["smoking_history"]
        ]

        selected_race = request.form["race"]

        race_dict = {
            "race:AfricanAmerican": 0,
            "race:Asian": 0,
            "race:Caucasian": 0,
            "race:Hispanic": 0,
            "race:Other": 0
        }

        race_dict[f"race:{selected_race}"] = 1

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

        prob = diabetes_model.predict_proba(
            pd.DataFrame([input_data])
        )[0][1]

        prediction = 1 if prob > DIABETES_THRESHOLD else 0

        result = (
            "Diabetic"
            if prediction
            else "Not Diabetic"
        )

        risk_level = (
            "Low Risk"
            if prob < 0.30
            else (
                "Moderate Risk"
                if prob < 0.60
                else "High Risk"
            )
        )

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
# HEART PAGE
# =====================================

@app.route("/heart")
def heart_page():
    return render_template("heart.html")

# =====================================
# HEART PREDICTION
# =====================================

@app.route("/predict_heart", methods=["POST"])
def predict_heart():

    try:

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

        prob = heart_model.predict_proba(
            pd.DataFrame([input_data])
        )[0][1]

        prediction = 1 if prob > HEART_THRESHOLD else 0

        result = (
            "Heart Disease Detected"
            if prediction
            else "No Heart Disease"
        )

        risk_level = (
            "Low Risk"
            if prob < 0.30
            else (
                "Moderate Risk"
                if prob < 0.60
                else "High Risk"
            )
        )

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
# BREAST CANCER PAGE
# =====================================

@app.route("/breast-cancer")
def breast_cancer_page():
    return render_template("breast_cancer.html")

# =====================================
# BREAST CANCER PREDICTION
# =====================================

@app.route("/predict-breast-cancer", methods=["POST"])
def predict_breast_cancer():

    try:

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

        input_df = pd.DataFrame([{
            f: float(request.form[f]) for f in features
        }])

        input_scaled = breast_cancer_scaler.transform(input_df)

        prediction = breast_cancer_model.predict(input_scaled)
        probability = breast_cancer_model.predict_proba(input_scaled)

        if prediction[0] == 1:
            result = "Malignant - Cancer Detected"
            probability_val = round(probability[0][1] * 100, 2)
            risk_level = "High Risk"
        else:
            result = "Benign - No Cancer"
            probability_val = round(probability[0][0] * 100, 2)
            risk_level = "Low Risk"

        return render_template(
            "result.html",
            disease="Breast Cancer",
            prediction=result,
            probability=probability_val,
            risk_level=risk_level
        )

    except Exception as e:
        return render_template(
            "breast_cancer.html",
            error=str(e)
        )

# =====================================
# KIDNEY PREPROCESSING
# =====================================

def preprocess_kidney(raw: dict) -> np.ndarray:

    raw_clean = {
        k.strip().lower(): v for k, v in raw.items()
    }

    df = pd.DataFrame([raw_clean])

    df[kidney_num_cols] = kidney_num_imputer.transform(
        df[kidney_num_cols]
    )

    df[kidney_cat_cols] = kidney_cat_imputer.transform(
        df[kidney_cat_cols]
    )

    for col in kidney_cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df[kidney_num_cols + kidney_cat_cols].values

# =====================================
# KIDNEY PAGE
# =====================================

@app.route("/kidney")
def kidney_page():
    return render_template("kidney.html")

# =====================================
# KIDNEY PREDICTION
# =====================================

@app.route("/predict_kidney", methods=["POST"])
def predict_kidney():

    try:

        raw = {
            'age of the patient': float(request.form["age"]),
            'blood pressure (mm/hg)': float(request.form["blood_pressure"]),
            'specific gravity of urine': float(request.form["specific_gravity"]),
            'albumin in urine': float(request.form["albumin"]),
            'sugar in urine': float(request.form["sugar"]),
            'random blood glucose level (mg/dl)': float(request.form["blood_glucose"]),
            'blood urea (mg/dl)': float(request.form["blood_urea"]),
            'serum creatinine (mg/dl)': float(request.form["serum_creatinine"]),
            'sodium level (meq/l)': float(request.form["sodium"]),
            'potassium level (meq/l)': float(request.form["potassium"]),
            'hemoglobin level (gms)': float(request.form["hemoglobin"]),
            'packed cell volume (%)': float(request.form["packed_cell_volume"]),
            'white blood cell count (cells/cumm)': float(request.form["wbc_count"]),
            'red blood cell count (millions/cumm)': float(request.form["rbc_count"]),
            'estimated glomerular filtration rate (egfr)': float(request.form["egfr"]),
            'urine protein-to-creatinine ratio': float(request.form["urine_protein_creatinine"]),
            'urine output (ml/day)': float(request.form["urine_output"]),
            'serum albumin level': float(request.form["serum_albumin"]),
            'cholesterol level': float(request.form["cholesterol"]),
            'parathyroid hormone (pth) level': float(request.form["pth_level"]),
            'serum calcium level': float(request.form["serum_calcium"]),
            'serum phosphate level': float(request.form["serum_phosphate"]),
            'body mass index (bmi)': float(request.form["bmi"]),
            'duration of diabetes mellitus (years)': float(request.form["diabetes_duration"]),
            'duration of hypertension (years)': float(request.form["hypertension_duration"]),
            'cystatin c level': float(request.form["cystatin_c"]),
            'c-reactive protein (crp) level': float(request.form["crp_level"]),
            'interleukin-6 (il-6) level': float(request.form["il6_level"]),
            'red blood cells in urine': request.form["rbc_urine"],
            'pus cells in urine': request.form["pus_cells"],
            'pus cell clumps in urine': request.form["pus_cell_clumps"],
            'bacteria in urine': request.form["bacteria"],
            'hypertension (yes/no)': request.form["hypertension_yn"],
            'diabetes mellitus (yes/no)': request.form["diabetes_yn"],
            'coronary artery disease (yes/no)': request.form["cad"],
            'appetite (good/poor)': request.form["appetite"],
            'pedal edema (yes/no)': request.form["pedal_edema"],
            'anemia (yes/no)': request.form["anemia"],
            'family history of chronic kidney disease': request.form["family_history"],
            'smoking status': request.form["smoking_status"],
            'physical activity level': request.form["physical_activity"],
            'urinary sediment microscopy results': request.form["urinary_sediment"],
        }

        X = preprocess_kidney(raw)
        pred_int = kidney_model.predict(X)[0]
        pred_label = kidney_le_target.inverse_transform([pred_int])[0]
        proba = kidney_model.predict_proba(X)[0]
        confidence = float(proba[pred_int])

        risk_map = {
            'No_Disease': 'Low Risk',
            'Low_Risk': 'Low Risk',
            'Moderate_Risk': 'Moderate Risk',
            'High_Risk': 'High Risk',
            'Severe_Disease': 'High Risk',
        }

        risk_level = risk_map.get(pred_label, 'Unknown')

        return render_template(
            "result.html",
            disease="Kidney Disease",
            prediction=pred_label.replace('_', ' '),
            probability=round(confidence * 100, 2),
            risk_level=risk_level
        )

    except Exception as e:
        return render_template(
            "kidney.html",
            error=str(e)
        )

# =====================================
# LIVER PAGE
# =====================================

@app.route("/liver")
def liver_page():
    return render_template("liver.html")

# =====================================
# LIVER PREDICTION
# =====================================

@app.route("/predict_liver", methods=["POST"])
def predict_liver():

    try:

        input_data = {
            "Age_of_the_patient": float(request.form["age"]),
            "Gender_of_the_patient": 1 if request.form["gender"] == "Male" else 0,
            "Total_Bilirubin": float(request.form["total_bilirubin"]),
            "Direct_Bilirubin": float(request.form["direct_bilirubin"]),
            "Alkphos_Alkaline_Phosphotase": float(request.form["alkphos"]),
            "Sgpt_Alamine_Aminotransferase": float(request.form["sgpt"]),
            "Sgot_Aspartate_Aminotransferase": float(request.form["sgot"]),
            "Total_Protiens": float(request.form["total_proteins"]),
            "ALB_Albumin": float(request.form["albumin"]),
            "A/G_Ratio_Albumin_and_Globulin_Ratio": float(request.form["ag_ratio"])
        }

        input_df = pd.DataFrame([input_data])

        probabilities = liver_model.predict_proba(input_df)

        prob_no_disease = probabilities[0][0]
        prob_disease = probabilities[0][1]

        prediction = 1 if prob_disease > LIVER_THRESHOLD else 0

        if prediction == 1:
            result = "Liver Disease Detected"
            probability = prob_disease
            risk_level = "High Risk" if prob_disease > 0.60 else "Moderate Risk"
        else:
            result = "No Liver Disease"
            probability = prob_no_disease
            risk_level = "Low Risk"

        return render_template(
            "result.html",
            disease="Liver Disease",
            prediction=result,
            probability=round(probability * 100, 2),
            risk_level=risk_level
        )

    except Exception as e:
        return f"Liver Prediction Error: {e}"

# =====================================
# RUN APP
# =====================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )