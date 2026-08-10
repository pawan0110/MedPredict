# 🧠 MedPredict – AI Disease Risk Prediction System

MedPredict is a **Machine Learning powered healthcare web application** that predicts the probability of major diseases based on patient medical data.

The system currently predicts the risk of:

* 🧪 **Diabetes**
* ❤️ **Heart Disease**
* 🫁 **Breast Cancer**
* 🫘 **Kidney Disease**
* 🫀 **Liver Disease**
* 🤖 **Medical Assistant Chatbot**

Users input medical parameters through a web interface and the system uses trained **machine learning models** to estimate disease risk.

This project demonstrates the integration of **Machine Learning, Flask Web Development, and Cloud Deployment**.

---

---

# 🚀 Features

* Diabetes Risk Prediction
* Heart Disease Risk Prediction
* Breast Cancer Risk Prediction
* Kidney Disease Risk Prediction
* Liver Disease Risk Prediction
* Medical Assistant Chatbot
* Real-time ML predictions
* Risk probability calculation
* Risk level classification (Low / Moderate / High)
* Interactive web interface
* Cloud deployment using Render

---

# 🏗 Project Architecture

```
User (Browser)
      ↓
HTML Forms
      ↓
Flask Backend (app.py)
      ↓
Machine Learning Models (.pkl)
      ↓
Prediction Result
```

---

# ⚙️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost
* Pandas
* NumPy

### Backend

* Flask

### Frontend

* HTML
* CSS

### Model Storage

* Joblib

### Deployment

* Render

---

# 📂 Project Structure

```
MedPredict
│
├── datasets
│   ├── breast_cancer.csv
│   ├── diabetes_dataset.csv
│   ├── heart.csv
│   ├── kidney_disease.csv
│   └── synthetic_heart_disease.csv
│
├── models
│   ├── diabetes_model.pkl
│   └── heart_disease_model.pkl
│
├── notebooks
│   ├── eda
│   │   ├── Diabetes_EDA.ipynb
│   │   └── HeartDiseases_EDA.ipynb
│   │
│   └── model_training
│
├── src
│   ├── diabetes_model_testing.py
│   └── heart_disease_testing.py
│
├── static
│   ├── images
│   └── style.css
│
├── templates
│   ├── home.html
│   ├── diabetes.html
│   ├── heart.html
│   └── result.html
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🤖 Machine Learning Models

The project uses trained classification models saved as **`.pkl` files**.

### Diabetes Model

The diabetes model predicts whether a patient is diabetic using features like:

* Gender
* Age
* Location
* Race
* Hypertension
* Heart Disease
* Smoking History
* BMI
* HbA1c Level
* Blood Glucose Level

Prediction is based on probability threshold:

```
Threshold = 0.45
```

---

### Heart Disease Model

The heart disease model uses a **pipeline model** which handles preprocessing automatically.

Input features include:

* Age
* Weight
* Height
* BMI
* Hypertension
* Diabetes
* Hyperlipidemia
* Family History
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Heart Rate
* Fasting Blood Sugar
* Total Cholesterol
* Gender
* Smoking
* Alcohol Intake
* Physical Activity
* Diet
* Stress Level

Prediction threshold:

```
Threshold = 0.30
```

---

### Medical Assistant Chatbot

The chatbot provides general health information, explains prediction results, and answers questions about the app. It uses AI (Groq) and web search (Tavily) for responses, but always advises consulting a professional doctor. It does not provide medical diagnoses or treatments.

---

# 🔄 Application Workflow

### 1️⃣ User opens the website

```
Home Page
```

Options available:

* Diabetes Prediction
* Heart Disease Prediction

---

### 2️⃣ User enters medical information

Example (Diabetes):

```
Gender
Age
Location
Race
Hypertension
Heart Disease
Smoking History
BMI
HbA1c Level
Blood Glucose Level
```

Example (Heart Disease):

```
Age
Weight
Height
BMI
Blood Pressure
Heart Rate
Blood Sugar
Cholesterol
Lifestyle Factors
```

---

### 3️⃣ Flask receives form data

Example:

```python
age = float(request.form["age"])
bmi = float(request.form["bmi"])
hbA1c_level = float(request.form["hbA1c"])
```

---

### 4️⃣ Data converted to DataFrame

```
input_df = pd.DataFrame([input_data])
```

---

### 5️⃣ Model prediction

```
prob = model.predict_proba(input_df)[0][1]
```

---

### 6️⃣ Risk classification

```
if prob < 0.30 → Low Risk
if prob < 0.60 → Moderate Risk
else → High Risk
```

---

### 7️⃣ Result displayed to user

Example output:

```
Disease: Diabetes
Prediction: Diabetic
Probability: 67%
Risk Level: High Risk
```

---

# 📊 Model Evaluation Metrics

### Accuracy

Measures overall prediction correctness.

```
Accuracy = Correct Predictions / Total Predictions
```

---

### Precision

Measures correctness of predicted positives.

```
Precision = TP / (TP + FP)
```

---

### Recall

Measures how many real positive cases were detected.

```
Recall = TP / (TP + FN)
```

---

### Confusion Matrix

```
             Predicted
             No   Yes

Actual No    TN   FP
Actual Yes   FN   TP
```

---

# 💻 Installation (Run Locally)

### Clone Repository

```
git clone https://github.com/pawan0110/MedPredict.git
```

---

### Navigate to project

```
cd MedPredict
```

---

### Create virtual environment

```
python -m venv venv
```

Activate environment (Windows)

```
venv\Scripts\activate
```

---

### Install dependencies

```
pip install -r requirements.txt
```

---

### Run application

```
python app.py
```

---

### Open browser

```
http://127.0.0.1:5000
```

---

# ☁ Deployment

The application is deployed on **Render Cloud Platform**.

Deployment workflow:

```
GitHub Repository
        ↓
Render connects to repo
        ↓
Install dependencies
        ↓
Run Flask application
        ↓
Public URL generated
```

Live App:

[https://medpredict-nlff.onrender.com](https://medpredict-nlff.onrender.com)

---

# ⚠ Disclaimer

This project is developed **for educational and research purposes only**.

It should **not be used as a medical diagnostic tool** or substitute professional healthcare advice.

---

# 👨‍💻 Author

**Pawan Kumar**

GitHub
[https://github.com/pawan0110](https://github.com/pawan0110)

---


