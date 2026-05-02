import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ── Load artifacts ────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / 'models'

model       = joblib.load(MODELS_DIR / 'best_model_Random_Forest.pkl')
num_imputer = joblib.load(MODELS_DIR / 'preprocessor_num_imputer.pkl')
cat_imputer = joblib.load(MODELS_DIR / 'preprocessor_cat_imputer.pkl')
le_target   = joblib.load(MODELS_DIR / 'preprocessor_label_encoder.pkl')

# Use exact order the imputers were fitted with — guaranteed correct
num_cols = list(num_imputer.feature_names_in_)
cat_cols = list(cat_imputer.feature_names_in_)

print('✔ All artifacts loaded')
print(f'  Classes  : {le_target.classes_}')
print(f'  Num cols : {len(num_cols)}')
print(f'  Cat cols : {len(cat_cols)}')


# ── Preprocessing ─────────────────────────────────────────────
def preprocess(raw: dict) -> np.ndarray:
    # Lowercase + strip keys to match training column names
    raw_clean = {k.strip().lower(): v for k, v in raw.items()}
    df = pd.DataFrame([raw_clean])

    # Impute using exact fitted order
    df[num_cols] = num_imputer.transform(df[num_cols])
    df[cat_cols] = cat_imputer.transform(df[cat_cols])

    # Encode each categorical column independently
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df[num_cols + cat_cols].values


# ── Predict ───────────────────────────────────────────────────
def predict(raw: dict) -> dict:
    X = preprocess(raw)
    pred_int   = model.predict(X)[0]
    pred_label = le_target.inverse_transform([pred_int])[0]
    proba      = model.predict_proba(X)[0]
    confidence = float(proba[pred_int])

    return {
        'prediction':  pred_label,
        'confidence':  round(confidence, 4),
        'class_probs': {
            le_target.classes_[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
    }


# ── Sample patient ────────────────────────────────────────────
if __name__ == '__main__':
    sample_patient = {
        'age of the patient':                          45,
        'blood pressure (mm/hg)':                      80,
        'specific gravity of urine':                   1.020,
        'albumin in urine':                            1,
        'sugar in urine':                              0,
        'random blood glucose level (mg/dl)':          120,
        'blood urea (mg/dl)':                          36,
        'serum creatinine (mg/dl)':                    1.2,
        'sodium level (meq/l)':                        137,
        'potassium level (meq/l)':                     4.5,
        'hemoglobin level (gms)':                      15.4,
        'packed cell volume (%)':                      44,
        'white blood cell count (cells/cumm)':         7800,
        'red blood cell count (millions/cumm)':        5.2,
        'estimated glomerular filtration rate (egfr)': 85,
        'urine protein-to-creatinine ratio':           0.2,
        'urine output (ml/day)':                       1500,
        'serum albumin level':                         4.0,
        'cholesterol level':                           180,
        'parathyroid hormone (pth) level':             30,
        'serum calcium level':                         9.5,
        'serum phosphate level':                       3.5,
        'body mass index (bmi)':                       24.5,
        'duration of diabetes mellitus (years)':       5,
        'duration of hypertension (years)':            3,
        'cystatin c level':                            0.9,
        'c-reactive protein (crp) level':              2.1,
        'interleukin-6 (il-6) level':                  3.0,
        'red blood cells in urine':                    'normal',
        'pus cells in urine':                          'normal',
        'pus cell clumps in urine':                    'notpresent',
        'bacteria in urine':                           'notpresent',
        'hypertension (yes/no)':                       'yes',
        'diabetes mellitus (yes/no)':                  'yes',
        'coronary artery disease (yes/no)':            'no',
        'appetite (good/poor)':                        'good',
        'pedal edema (yes/no)':                        'no',
        'anemia (yes/no)':                             'no',
        'family history of chronic kidney disease':    'no',
        'smoking status':                              'non-smoker',
        'physical activity level':                     'moderate',
        'urinary sediment microscopy results':         'normal',
    }

    result = predict(sample_patient)

    print('\n' + '=' * 45)
    print('  PREDICTION RESULT')
    print('=' * 45)
    print(f"  Diagnosis   : {result['prediction'].upper()}")
    print(f"  Confidence  : {result['confidence'] * 100:.2f}%")
    print('\n  Class Probabilities:')
    for cls, prob in result['class_probs'].items():
        bar = '█' * int(prob * 30)
        print(f"    {cls:<20} {prob * 100:5.2f}%  {bar}")
    print('=' * 45)