"""
Diabetes Prediction Web Application
Flask backend with ML Ensemble (Gradient Boosting + XGBoost)
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import os
import xgboost as xgb # Ensure this is installed

app = Flask(__name__)

# Load model artifacts
BASE = os.path.dirname(os.path.abspath(__file__))

# Load BOTH models for the ensemble
gb_model  = joblib.load(os.path.join(BASE, 'model', 'diabetes_gb_model.pkl'))
xgb_model = joblib.load(os.path.join(BASE, 'model', 'diabetes_xgb_model.pkl'))

scaler  = joblib.load(os.path.join(BASE, 'model', 'diabetes_scaler.pkl'))
features = joblib.load(os.path.join(BASE, 'model', 'feature_names.pkl'))

# ── Dataset stats for charts ─────────────────────────────────────────────────
df_raw = pd.read_csv(os.path.join(BASE, 'model', 'diabetes.csv'))
for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    median = df_raw[col].replace(0, np.nan).median()
    df_raw[col] = df_raw[col].replace(0, median)

# Feature engineering (must match training)
df_raw['BMI_Age']          = df_raw['BMI'] * df_raw['Age']
df_raw['Glucose_Insulin']  = df_raw['Glucose'] / (df_raw['Insulin'] + 1)
df_raw['BP_Age']           = df_raw['BloodPressure'] * df_raw['Age']


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extracting data directly
        pregnancies    = float(data.get('pregnancies', 0))
        glucose        = float(data.get('glucose', 120))
        blood_pressure = float(data.get('blood_pressure', 70))
        skin_thickness = float(data.get('skin_thickness', 20))
        insulin        = float(data.get('insulin', 80))
        bmi            = float(data.get('bmi', 30.0))
        dpf            = float(data.get('dpf', 0.5))
        age            = float(data.get('age', 33))

        # Engineered features
        bmi_age         = bmi * age
        glucose_insulin = glucose / (insulin + 1)
        bp_age          = blood_pressure * age

        # Construct DataFrame to ensure feature order matches training exactly
        input_data = pd.DataFrame([{
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf, 
            'Age': age,
            'BMI_Age': bmi_age,
            'Glucose_Insulin': glucose_insulin,
            'BP_Age': bp_age
        }])
        
        # Ensure column order matches the loaded features list
        input_data = input_data[features]
        input_scaled = scaler.transform(input_data)

        # ── ENSEMBLE PREDICTION LOGIC ──
        # Get probabilities from both models
        gb_prob = float(gb_model.predict_proba(input_scaled)[0][1])
        xgb_prob = float(xgb_model.predict_proba(input_scaled)[0][1])
        
        # Average the probabilities (Soft Voting)
        ensemble_probability = (gb_prob + xgb_prob) / 2
        
        # Final prediction: 1 if combined probability >= 50%, else 0
        prediction = 1 if ensemble_probability >= 0.5 else 0

        # Risk factors
        risk_factors = []
        if glucose > 140:
            risk_factors.append({'factor': 'High Glucose', 'level': 'high', 'value': f'{glucose:.0f} mg/dL'})
        elif glucose > 100:
            risk_factors.append({'factor': 'Elevated Glucose', 'level': 'medium', 'value': f'{glucose:.0f} mg/dL'})

        if bmi > 30:
            risk_factors.append({'factor': 'Obesity (BMI)', 'level': 'high', 'value': f'{bmi:.1f}'})
        elif bmi > 25:
            risk_factors.append({'factor': 'Overweight (BMI)', 'level': 'medium', 'value': f'{bmi:.1f}'})

        if blood_pressure > 90:
            risk_factors.append({'factor': 'High Blood Pressure', 'level': 'high', 'value': f'{blood_pressure:.0f} mmHg'})
        elif blood_pressure > 80:
            risk_factors.append({'factor': 'Elevated Blood Pressure', 'level': 'medium', 'value': f'{blood_pressure:.0f} mmHg'})

        if age > 45:
            risk_factors.append({'factor': 'Age Risk', 'level': 'medium', 'value': f'{age:.0f} years'})

        if dpf > 0.8:
            risk_factors.append({'factor': 'Genetic Predisposition', 'level': 'high', 'value': f'{dpf:.3f}'})

        if insulin > 200:
            risk_factors.append({'factor': 'High Insulin', 'level': 'medium', 'value': f'{insulin:.0f} μU/mL'})

        # Recommendations
        recommendations = []
        if prediction == 1:
            recommendations = [
                'Consult an endocrinologist or diabetologist immediately',
                'Monitor blood glucose levels daily',
                'Follow a low-glycemic diet plan',
                'Engage in at least 30 min moderate exercise daily',
                'Reduce refined carbohydrates and sugary drinks',
                'Schedule regular HbA1c tests every 3 months',
            ]
        else:
            recommendations = [
                'Maintain a balanced, nutritious diet',
                'Exercise regularly (150 min/week moderate activity)',
                'Monitor blood glucose annually',
                'Maintain a healthy weight (BMI 18.5–24.9)',
                'Limit alcohol and avoid smoking',
                'Schedule routine health check-ups every year',
            ]

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': round(ensemble_probability * 100, 1),
            'model_breakdown': {
                'gb_probability': round(gb_prob * 100, 1),
                'xgb_probability': round(xgb_prob * 100, 1)
            },
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'input_values': {
                'Pregnancies': pregnancies, 'Glucose': glucose,
                'Blood Pressure': blood_pressure, 'Skin Thickness': skin_thickness,
                'Insulin': insulin, 'BMI': bmi,
                'DPF': dpf, 'Age': age
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/stats')
def stats():
    """Dataset statistics for the dashboard charts."""
    diabetic     = df_raw[df_raw['Outcome'] == 1]
    non_diabetic = df_raw[df_raw['Outcome'] == 0]

    # Age distribution buckets
    bins   = [20, 30, 40, 50, 60, 70, 82]
    labels = ['21–30', '31–40', '41–50', '51–60', '61–70', '70+']
    d_ages  = pd.cut(diabetic['Age'],     bins=bins, labels=labels).value_counts().sort_index()
    nd_ages = pd.cut(non_diabetic['Age'], bins=bins, labels=labels).value_counts().sort_index()

    # Feature averages by outcome
    feat_cols = ['Glucose','BloodPressure','BMI','Insulin','Age']
    feat_avg_d  = diabetic[feat_cols].mean().round(1).to_dict()
    feat_avg_nd = non_diabetic[feat_cols].mean().round(1).to_dict()

    return jsonify({
        'total':         len(df_raw),
        'diabetic':      int(df_raw['Outcome'].sum()),
        'non_diabetic':  int((df_raw['Outcome'] == 0).sum()),
        'accuracy':      83.1, # Updated example accuracy
        'age_labels':    labels,
        'age_diabetic':  d_ages.tolist(),
        'age_non_diab':  nd_ages.tolist(),
        'feat_labels':   feat_cols,
        'feat_diabetic': [feat_avg_d[c]  for c in feat_cols],
        'feat_non_diab': [feat_avg_nd[c] for c in feat_cols],
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)