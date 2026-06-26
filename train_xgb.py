import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import joblib
import os

# 1. Define Paths (Assuming this script is in your project root, next to app.py)
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
DATA_PATH = os.path.join(BASE_DIR, 'diabetes.csv')

# Ensure the model directory exists
os.makedirs(BASE_DIR, exist_ok=True)

print("Loading dataset...")
# 2. Load the Dataset
df = pd.read_csv(DATA_PATH)

# 3. Preprocessing: Zero-Imputation (Medians)
# Replacing 0s with medians for biologically impossible zero values
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_impute:
    median_val = df[col].replace(0, np.nan).median()
    df[col] = df[col].replace(0, median_val)

print("Engineering features...")
# 4. Feature Engineering (Must perfectly match app.py)
df['BMI_Age']         = df['BMI'] * df['Age']
df['Glucose_Insulin'] = df['Glucose'] / (df['Insulin'] + 1)
df['BP_Age']          = df['BloodPressure'] * df['Age']

# 5. Define Features (X) and Target (y)
# Explicitly ordering features so we can save the exact names
feature_columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
    'BMI', 'DiabetesPedigreeFunction', 'Age', 
    'BMI_Age', 'Glucose_Insulin', 'BP_Age'
]

X = df[feature_columns]
y = df['Outcome']

# 6. Train/Test Split
# Stratify ensures the 0/1 ratio remains the same in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Scaling data...")
# 7. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training XGBoost Classifier...")
# 8. Initialize and Train XGBoost Model
# These hyperparameters are a great starting point for this specific dataset
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

# 9. Evaluate the Model
y_pred = xgb_model.predict(X_test_scaled)
y_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"AUC Score: {auc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Save the Artifacts
print("\nSaving model artifacts to /model directory...")

# Save the new XGBoost model
joblib.dump(xgb_model, os.path.join(BASE_DIR, 'diabetes_xgb_model.pkl'))

# Save the scaler and feature names (Overrides existing to ensure 100% sync)
joblib.dump(scaler, os.path.join(BASE_DIR, 'diabetes_scaler.pkl'))
joblib.dump(feature_columns, os.path.join(BASE_DIR, 'feature_names.pkl'))

print("Done! You can now run your Flask app.")