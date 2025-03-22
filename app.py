import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("./csv/Synthetic_Financial_datasets_log.csv")  # Replace with the actual CSV file name

# 🔹 Drop Unnecessary Columns (Transaction ID is not needed)
df.drop(columns=['step', 'nameOrig', 'nameDest'], inplace=True)

# 🔹 Encode Categorical Columns (Transaction Type)
df['type'] = LabelEncoder().fit_transform(df['type'])  # Encode "PAYMENT", "TRANSFER", etc.

# 🔹 Feature Engineering (Create New Features)
df['balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['dest_balance_change'] = df['oldbalanceDest'] - df['newbalanceDest']

# 🔹 Define Features & Target
X = df.drop(columns=['isFraud', 'isFlaggedFraud'])
y = df['isFraud']  # We predict fraudulent transactions

# 🔹 Handle Imbalanced Data (Fraud Cases are Rare)
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Oversample fraud cases
X_resampled, y_resampled = smote.fit_resample(X, y)

# 🔹 Normalize Data (Scaling for Better Model Performance)
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# 🔹 Split Data (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 🔹 Train XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# 🔹 Evaluate Model
y_pred = xgb_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔹 Save Model for Future Use
import joblib
joblib.dump(xgb_model, "fraud_detection_model.pkl")
