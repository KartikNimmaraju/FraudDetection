import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 🔹 Load the synthetic dataset
synthetic_df = pd.read_csv("synthetic_fraud_data.csv")  # Ensure the file exists

# 🔹 Drop Unnecessary Columns
synthetic_df.drop(columns=['step', 'nameOrig', 'nameDest'], errors='ignore', inplace=True)

# 🔹 Encode Categorical Columns (Transaction Type)
synthetic_df['type'] = LabelEncoder().fit_transform(synthetic_df['type'])

# 🔹 Feature Engineering (Balance Changes)
synthetic_df['balancechange'] = synthetic_df['oldbalanceOrg'] - synthetic_df['newbalanceOrg']
synthetic_df['destbalancechange'] = synthetic_df['oldbalanceDest'] - synthetic_df['newbalanceDest']

# 🔹 Define Features & Target
X_synthetic = synthetic_df.drop(columns=['isFraud', 'isFlaggedFraud'], errors='ignore')
y_synthetic = synthetic_df['isFraud']

# 🔹 Load the Saved Model
xgb_model = joblib.load("fraud_detection_model.pkl")

# 🔹 Standardize Data (Using Same Scaling Process)
scaler = StandardScaler()
X_synthetic = scaler.fit_transform(X_synthetic)  # Note: Use same scaler used in training

# 🔹 Make Predictions
y_pred_synthetic = xgb_model.predict(X_synthetic)

# 🔹 Evaluate Model
print("🔹 Confusion Matrix:\n", confusion_matrix(y_synthetic, y_pred_synthetic))
print("\n🔹 Classification Report:\n", classification_report(y_synthetic, y_pred_synthetic))
