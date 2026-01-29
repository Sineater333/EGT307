import kagglehub
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# 1. Setup Paths
dataset_path = kagglehub.dataset_download("stephanmatzka/predictive-maintenance-dataset-ai4i-2020")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create the directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 2. Load and Preprocess
df = pd.read_csv(os.path.join(dataset_path, "ai4i2020.csv"))
df_clean = df.drop(['UDI', 'Product ID'], axis=1)

# Encode 'Type' (L, M, H)
le = LabelEncoder()
df_clean['Type'] = le.fit_transform(df_clean['Type'])

# --- STAGE 1: Train/Eval the Binary Classifier (Is it broken?) ---
X = df_clean.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1) #drop random failure as it is noise
y_binary = df_clean['Machine failure']

X_train, X_test, y_binary_train, y_binary_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# SMOTE to handle the heavy imbalance of "No Failure" vs "Failure"
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_binary_train)

binary_model = RandomForestClassifier(n_estimators=100, random_state=42)
binary_model.fit(X_res, y_res)

# --- Lower false alarm rate ---
threshold = 0.55  # Increase this to reduce False Alarms (0.5 is default)
y_probs = binary_model.predict_proba(X_test)[:, 1]

# Evaluate Stage 1
y_pred_binary_custom = (y_probs >= threshold).astype(int)
binary_report = classification_report(y_binary_test, y_pred_binary_custom)

# --- STAGE 2: Train/Eval the Specialist (Why is it broken?) ---
# We isolate ONLY the rows where a failure occurred
failure_df = df_clean[df_clean['Machine failure'] == 1].copy()

X_f = failure_df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
# Combine 4 failure columns into one target column
y_f = failure_df[['TWF', 'HDF', 'PWF', 'OSF']].idxmax(axis=1)

# Split failure data into train/test to evaluate the "diagnostician"
X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_f, y_f, test_size=0.2, random_state=42)

type_model = RandomForestClassifier(n_estimators=100, random_state=42)
type_model.fit(X_f_train, y_f_train)

# Evaluate Stage 2
y_pred_type = type_model.predict(X_f_test)
type_report = classification_report(y_f_test, y_pred_type)

# --- 3. Results and Saving ---
print("--- STAGE 1: BINARY MODEL PERFORMANCE ---")
print(binary_report)

print("\n--- STAGE 2: TYPE MODEL (SPECIALIST) PERFORMANCE ---")
print(type_report)

# Save Models
joblib.dump(binary_model, os.path.join(MODEL_DIR, 'binary_model.pkl'))
joblib.dump(type_model, os.path.join(MODEL_DIR, 'type_model.pkl'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

# Save combined metrics to file
with open(os.path.join(MODEL_DIR, 'metrics.txt'), 'w') as f:
    f.write(f"THRESHOLD USED: {threshold}\n")
    f.write("STAGE 1: BINARY CLASSIFIER\n")
    f.write(binary_report)
    f.write("\n" + "="*50 + "\n")
    f.write("STAGE 2: FAILURE TYPE SPECIALIST\n")
    f.write(type_report)

print(f"\nSaved with high-precision threshold {threshold} in: {MODEL_DIR}")