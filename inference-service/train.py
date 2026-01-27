import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


dataset_path = kagglehub.dataset_download("stephanmatzka/predictive-maintenance-dataset-ai4i-2020")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models')

df = pd.read_csv(dataset_path + "/ai4i2020.csv")

# 2. Preprocessing
# Drop UDI and Product ID as they are unique identifiers with no predictive power
df = df.drop(['UDI', 'Product ID'], axis=1)

# Encode the 'Type' column (L, M, H) into numbers (0, 1, 2)
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# 3. Define Features (X) and Target (y)
X = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']

# 4. Split the data for trainand test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train  Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Save the model and the label encoder 
joblib.dump(model, os.path.join(model_path, 'model.pkl'))
joblib.dump(le, os.path.join(model_path, 'label_encoder.pkl'))

print(f"Model and Encoder saved in the '{model_path}' folder.")