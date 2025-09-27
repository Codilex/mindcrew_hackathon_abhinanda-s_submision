import pandas as pd
import os
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 1. Load dataset
df = pd.read_csv("dataset/dataset.csv")

# 2. Collect all unique symptoms
symptom_cols = [col for col in df.columns if col.startswith("Symptom")]
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().str.strip().unique())

all_symptoms = sorted(list(all_symptoms))  # fixed order
print(f"Total unique symptoms: {len(all_symptoms)}")

# 3. Create one-hot encoding for each row
def encode_symptoms(row):
    features = [0] * len(all_symptoms)
    for col in symptom_cols:
        symptom = row[col]
        if pd.notna(symptom):
            symptom = symptom.strip()
            if symptom in all_symptoms:
                features[all_symptoms.index(symptom)] = 1
    return features

X = df.apply(encode_symptoms, axis=1, result_type="expand")
y = df["Disease"]

# 4. Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 6. Simulated loader before training
print("\n🚀 Training SVM model...")
for _ in tqdm(range(100), desc="Training Progress", ncols=100):
    time.sleep(0.01)  # just to simulate progress

# 7. Train SVM
model = SVC(kernel="linear", probability=True, verbose=True)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# 9. Save model, encoder & symptom list
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/svm_disease_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")
joblib.dump(all_symptoms, "models/symptom_list.pkl")

print("\n✅ Model, encoder, and symptom list saved in 'models/' directory")
