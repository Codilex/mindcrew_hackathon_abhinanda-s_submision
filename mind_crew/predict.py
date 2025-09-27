# import joblib
# import numpy as np

# # Load saved artifacts
# model = joblib.load("models/svm_disease_model.pkl")
# encoder = joblib.load("models/label_encoder.pkl")
# symptom_list = joblib.load("models/symptom_list.pkl")

# def predict_disease(user_symptoms):
#     """
#     Predict disease from list of symptoms
#     :param user_symptoms: list of symptoms (strings)
#     :return: predicted disease name (string)
#     """
#     # Create one-hot encoding for symptoms
#     input_data = [0] * len(symptom_list)
#     for symptom in user_symptoms:
#         if symptom in symptom_list:
#             idx = symptom_list.index(symptom)
#             input_data[idx] = 1

#     # Convert to numpy array and reshape for model
#     input_data = np.array(input_data).reshape(1, -1)

#     # Predict
#     prediction = model.predict(input_data)[0]
#     disease = encoder.inverse_transform([prediction])[0]

#     return disease

# if __name__ == "__main__":
#     # Example usage
#     test_input = ["itching", "skin_rash", "nodal_skin_eruptions"]
#     result = predict_disease(test_input)
#     print("Predicted Disease:", result)

# import os
# print(os.getcwd())

# import joblib
# import numpy as np
# import pandas as pd

# # Load saved artifacts
# model = joblib.load("models/svm_disease_model.pkl")
# encoder = joblib.load("models/label_encoder.pkl")
# symptom_list = joblib.load("models/symptom_list.pkl")

# # Load precaution and description CSVs
# precaution_df = pd.read_csv("dataset/symptom_precaution.csv")
# description_df = pd.read_csv("dataset/symptom_description.csv")

# # Convert precautions to a dictionary for faster lookup
# disease_precautions = {}
# for _, row in precaution_df.iterrows():
#     disease = row["Disease"]
#     precautions = [row[f"Precaution_{i}"] for i in range(1, 5) if pd.notna(row[f"Precaution_{i}"])]
#     disease_precautions[disease] = precautions

# # Convert descriptions to a dictionary for faster lookup
# disease_descriptions = dict(zip(description_df["Disease"], description_df["Description"]))

# def predict_disease(user_symptoms):
#     """
#     Predict disease from list of symptoms and provide description + precautions
#     :param user_symptoms: list of symptoms (strings)
#     :return: dict with keys: 'disease', 'description', 'precautions'
#     """
#     # Create one-hot encoding for symptoms
#     input_data = [0] * len(symptom_list)
#     for symptom in user_symptoms:
#         if symptom in symptom_list:
#             idx = symptom_list.index(symptom)
#             input_data[idx] = 1

#     # Convert to numpy array and reshape for model
#     input_data = np.array(input_data).reshape(1, -1)

#     # Predict disease
#     prediction = model.predict(input_data)[0]
#     disease = encoder.inverse_transform([prediction])[0]

#     # Fetch precautions and description
#     precautions = disease_precautions.get(disease, ["No precautions available."])
#     description = disease_descriptions.get(disease, "No description available.")

#     return {
#         "disease": disease,
#         "description": description,
#         "precautions": precautions
#     }

# if __name__ == "__main__":
#     # Example usage
#     test_input = ["itching", "skin_rash", "nodal_skin_eruptions"]
#     result = predict_disease(test_input)
    
#     print("Predicted Disease:", result["disease"])
#     print("\nDescription:")
#     print(result["description"])
#     print("\nPrecautions:")
#     for p in result["precautions"]:
#         print("-", p)
import os
import joblib
import numpy as np
import pandas as pd

# Helper function to construct paths relative to the script location
def resource_path(*paths):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *paths)

# Load saved artifacts
model = joblib.load(resource_path("models", "svm_disease_model.pkl"))
encoder = joblib.load(resource_path("models", "label_encoder.pkl"))
symptom_list = joblib.load(resource_path("models", "symptom_list.pkl"))

# Load precaution and description CSVs
precaution_csv = resource_path("dataset", "symptom_precaution.csv")
description_csv = resource_path("dataset", "symptom_description.csv")

# Check if files exist
if not os.path.exists(precaution_csv):
    raise FileNotFoundError(f"{precaution_csv} not found")
if not os.path.exists(description_csv):
    raise FileNotFoundError(f"{description_csv} not found")

precaution_df = pd.read_csv(precaution_csv)
description_df = pd.read_csv(description_csv)

# Convert precautions to a dictionary for faster lookup
disease_precautions = {}
for _, row in precaution_df.iterrows():
    disease = row["Disease"]
    precautions = [row[f"Precaution_{i}"] for i in range(1, 5) if pd.notna(row[f"Precaution_{i}"])]
    disease_precautions[disease] = precautions

# Convert descriptions to a dictionary for faster lookup
disease_descriptions = dict(zip(description_df["Disease"], description_df["Description"]))

def predict_disease(user_symptoms):
    """
    Predict disease from list of symptoms and provide description + precautions
    :param user_symptoms: list of symptoms (strings)
    :return: dict with keys: 'disease', 'description', 'precautions'
    """
    # Create one-hot encoding for symptoms
    input_data = [0] * len(symptom_list)
    for symptom in user_symptoms:
        if symptom in symptom_list:
            idx = symptom_list.index(symptom)
            input_data[idx] = 1

    # Convert to numpy array and reshape for model
    input_data = np.array(input_data).reshape(1, -1)

    # Predict disease
    prediction = model.predict(input_data)[0]
    disease = encoder.inverse_transform([prediction])[0]

    # Fetch precautions and description
    precautions = disease_precautions.get(disease, ["No precautions available."])
    description = disease_descriptions.get(disease, "No description available.")

    return {
        "disease": disease,
        "description": description,
        "precautions": precautions
    }

if __name__ == "__main__":
    # Example usage
    test_input = ["itching", "skin_rash", "nodal_skin_eruptions"]
    result = predict_disease(test_input)
    
    print("Predicted Disease:", result["disease"])
    print("\nDescription:")
    print(result["description"])
    print("\nPrecautions:")
    for p in result["precautions"]:
        print("-", p)
