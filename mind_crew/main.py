# main.py
from predict import predict_disease, symptom_list

import re

def extract_symptoms(user_input):
    """
    Extract symptoms from a user input string by matching words/phrases from symptom_list.
    :param user_input: string
    :return: list of symptoms found in the input
    """
    user_input = user_input.lower()
    # Remove punctuation
    user_input = re.sub(r'[^\w\s]', ' ', user_input)

    extracted = []
    for symptom in symptom_list:
        # Check if symptom appears as a whole word in the input
        pattern = r'\b' + re.escape(symptom.lower()) + r'\b'
        if re.search(pattern, user_input):
            extracted.append(symptom)

    return extracted

def main():
    print("=== Disease Symptom Prediction ===")
    user_input = input("Please describe your symptoms:\n")
    
    user_symptoms = extract_symptoms(user_input)
    
    if not user_symptoms:
        print("No recognizable symptoms found in your input. Please try again.")
        return

    result = predict_disease(user_symptoms)

    print("\nPredicted Disease:", result["disease"])
    print("\nDescription:")
    print(result["description"])
    print("\nPrecautions:")
    for p in result["precautions"]:
        print("-", p)

if __name__ == "__main__":
    main()
