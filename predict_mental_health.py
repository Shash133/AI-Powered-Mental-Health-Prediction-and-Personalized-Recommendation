import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('mental_health_model.h5')

# Default values for missing data
intDefault = 0
strDefault = 'NaN'
floatDefault = 0.0

def preprocess_input(symptom_input):
    # Convert input dictionary to DataFrame
    df_input = pd.DataFrame([symptom_input])

    # Drop unnecessary columns
    df_input = df_input.drop(['Timestamp', 'comments'], axis=1, errors='ignore')

    # Add missing columns with default values (if any are missing)
    required_columns = ['Age', 'Gender', 'self_employed', 'work_interfere', 'family_history', 'benefits', 'care_options', 
                        'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 
                        'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 
                        'mental_vs_physical', 'obs_consequence']
    
    for col in required_columns:
        if col not in df_input.columns:
            if col == 'self_employed':
                df_input[col] = 'No'  # Default value for 'self_employed'
            else:
                df_input[col] = 0  # Default for other columns (you can adjust this based on the column type)

    # Now proceed with the rest of your processing as before
    # Handle missing values by filling NaNs with defaults
    intColumns = ['Age']
    strColumns = ['Gender', 'self_employed', 'work_interfere', 'family_history', 'benefits', 'care_options', 
                  'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 
                  'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                  'mental_vs_physical', 'obs_consequence']

    for col in df_input.columns:
        if col in intColumns:
            df_input[col] = df_input[col].fillna(intDefault)
        elif col in strColumns:
            df_input[col] = df_input[col].fillna(strDefault)

    # Gender mapping (clean up non-standard terms)
    male_terms = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
    female_terms = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]
    trans_terms = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer"]

    for _, row in df_input.iterrows():
        if str.lower(row.Gender) in male_terms:
            df_input.loc[df_input['Gender'] == row.Gender, 'Gender'] = 'male'
        elif str.lower(row.Gender) in female_terms:
            df_input.loc[df_input['Gender'] == row.Gender, 'Gender'] = 'female'
        elif str.lower(row.Gender) in trans_terms:
            df_input.loc[df_input['Gender'] == row.Gender, 'Gender'] = 'trans'

    # Handle Age outliers (same as in the training)
    df_input['Age'] = df_input['Age'].fillna(df_input['Age'].median())
    df_input['Age'][df_input['Age'] < 18] = df_input['Age'].median()
    df_input['Age'][df_input['Age'] > 120] = df_input['Age'].median()

    # Age range binning (same as training)
    df_input['age_range'] = pd.cut(df_input['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

    # Self-employed and work interfere handling
    df_input['self_employed'] = df_input['self_employed'].replace([strDefault], 'No')
    df_input['work_interfere'] = df_input['work_interfere'].replace([strDefault], 'Don\'t know')

    # Label encoding for categorical features
    label_encoder = LabelEncoder()
    df_input = df_input.apply(label_encoder.fit_transform)

    # Feature scaling (MinMaxScaler for Age)
    scaler = MinMaxScaler()
    df_input['Age'] = scaler.fit_transform(df_input[['Age']])

    # Selecting only the relevant features as used in training
    feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
    df_input = df_input[feature_cols]

    return df_input

# Function to predict mental health status from symptom data
def predict(symptom_input):
    # Preprocess the input data
    processed_input = preprocess_input(symptom_input)

    # Make the prediction
    prediction = model.predict(processed_input)

    # Return the prediction (0 or 1)
    if prediction > 0.5:
        return "The individual is predicted to have mental health concerns. It is recommended to seek further support and intervention."
    else:
        return "The individual is predicted to be in good mental health, with no major concerns identified."

#import argparse

# Function to get user input interactively
def get_input():
    symptom_input = {}

    # Prompt the user for each feature with explanations
    print("\nPlease enter the following details:")

    # Age
    print("Age: Age of the individual (typically between 18 and 100)")
    symptom_input['Age'] = int(input("Enter your age (e.g., 29): "))  # Age of the individual
    
    # Gender
    print("Gender: Specify whether the individual is male, female, or trans.")
    symptom_input['Gender'] = input("Enter your gender (male/female/trans): ").lower()  # Gender of the individual
    
    print("Family History: Does the individual have a family history of mental health issues (1 for Yes, 0 for No)?")
    symptom_input['family_history'] = int(input("Enter family history (1 for Yes, 0 for No): "))  # Family history of mental health issues
    
    # Employee Benefits
    print("Benefits: Does the individual have access to employee benefits like health insurance or paid leave?")
    symptom_input['benefits'] = int(input("Enter benefits (Employee Benefits, 1 for Yes, 0 for No): "))  # Whether the individual has access to employee benefits
    
    print("Care Options: Does the individual have access to mental health care resources such as therapy or counseling?")
    symptom_input['care_options'] = int(input("Enter care options (access to mental health care resources) (1 for Yes, 0 for No): "))  # Access to mental health care resources

    print("Anonymity: Does the individual feel their identity is protected (e.g., anonymous participation in the survey)?")
    symptom_input['anonymity'] = int(input("Enter anonymity (0 for No, 1 for Yes): "))  # Whether the individual feels their identity is anonymous in the survey/intervention

    # Leave
    print("Leave: Does the individual have access to paid time off or sick leave for personal or mental health reasons?")
    symptom_input['leave'] = int(input("Enter leave (1 for Yes, 0 for No): "))  # Whether the individual has access to leave (e.g., paid time off or sick leave)

    # Work Interference
    print("Work Interference: Does mental health interfere with the individual's work performance?")
    symptom_input['work_interfere'] = int(input("Enter work interference (1 for Yes, 0 for No): "))  # Whether mental health issues interfere with work performance

    return symptom_input


def predict(user_input):
    # Assuming a trained model is loaded and we perform predictions here
    # For now, we will simply return a mocked prediction output
    predicted_status = "Needs Treatment" if user_input['work_interfere'] == 1 else "No Treatment Needed"
    return predicted_status


def main():
    # Command-line arguments parsing
    parser = argparse.ArgumentParser(description="Mental Health Prediction and Recommendation CLI")
    
    # Define arguments for user input (optional because we also use `get_input`)
    parser.add_argument('--age', type=int, help='Enter Age')
    parser.add_argument('--gender', type=int, choices=[0, 1], help='Enter Gender (0 = Female, 1 = Male)')
    parser.add_argument('--family_history', type=int, choices=[0, 1], help='Enter family history (1 for Yes, 0 for No)')
    parser.add_argument('--benefits', type=int, choices=[0, 1], help='Enter benefits (Employee Benefits, 1 for Yes, 0 for No)')
    parser.add_argument('--care_options', type=int, choices=[0, 1], help='Enter care options (1 for Yes, 0 for No)')
    parser.add_argument('--anonymity', type=int, choices=[0, 1], help='Enter anonymity (0 for No, 1 for Yes)')
    parser.add_argument('--leave', type=int, choices=[0, 1], help='Enter leave (1 for Yes, 0 for No)')
    parser.add_argument('--work_interfere', type=int, choices=[0, 1], help='Enter work interference (1 for Yes, 0 for No)')

    args = parser.parse_args()

    # Use either command-line input or interactive input (if no command-line arguments provided)
    if args.age is None:
        user_input = get_input()
    else:
        user_input = {
            'Age': args.age,
            'Gender': args.gender,
            'family_history': args.family_history,
            'benefits': args.benefits,
            'care_options': args.care_options,
            'anonymity': args.anonymity,
            'leave': args.leave,
            'work_interfere': args.work_interfere
        }

    # Get the prediction result
    predicted_status = predict(user_input)
    
    # Display the result
    print(f"Prediction: {predicted_status}")


if __name__ == "__main__":
    main()

# Get the predicted mental health status

