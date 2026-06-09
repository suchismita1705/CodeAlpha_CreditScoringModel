import pandas as pd
import numpy as np
import joblib
import os

def load_production_artifacts():
    """Load the trained model and calibrated scaler from the project paths."""
    model_path = 'saved_models/credit_scoring_rf_model.joblib'
    scaler_path = 'saved_models/credit_scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Production artifacts missing! Ensure the model and scaler exist in your saved_models folder."
        )
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def run_credit_scoring_pipeline(new_customer_data):
    """Processes raw customer data and returns a credit scoring decision."""
    # 1. Load our saved machine learning dependencies
    model, scaler = load_production_artifacts()
    
    # 2. Convert input dictionary to a starting DataFrame
    df_input = pd.DataFrame([new_customer_data])
    
    # 3. Apply Feature Engineering & Native Calculations
    df_input['loan_to_income_ratio'] = df_input['loan_amnt'] / df_input['person_income']
    df_input['loan_percent_income'] = df_input['loan_amnt'] / df_input['person_income']
    
    # 4. Reconstruct the exact 23 columns in their precise training order
    training_features = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
        'loan_percent_income', 'cb_person_nodefault_placeholder', # assigned below dynamically
        'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
        'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E', 'loan_grade_F', 'loan_grade_G',
        'cb_person_default_on_file_Y', 'loan_to_income_ratio'
    ]
    
    # Overwrite the placeholder to ensure exact naming replication
    training_features[6] = 'cb_person_cred_hist_length'
    
    # Create an empty DataFrame structured with all 23 columns filled with zeros
    df_processed = pd.DataFrame(0, index=[0], columns=training_features)
    
    # Map numerical inputs directly across
    numerical_cols = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_to_income_ratio'
    ]
    for col in numerical_cols:
        df_processed[col] = df_input[col].values
        
    # Map categorical text items by dynamically switching their specific dummy column to 1
    home_col = f"person_home_ownership_{new_customer_data['person_home_ownership']}"
    if home_col in df_processed.columns:
        df_processed[home_col] = 1
        
    intent_col = f"loan_intent_{new_customer_data['loan_intent']}"
    if intent_col in df_processed.columns:
        df_processed[intent_col] = 1
        
    grade_col = f"loan_grade_{new_customer_data['loan_grade']}"
    if grade_col in df_processed.columns:
        df_processed[grade_col] = 1
        
    if new_customer_data['cb_person_default_on_file'] == 'Y':
        df_processed['cb_person_default_on_file_Y'] = 1
        
    # 5. Apply Feature Scaling using the imported calibrated parameters
    scaled_features = scaler.transform(df_processed)
    
    # 6. Generate the risk assessment class and the probability confidence score
    risk_prediction = model.predict(scaled_features)[0]
    risk_probability = model.predict_proba(scaled_features)[0][1]
    
    # 7. Convert prediction code into actionable business profiles
    result = "BAD CREDIT RISK (Deny Application)" if risk_prediction == 1 else "GOOD CREDIT RISK (Approve Application)"
    
    print("\n=========================================")
    print("      REAL-TIME RISK DECISION ENGINE     ")
    print("=========================================")
    print(f"Applicant Evaluation:  {result}")
    print(f"Risk Probability:      {risk_probability * 100:.2f}%")
    print("=========================================\n")

if __name__ == "__main__":
    # Mocking a live incoming customer application profile
    sample_applicant = {
        'person_age': 25,
        'person_income': 40000,
        'person_emp_length': 2,
        'loan_amnt': 25000,              
        'loan_int_rate': 14.5,
        'cb_person_cred_hist_length': 3,   # Added missing history feature
        'person_home_ownership': 'RENT',
        'loan_intent': 'PERSONAL',
        'loan_grade': 'D',
        'cb_person_default_on_file': 'N'
    }
    
    run_credit_scoring_pipeline(sample_applicant)