import pandas as pd
import joblib
import argparse
import config

def predict(input_path):
    # Load model and preprocessor
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    model = joblib.load(config.MODEL_PATH)
    
    # Load data
    new_data = pd.read_csv(input_path)
    
    # Preprocessing
    if 'customerID' in new_data.columns:
        new_data.drop('customerID', axis=1, inplace=True)
    if 'TotalCharges' in new_data.columns:
        new_data['TotalCharges'] = pd.to_numeric(
            new_data['TotalCharges'], errors='coerce'
        )
        # Handle missing values if needed
        # new_data = new_data.dropna(subset=['TotalCharges'])
    
    # Transform data
    processed_data = preprocessor.transform(new_data)
    
    # Predict
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]
    
    # Save results
    result = new_data.copy()
    result['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
    result['Churn_Probability'] = probabilities
    
    output_path = input_path.replace('.csv', '_predicted.csv')
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Churn rate: {(result['Churn_Prediction'] == 'Yes').mean():.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict customer churn')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    args = parser.parse_args()
    predict(args.input)