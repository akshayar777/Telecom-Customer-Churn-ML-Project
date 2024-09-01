from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model, scaler, and encoder
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

def preprocess_input(data):
    # Define numerical and categorical columns
    numerical_cols = ['tenure', 'MonthlyCharges']
    categorical_cols = ['PaymentMethod', 'SeniorCitizen', 'Partner', 'Dependents', 'OnlineSecurity', 'TechSupport', 'Contract', 'PaperlessBilling']
    
    # Scale numerical data
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    # Encode categorical data
    encoded_data = encoder.transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Concatenate scaled and encoded data
    processed_data = pd.concat((data[numerical_cols], encoded_df), axis=1)
    
    return processed_data


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_churn():
    if request.method == 'POST':
        # Get form data
        input_data = {
            'tenure': [float(request.form['tenure'])],
            'MonthlyCharges': [float(request.form['MonthlyCharges'])],
            'PaymentMethod': [(request.form['PaymentMethod'])],
            'SeniorCitizen': [request.form['SeniorCitizen']],
            'Partner': [request.form['Partner']],
            'Dependents': [request.form['Dependents']],
            'OnlineSecurity': [request.form['OnlineSecurity']],
            'TechSupport': [request.form['TechSupport']],
            'Contract': [request.form['Contract']],
            'PaperlessBilling': [request.form['PaperlessBilling']]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # Preprocess input data
        processed_data = preprocess_input(input_df)

        # Make prediction
        prediction = model.predict(processed_data)

        # Convert prediction to readable format
        churn_prediction = 'churn' if prediction[0] == 1 else 'not churn'

        # Render result template
        return render_template('result.html', prediction=churn_prediction)

if __name__ == '__main__':
    app.run(debug=True)
