import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Churn_Model_Data.csv')

# Define features and target
numerical_cols = ['tenure', 'MonthlyCharges']
categorical_cols = ['PaymentMethod', 'SeniorCitizen', 'Partner', 'Dependents', 'OnlineSecurity', 
                    'TechSupport', 'Contract', 'PaperlessBilling']
target = 'Churn'

# Split data into features and target
X = data[numerical_cols + categorical_cols]
y = data[target]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler and encoder
scaler = StandardScaler()
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Fit and transform the training data
X_train_numerical = scaler.fit_transform(X_train[numerical_cols])
X_train_categorical = encoder.fit_transform(X_train[categorical_cols])

# Combine numerical and categorical features
X_train_processed = np.hstack((X_train_numerical, X_train_categorical))

# Train the model
model = XGBClassifier()
model.fit(X_train_processed, y_train)

# Save the trained model, scaler, and encoder
with open('xgb_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

print("Model training and saving completed.")
