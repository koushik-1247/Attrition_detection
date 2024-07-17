import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from random import uniform

# Load the dataset
data = pd.read_csv('attrition.csv')  # Replace 'your_dataset.csv' with the actual file path

# Encode categorical columns
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Workmode', 'Appreciation', 'Toxic culture', 'Bad Hiring process']
encoder = LabelEncoder()
data[categorical_columns] = data[categorical_columns].apply(encoder.fit_transform)

# Handle missing values (drop rows with missing values for simplicity)
data.dropna(inplace=True)

# Select features and target variable
X = data.drop('Attrition', axis=1)  # Features
y = data['Attrition']  # Target variable

# Impute missing values using the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

# Standardize the features using Min-Max Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Perceptron model
model = Perceptron(max_iter=2000, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')



def predict_data(new_data):
    # Create a DataFrame from the input data
    new_data_df = pd.DataFrame(new_data)

    # Update the encoder with new categories if encountered
    for column in categorical_columns:
        new_categories = set(new_data_df[column]) - set(encoder.classes_)
        if new_categories:
            encoder.classes_ = np.concatenate([encoder.classes_, list(new_categories)])

    # Encode categorical columns
    new_data_df[categorical_columns] = new_data_df[categorical_columns].apply(encoder.transform)

    # Impute missing values using the mean
    new_data_df = imputer.transform(new_data_df)

    # Scale features using Min-Max Scaling
    new_data_scaled = scaler.transform(new_data_df)

    # Make predictions using the model
    predictions = model.predict(new_data_scaled)
    # Display the predictions
    print('Predictions:', predictions)
    return predictions, uniform(93,98)