import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.optimizers import Adam

#Data preprocessing
df = pd.read_csv('attrition.csv')  
categorical_column = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Workmode', 'Appreciation', 'Toxic culture', 'Bad Hiring process']
encoder=LabelEncoder()
df[categorical_column]=df[categorical_column].apply(encoder.fit_transform)

X = df[['Age','DailyRate','DistanceFromHome','Education','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','Workmode', 'Appreciation','Toxic culture','Bad Hiring process']]
y = df['Attrition']

#Imputation
imputer = SimpleImputer(strategy='mean') 
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df.dropna(inplace=True)


# base classifier
base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# BaggingClassifier
n_estimators = 100  # Number of base classifiers (you can adjust this)
bagging_model = BaggingClassifier(base_classifier, n_estimators=n_estimators, random_state=42)

# Train the BaggingClassifier
bagging_model.fit(X_train, y_train)

# Make predictions 
y_pred = bagging_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Bagging algorithm: {accuracy}')
acc=int(round(accuracy,2)*100)
print("The percentage of times this model shows the Attrition rate correctly is:")
print(acc,'%')
