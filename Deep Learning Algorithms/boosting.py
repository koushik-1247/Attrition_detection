import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
data = pd.read_csv('attrition.csv')
target_column = 'Attrition'
data[target_column] = data[target_column].map({'Yes': 1, 'No': 0})

feature_columns = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Workmode', 'Appreciation', 'Toxic culture', 'Bad Hiring process']

X = data[feature_columns]
y = data[target_column]

# encode categorical features
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Workmode', 'Appreciation', 'Toxic culture', 'Bad Hiring process']
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost 
base_learner = XGBClassifier(
    booster='gbtree', 
    max_depth=5,       
    learning_rate=0.2,  
    n_estimators=100,   
    subsample=0.8,     
    colsample_bytree=0.8,  
    objective='binary:logistic'  
)


base_learner.fit(X_train, y_train)
y_pred = base_learner.predict(X_test)

# Evaluate 
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
acc=int(round(accuracy,2)*100)
print("The percentage of times this model shows the Attrition rate correctly is:")
print(acc,'%')

