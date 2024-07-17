import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier


df = pd.read_csv('attrition.csv') 
print(df.shape)
print(df.columns)

#Encode
categorical_column = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Workmode', 'Appreciation', 'Toxic culture', 'Bad Hiring process']
encoder=LabelEncoder()
df[categorical_column]=df[categorical_column].apply(encoder.fit_transform)

X = df[['Age','DailyRate','DistanceFromHome','Education','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]
y = df['Attrition']

#Imputation
imputer = SimpleImputer(strategy='mean') 
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df.dropna(inplace=True)


#base classifiers
base_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# Define the meta-classifier
meta_classifier = LogisticRegression()

# Create the stacking model
stacking_model = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)

# Fit the stacking model 
stacking_model.fit(X_train, y_train)

# Make predictions 
stacking_predictions = stacking_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, stacking_predictions)
print(f"Accuracy of the stacking model: {accuracy:.2f}")
acc=int(round(accuracy,2)*100)
print("The percentage of times this model shows the Attrition rate correctly is:")
print(acc,'%')
