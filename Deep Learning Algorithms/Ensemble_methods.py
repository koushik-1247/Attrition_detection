import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv('attrition.csv') 

#Encode
categorical_column = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Workmode', 'Appreciation', 'Toxic culture', 'Bad Hiring process']
encoder=LabelEncoder()
df[categorical_column]=df[categorical_column].apply(encoder.fit_transform)

X = df[['Age','DailyRate','DistanceFromHome','Education','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','Workmode', 'Appreciation', 'Toxic culture', 'Bad Hiring process']]
y = df['Attrition']
df.dropna(inplace=True)

#Imputation
imputer = SimpleImputer(strategy='mean') 
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define base classifiers
base_classifier1 = DecisionTreeClassifier(random_state=45)
base_classifier2 = RandomForestClassifier(n_estimators=30, random_state=45)
base_classifier3 = LogisticRegression(solver='lbfgs', max_iter=1000)

# Stacking Classifier
estimators = [('dt', base_classifier1), ('rf', base_classifier2), ('lr', base_classifier3)]
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_classifier.fit(X_train, y_train)
stacking_predictions = stacking_classifier.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_predictions)
print(f'Stacking Classifier Accuracy: {stacking_accuracy:.2f}')
acc=int(round(stacking_accuracy,2)*100)
print("The percentage of times this model shows the Attrition rate correctly is:")
print(acc,'%')

# Bagging Classifier
bagging_classifier = BaggingClassifier(base_classifier1, n_estimators=10, random_state=42)
bagging_classifier.fit(X_train, y_train)
bagging_predictions = bagging_classifier.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)
print(f'Bagging Classifier Accuracy: {bagging_accuracy:.2f}')
acc=int(round(bagging_accuracy,2)*100)
print("The percentage of times this model shows the Attrition rate correctly is:")
print(acc,'%')

# AdaBoost Classifier
adaboost_classifier = AdaBoostClassifier(base_classifier2, n_estimators=20, random_state=42)
adaboost_classifier.fit(X_train, y_train)
adaboost_predictions = adaboost_classifier.predict(X_test)
adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)
print(f'AdaBoost Classifier Accuracy: {adaboost_accuracy:.2f}')
acc=int(round(adaboost_accuracy,2)*100)
print("The percentage of times this model shows the Attrition rate correctly is:")
print(acc,'%')
