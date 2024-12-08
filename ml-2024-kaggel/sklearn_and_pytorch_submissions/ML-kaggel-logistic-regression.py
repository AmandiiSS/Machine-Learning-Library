import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report
import csv

# Load train data 
data = pd.read_csv('data/train_final.csv')

# Divide categorical and numerical columns
categorical_cols = ['workclass', 'education','marital.status','occupation','relationship','race','sex', 'native.country']
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

# Handle missing values in categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Handle missing values in numerical columns (using median)
num_imputer = SimpleImputer(strategy='median')
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Convert categorical variables to one-hot encoding
data = pd.get_dummies(data, columns=categorical_cols)

# Scale numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Separate features and labels
X = data.drop('income>50K', axis=1) 
y = data['income>50K']  

# Train model
logr = linear_model.LogisticRegression(max_iter=1000)
logr.fit(X, y)




# Predictions on training set for evaluation
y_train_pred = logr.predict(X)
print(classification_report(y, y_train_pred))




# Load test data 
test_data = pd.read_csv('data/test_final.csv')
test_data_ID = test_data['ID']
test_data = test_data.drop('ID', axis=1) 

# Handle missing values in categorical columns
test_data[categorical_cols] = cat_imputer.transform(test_data[categorical_cols])

# Handle missing values in numerical columns (using median)
test_data[numerical_cols] = num_imputer.transform(test_data[numerical_cols])

# Convert categorical variables to one-hot encoding
test_data = pd.get_dummies(test_data, columns=categorical_cols)

# Align test data with training data: ensure both have the same columns
X_test = test_data.reindex(columns=X.columns, fill_value=0)

# Scale numerical features
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Predictions
predictions = logr.predict(X_test)

#Write the results in a CSV
submission_data = pd.DataFrame({'ID': test_data_ID, 'Prediction': predictions})
submission_data.to_csv('predictions/submission_Amanda_SS_logistic_test4.csv', index=False)
