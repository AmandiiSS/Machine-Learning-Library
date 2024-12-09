from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load train data
data = pd.read_csv('data/train_final.csv')

# Divide categorical and numerical columns
categorical_cols = ['workclass', 'education','marital.status','occupation','relationship','race','sex', 'native.country']
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

# Handle missing values
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='median')

data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Convert categorical variables to one-hot encoding
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Separate features and labels
X = data.drop('income>50K', axis=1)
y = data['income>50K']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize XGBoost Classifier
xgb = XGBClassifier(objective='binary:logistic',eval_metric='logloss',use_label_encoder=False,random_state=42)

# Define hyperparameters for GridSearch
param_grid = {'n_estimators': [50, 100, 200],'max_depth': [3, 5, 7],'learning_rate': [0.01, 0.1, 0.2],'subsample': [0.8, 1.0],'colsample_bytree': [0.8, 1.0]}

# GridSearch to find the best 0
grid_search = GridSearchCV(estimator=xgb,param_grid=param_grid,cv=3,scoring='accuracy',verbose=1,n_jobs=-1)

grid_search.fit(X_train, y_train)

# Best parameters and model
best_xgb = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate on the test set
y_pred = best_xgb.predict(X_test)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Load and preprocess test data
test_data = pd.read_csv('data/test_final.csv')
test_data_ID = test_data['ID']
test_data = test_data.drop('ID', axis=1)

# Handle missing values
test_data[categorical_cols] = cat_imputer.transform(test_data[categorical_cols])
test_data[numerical_cols] = num_imputer.transform(test_data[numerical_cols])

# One-hot encode categorical variables
test_data = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

# Align columns with training data
X_test_final = test_data.reindex(columns=X.columns, fill_value=0)

# Scale numerical features
X_test_final[numerical_cols] = scaler.transform(X_test_final[numerical_cols])

# Predict test data
predictions = best_xgb.predict(X_test_final)

# Write predictions to a CSV
submission_data = pd.DataFrame({'ID': test_data_ID, 'Prediction': predictions})
submission_data.to_csv('predictions/submission_XGBoost.csv', index=False)
