from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pandas as pd

# Load train data
data = pd.read_csv('data/train_final.csv')

# Divide categorical and numerical columns
categorical_cols = ['workclass', 'education','marital.status','occupation',
                    'relationship','race','sex', 'native.country']
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain',
                  'capital.loss', 'hours.per.week']

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

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Depth and width research
depths = [3, 5, 9]
widths = [5, 10, 25, 50]

# Loop through configurations
results = []
for depth in depths:
    for width in widths:
        # Define hidden layer sizes
        hidden_layer_sizes = tuple([width] * depth)

        # Create the MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                            activation='relu',  # Use ReLU activation
                            solver='adam',  # Adam optimizer
                            learning_rate_init=1e-3,  # Initial learning rate
                            max_iter=200,  # Maximum iterations
                            random_state=42)

        # Train the model
        mlp.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = mlp.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((depth, width, acc))
        print(f"Depth: {depth}, Width: {width}, Accuracy: {acc:.4f}")



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

# Use the best configuration for predictions
best_config = max(results, key=lambda x: x[2])  # Get the configuration with max accuracy
best_depth, best_width, _ = best_config
best_hidden_layer_sizes = tuple([best_width] * best_depth)

# Retrain on the best configuration
best_mlp = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes,activation='relu',solver='adam',learning_rate_init=1e-3,max_iter=300,random_state=42)

best_mlp.fit(X, y)  # Retrain on entire training set
predictions = best_mlp.predict(X_test)

# Write the results in a CSV
submission_data = pd.DataFrame({'ID': test_data_ID, 'Prediction': predictions})
submission_data.to_csv('predictions/submission_Amanda_SS_mlp_test1.csv', index=False)
