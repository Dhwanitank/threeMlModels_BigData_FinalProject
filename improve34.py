import pandas as pd
import redis
import json
import yaml
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load Redis configuration from YAML file
with open("/Users/dhwanitank/Desktop/fraud_BigData/config.yaml", 'r') as stream:
    redis_config = yaml.safe_load(stream)["redis"]

# Connect to Redis
r = redis.Redis(
    host=redis_config["host"],
    port=redis_config["port"],
    password=redis_config.get("password")  # Check if password exists in config
)

# Retrieve data from Redis and convert to DataFrame
transaction_data = []
for key in r.keys('transaction_*'):
    transaction_json = r.get(key)
    transaction_data.append(json.loads(transaction_json))

df = pd.DataFrame(transaction_data)

# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Define features and target variable
X = df[['BMI', 'Smoking', 'AlcoholDrinking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity']]
y = df['HeartDisease']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['Smoking', 'AlcoholDrinking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling Data Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 4]
}

# Initialize Gradient Boosting Classifier
gradient_boosting = GradientBoostingClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(gradient_boosting, param_grid, cv=5, scoring='f1_macro')

# Fit GridSearchCV
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model
best_gradient_boosting = grid_search.best_estimator_

# Make predictions
y_pred = best_gradient_boosting.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Best Gradient Boosting Parameters: {grid_search.best_params_}")
print(f"Gradient Boosting Accuracy after tuning: {accuracy}")
print(f"Gradient Boosting Classification Report after tuning:\n{report}")
