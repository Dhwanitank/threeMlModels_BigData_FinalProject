import pandas as pd
import redis
import json
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV  # Import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

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

# Handling Data Imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model Tuning
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Adjust the values as needed
log_reg = LogisticRegression()
grid_search = GridSearchCV(log_reg, parameters, cv=5)
grid_search.fit(X_train_resampled, y_train_resampled)

best_log_reg = grid_search.best_estimator_

# Train and evaluate the best logistic regression model
best_log_reg.fit(X_train_resampled, y_train_resampled)
y_pred = best_log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Best Logistic Regression Accuracy: {accuracy}")
print(f"Best Logistic Regression Classification Report:\n{report}")
