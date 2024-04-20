import pandas as pd
import redis
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Train and evaluate three different ML models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")
    print(f"{model_name} Classification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

    # Plotting feature importance for tree-based models
    if model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importance = model.feature_importances_
        feature_names = X.columns
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance, y=feature_names)
        plt.title(f"{model_name} Feature Importance")
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.show()
