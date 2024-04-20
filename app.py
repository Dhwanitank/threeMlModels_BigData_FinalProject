import pandas as pd
import redis
import json
import yaml

# Load Redis configuration from YAML file
with open("/Users/dhwanitank/Desktop/fraud_BigData/config.yaml", 'r') as stream:
    redis_config = yaml.safe_load(stream)["redis"]

# Connect to Redis
r = redis.Redis(
    host=redis_config["host"],
    port=redis_config["port"],
    password=redis_config.get("password")  # Check if password exists in config
)

# Load the dataset from CSV
df = pd.read_csv("/Users/dhwanitank/Downloads/updated_heart_2020_cleaned.csv")

# Loop through each row and store it in Redis
for index, row in df.iterrows():
    transaction_id = f"transaction_{index}"  # Assuming index as the transaction ID
    transaction_data = {
        'HeartDisease': row['HeartDisease'],
        'BMI': row['BMI'],
        'Smoking': row['Smoking'],
        'AlcoholDrinking': row['AlcoholDrinking'],
        'Sex': row['Sex'],
        'AgeCategory': row['AgeCategory'],
        'Diabetic': row['Diabetic'],
        'PhysicalActivity': row['PhysicalActivity']
    }
    r.set(transaction_id, json.dumps(transaction_data))

print("Dataset loaded into Redis successfully!")
