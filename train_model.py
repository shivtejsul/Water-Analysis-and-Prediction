import pandas as pd
import joblib
from prophet import Prophet
import os

# Load dataset
df = pd.read_csv("D:\Water Anaysis and Predction\Dataset\Krishna_River_Dataset.csv")  # replace with your file

# Ensure date column is datetime
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")


# Create models directory
os.makedirs("models", exist_ok=True)

# Parameters to forecast (example, adjust to your dataset)
parameters = ["DO", "BOD", "pH", "Conductivity", "Nitrate"]

for param in parameters:
    if param not in df.columns:
        continue  # skip if column missing

    # Prepare data for Prophet
    temp = df[["date", param]].dropna()
    temp = temp.rename(columns={"date": "ds", param: "y"})

    # Train Prophet model
    model = Prophet()
    model.fit(temp)

    # Save trained model
    joblib.dump(model, f"models/prophet_{param}.pkl")

    print(f"✅ Trained and saved Prophet model for {param}")
