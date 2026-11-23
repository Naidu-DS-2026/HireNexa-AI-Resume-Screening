
from datasets import load_dataset
import pandas as pd
import os

# Create dataset directory
os.makedirs("dataset/raw", exist_ok=True)

# Load dataset
dataset = load_dataset(
    "med2425/resume-job-fit-merged-v1"
)

# Display available splits
print("Available splits:", dataset.keys())

# Load training data
df = dataset["train"].to_pandas()

# Save raw data
df.to_csv(
    "dataset/raw/resume_job_fit.csv",
    index=False
)

print("Dataset downloaded successfully!")

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())
