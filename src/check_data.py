import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "heart_disease.csv"

df = pd.read_csv(file_path)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nTarget Values:")
print(df["num"].value_counts())