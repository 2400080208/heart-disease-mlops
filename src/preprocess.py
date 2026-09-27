import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project folder
project_folder = Path(__file__).resolve().parent.parent

# File paths
input_file = project_folder / "data" / "raw" / "heart_disease.csv"
output_folder = project_folder / "data" / "processed"

# Create processed folder
output_folder.mkdir(parents=True, exist_ok=True)

# Read dataset
df = pd.read_csv(input_file)

# Convert target into binary classification
df["num"] = (df["num"] > 0).astype(int)

# Fill missing values
df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].median())

# Split data
train_data, test_data = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["num"]
)

# Save processed data
train_data.to_csv(output_folder / "train.csv", index=False)
test_data.to_csv(output_folder / "test.csv", index=False)

print("Preprocessing completed!")
print("Total records:", len(df))
print("Training records:", len(train_data))
print("Testing records:", len(test_data))
print("\nTarget distribution:")
print(df["num"].value_counts())
print("\nMissing values after preprocessing:")
print(df.isnull().sum())