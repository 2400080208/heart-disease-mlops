from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# Download dataset
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)

# Get project folder
project_folder = Path(__file__).resolve().parent.parent

# Create data/raw folder
data_folder = project_folder / "data" / "raw"
data_folder.mkdir(parents=True, exist_ok=True)

# Save dataset
file_path = data_folder / "heart_disease.csv"

df.to_csv(file_path, index=False)

print("Dataset downloaded successfully!")
print("Shape:", df.shape)
print("Saved at:", file_path)