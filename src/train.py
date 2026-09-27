import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

project_folder = Path(__file__).resolve().parent.parent

train_file = project_folder / "data" / "processed" / "train.csv"
test_file = project_folder / "data" / "processed" / "test.csv"
model_folder = project_folder / "models"

model_folder.mkdir(parents=True, exist_ok=True)

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data.drop("num", axis=1)
y_train = train_data["num"]

X_test = test_data.drop("num", axis=1)
y_test = test_data["num"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

mlflow.set_experiment("Heart_Disease_Prediction")

with mlflow.start_run():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model,
        name="heart_disease_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("Model training completed!")
    print("Accuracy:", accuracy)

    model_file = model_folder / "heart_disease_model.joblib"
    joblib.dump(model, model_file)

    print("Model saved at:", model_file)
    print("MLflow tracking completed!")