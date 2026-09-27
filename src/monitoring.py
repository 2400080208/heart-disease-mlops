import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

project_folder = Path(__file__).resolve().parent.parent

reference_file = project_folder / "data" / "processed" / "train.csv"
current_file = project_folder / "data" / "processed" / "test.csv"

reference_data = pd.read_csv(reference_file)
current_data = pd.read_csv(current_file)

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

output_file = project_folder / "reports" / "data_drift_report.html"
output_file.parent.mkdir(parents=True, exist_ok=True)

result.save_html(str(output_file))

print("Evidently monitoring completed!")
print("Report saved at:", output_file)
print("File exists:", output_file.exists())