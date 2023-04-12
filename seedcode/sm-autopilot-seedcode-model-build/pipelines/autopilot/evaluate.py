import json
import os
import pathlib
import pandas as pd
from sklearn.metrics import f1_score


if __name__ == "__main__":
    y_pred_path = "/opt/ml/processing/input/predictions/x_test.csv.out"
    y_pred = pd.read_csv(y_pred_path, header=None, skipinitialspace=True)
    y_true_path = "/opt/ml/processing/input/true_labels/y_test.csv"
    y_true = pd.read_csv(y_true_path, header=None, skipinitialspace=True)
    report_dict = {
        "classification_metrics": {
            "weighted_f1": {
                "value": f1_score(y_true, y_pred, average="weighted"),
                "standard_deviation": "NaN",
            },
        },
    }
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
