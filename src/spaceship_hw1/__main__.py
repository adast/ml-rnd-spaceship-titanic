"""Команды для экспериментов, отчетов и прогнозов."""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import freeze_splits, read_config, read_training
from .experiment import audit_saved_metrics, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["run", "report", "predict", "verify"]
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--input", default="data/raw/test.csv")
    parser.add_argument("--model", default="artifacts/baseline.joblib")
    parser.add_argument("--candidate", default="reports/reproduction")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    config_path = root / args.config
    config = read_config(config_path)
    output = root / args.output
    if args.command in ["run", "report"]:
        import matplotlib

        matplotlib.use("Agg")
        from .reporting import generate_report

        if args.command == "run":
            config, frame, assignment, metrics, predictions = run_experiment(
                root, config_path, output
            )
        else:
            frame = read_training(
                root / config["data"]["train"], config["data"]["expected_rows"]
            )
            assignment, _ = freeze_splits(frame, config, root)
            metrics = pd.read_csv(output / "fold_metrics.csv")
            predictions = pd.read_csv(
                output / "predictions.csv",
                dtype={"PassengerId": "string", "group": "string"},
            )
        generate_report(
            config, frame, assignment, metrics, predictions, output
        )
        print(f"Воспроизводимые результаты сохранены в {output}")
    elif args.command == "predict":
        frame = pd.read_csv(root / args.input, dtype={"PassengerId": "string"})
        model = joblib.load(root / args.model)
        probability = model.predict_proba(frame)[:, 1]
        submission = pd.DataFrame(
            {
                "PassengerId": frame["PassengerId"],
                "Transported": probability >= config["logistic"]["threshold"],
            }
        )
        destination = output / "submission.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(destination, index=False)
        print(
            f"Сохранено {len(submission)} прогнозов в {destination}; "
            "метки Kaggle не использовались"
        )
    elif args.command == "verify":
        candidate = root / args.candidate
        metric_audit = audit_saved_metrics(
            output, config["logistic"]["threshold"]
        )
        audit_saved_metrics(candidate, config["logistic"]["threshold"])
        columns = ["protocol", "seed", "model", "fold", "PassengerId"]
        a = (
            pd.read_csv(
                output / "predictions.csv",
                dtype={"PassengerId": "string", "group": "string"},
            )
            .sort_values(columns)
            .reset_index(drop=True)
        )
        b = (
            pd.read_csv(
                candidate / "predictions.csv",
                dtype={"PassengerId": "string", "group": "string"},
            )
            .sort_values(columns)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            a.drop(columns="probability"), b.drop(columns="probability")
        )
        np.testing.assert_allclose(
            a["probability"], b["probability"], rtol=0, atol=1e-9
        )
        original = json.loads((output / "run_metadata.json").read_text())
        reproduced = json.loads((candidate / "run_metadata.json").read_text())
        for name in [
            "data_sha256", "config_sha256", "split_manifest_sha256",
            "source_files_sha256", "versions",
        ]:
            if original[name] != reproduced[name]:
                raise AssertionError(f"Reproduction metadata mismatch: {name}")
        result = {
            "passed": True,
            "prediction_rows": len(a),
            "maximum_probability_difference": float(
                np.max(np.abs(a["probability"] - b["probability"]))
            ),
            "absolute_tolerance": 1e-9,
            "holdout_evaluated": False,
            "independent_metric_audit": metric_audit,
        }
        (output / "reproduction_check.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
