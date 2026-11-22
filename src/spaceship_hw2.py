"""Factorial HW2 experiment on the frozen HW1 development folds."""

import datetime as dt
import json
import platform
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from spaceship_hw1.data import (
    freeze_splits,
    read_config,
    read_training,
    sha256,
)
from spaceship_hw1.experiment import score_predictions, write_json
from spaceship_hw1.features import PassengerFeatures, make_model

MODELS = ("B1", "C", "M", "MC")


class CabinNumberFeatures(PassengerFeatures):
    """Add the middle numeric field of a cabin accepted by the HW1 parser."""

    def transform(self, X):
        result = super().transform(X)
        number = X["Cabin"].astype("string").str.extract(
            r"/([0-9]+)/", expand=False
        )
        result["CabinNumber"] = number.where(
            result["CabinDeck"].notna()
        ).astype(float)
        return result


def make_hw2_model(name, baseline, boosting):
    if name not in MODELS:
        raise ValueError(f"Unknown HW2 model {name}")
    model = make_model("logistic_cabin", baseline)
    if name in ("C", "MC"):
        model.set_params(features=CabinNumberFeatures())
        preprocess = model.named_steps["preprocess"]
        preprocess.transformers = [
            (key, step, [*columns, "CabinNumber"] if key == "numeric"
             else columns)
            for key, step, columns in preprocess.transformers
        ]
    if name in ("M", "MC"):
        model.set_params(classifier=HistGradientBoostingClassifier(**boosting))
    return model


def load_development(root, baseline):
    """Require the HW1 inputs and return only development rows and folds."""
    manifest_file = root / "data/splits/manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Frozen HW1 manifest required: {manifest_file}"
        )
    provenance = json.loads((root / "data/PROVENANCE.json").read_text())
    train_path = root / baseline["data"]["train"]
    if sha256(train_path) != provenance["files"]["train.csv"]["sha256"]:
        raise ValueError("Training CSV differs from HW1 provenance")
    records = read_training(train_path, baseline["data"]["expected_rows"])
    assignments, manifest = freeze_splits(records, baseline, root)
    keep = assignments["role"].eq("development")
    return (
        records.loc[keep].reset_index(drop=True),
        assignments.loc[keep].reset_index(drop=True),
        manifest,
    )


def compare_historical_b1(predictions, reference_path):
    previous = pd.read_csv(
        reference_path, dtype={"PassengerId": "string", "group": "string"},
        converters={"probability": float},
    )
    previous = previous.loc[
        previous["protocol"].eq("grouped")
        & previous["model"].eq("logistic_cabin")
    ]
    latest = predictions.loc[predictions["model"].eq("B1")]
    keys = ["seed", "PassengerId"]
    previous = previous.set_index(keys).sort_index()
    latest = latest.set_index(keys).sort_index()
    if not latest.index.is_unique or not previous.index.is_unique:
        raise AssertionError("Duplicate B1 passenger/seed keys")
    if not latest.index.equals(previous.index):
        raise AssertionError("HW2 and HW1 B1 passenger/seed keys differ")
    for column in ("group", "fold", "y_true", "y_pred"):
        if not np.array_equal(latest[column], previous[column]):
            raise AssertionError(f"HW1 B1 mismatch in {column}")
    difference = latest["probability"] - previous["probability"]
    tolerance = 1e-9
    np.testing.assert_allclose(
        latest["probability"], previous["probability"], rtol=0, atol=tolerance
    )
    return {
        "passed": True, "prediction_rows": len(latest),
        "maximum_probability_difference": float(difference.abs().max()),
        "absolute_tolerance": tolerance,
    }


def fit_folds(dev, split, baseline, boosting):
    features = dev.drop(columns="Transported")
    target = dev["Transported"].to_numpy()
    threshold = baseline["logistic"]["threshold"]
    seeds = baseline["split"]["seeds"]
    n_folds = baseline["split"]["n_splits"]
    fold_metrics, predictions, timings = [], [], []
    for seed in seeds:
        column = f"fold_{seed}"
        if set(split[column]) != set(range(n_folds)):
            raise ValueError(f"Missing or unexpected folds for seed {seed}")
        if split.groupby("group")[column].nunique().ne(1).any():
            raise ValueError(f"Group overlap across folds for seed {seed}")
        for fold in range(n_folds):
            val = split[column].eq(fold).to_numpy()
            train = ~val
            for name in MODELS:
                model = make_hw2_model(name, baseline, boosting)
                started = time.perf_counter()
                model.fit(features.loc[train], target[train])
                fitted = time.perf_counter()
                probability = model.predict_proba(features.loc[val])[:, 1]
                finished = time.perf_counter()
                run = {"seed": seed, "fold": fold, "model": name}
                timings.append({
                    **run, "fit_seconds": fitted - started,
                    "predict_seconds": finished - fitted,
                })
                fold_metrics.append({
                    **run, "n_train": int(train.sum()),
                    "n_validation": int(val.sum()),
                    **score_predictions(target[val], probability, threshold),
                })
                rows = split.loc[val, ["PassengerId", "group"]].copy()
                predictions.append(rows.assign(
                    seed=seed, fold=fold, y_true=target[val], model=name,
                    probability=probability,
                    y_pred=(probability >= threshold).astype(int),
                ))
        print(f"Completed grouped validation for seed {seed}", flush=True)
    oof = pd.concat(predictions, ignore_index=True)
    if (len(oof) != len(dev) * len(seeds) * len(MODELS)
            or oof.duplicated(["PassengerId", "seed", "model"]).any()):
        raise AssertionError("OOF coverage differs from the experiment design")
    return pd.DataFrame(fold_metrics), oof, timings


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    baseline_path = root / "configs/baseline.yaml"
    config_path = root / "configs/hw2.yaml"
    reference_path = root / "reports/predictions.csv"
    baseline = read_config(baseline_path)
    config = yaml.safe_load(config_path.read_text())
    boosting = config["boosting"]
    started_utc = dt.datetime.now(dt.UTC)
    dev, split, manifest = load_development(root, baseline)
    threads = baseline["runtime"]["blas_threads"]
    with threadpool_limits(limits=threads), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        folds, predictions, timings = fit_folds(dev, split, baseline, boosting)
    parity = compare_historical_b1(predictions, reference_path)
    inputs = (
        "configs/baseline.yaml", "configs/hw2.yaml", "docs/HW2_PLAN.md",
        "data/PROVENANCE.json",
        "data/splits/manifest.json", "data/splits/assignments.csv",
        "reports/predictions.csv", "src/spaceship_hw1/data.py",
        "src/spaceship_hw1/features.py", "src/spaceship_hw1/experiment.py",
        "src/spaceship_hw2.py", "scripts/run_hw2.py",
    )
    metadata = {
        "started_at_utc": started_utc.isoformat(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "versions": {
            package: version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn",
                            "PyYAML", "threadpoolctl")
        },
        "source_and_input_sha256": {p: sha256(root / p) for p in inputs},
        "frozen_split": manifest, "final_holdout_evaluated": False,
        "model_ids": list(MODELS), "config": config,
        "logistic": baseline["logistic"], "split": baseline["split"],
        "threads": threads, "threshold": baseline["logistic"]["threshold"],
        "historical_b1_check": parity, "fits": len(folds),
        "prediction_rows": len(predictions), "timings": timings,
    }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as scratch:
        stage = Path(scratch)
        folds.to_csv(stage / "folds.csv", index=False, float_format="%.17g")
        predictions.to_csv(
            stage / "predictions.csv.gz", index=False, float_format="%.17g",
            compression={"method": "gzip", "mtime": 0},
        )
        files = ("folds.csv", "predictions.csv.gz")
        metadata["output_sha256"] = {p: sha256(stage / p) for p in files}
        finished_utc = dt.datetime.now(dt.UTC)
        metadata["finished_at_utc"] = finished_utc.isoformat()
        duration = finished_utc - started_utc
        metadata["elapsed_seconds"] = duration.total_seconds()
        write_json(stage / "metadata.json", metadata)
        for file in (*files, "metadata.json"):
            (stage / file).replace(output / file)
