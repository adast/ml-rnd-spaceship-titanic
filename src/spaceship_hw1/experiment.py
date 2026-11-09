"""Эксперименты на фиксированных фолдах и парный групповой бутстрэп."""

import datetime as dt
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits

from .data import freeze_splits, read_config, read_training, sha256
from .features import NUMERIC, RAW_FEATURES, make_model


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def score_predictions(y, probability, threshold=0.5):
    y = np.asarray(y)
    probability = np.asarray(probability)
    if (
        not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("Invalid predicted probabilities")
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(
            log_loss(
                y,
                np.column_stack([1 - probability, probability]),
                labels=[0, 1],
            )
        ),
        "brier": float(brier_score_loss(y, probability)),
    }


def audit_development(frame, assignment, manifest):
    dev_mask = assignment["role"].eq("development")
    dev = frame.loc[dev_mask]
    return {
        "split": manifest,
        "target_counts_development": {
            str(k): int(v)
            for k, v in dev["Transported"].value_counts().sort_index().items()
        },
        "missing_development": {
            name: int(dev[name].isna().sum()) for name in RAW_FEATURES
        },
        "numeric_development": dev[NUMERIC]
        .describe()
        .round(4)
        .astype(object)
        .where(lambda values: values.notna(), None)
        .to_dict(),
        "duplicate_predictor_records_development": int(
            dev.duplicated(subset=RAW_FEATURES, keep=False).sum()
        ),
        "group_size_counts_development": {
            str(k): int(v)
            for k, v in assignment.loc[dev_mask]
            .groupby("group")
            .size()
            .value_counts()
            .sort_index()
            .items()
        },
        "holdout_labels_used_in_fit_or_evaluation": False,
        "audit_scope": (
            "Only development labels and features are analyzed. "
            "Whole-data counts and hashes record provenance, "
            "not holdout target analysis."
        ),
    }


def paired_group_intervals(predictions, config):
    primary = predictions[
        (predictions["seed"] == config["split"]["primary_seed"])
        & predictions["protocol"].eq("grouped")
    ]
    wide = primary.pivot(
        index=["PassengerId", "group", "y_true"],
        columns="model",
        values="y_pred",
    ).reset_index()
    groups, codes = np.unique(wide["group"].to_numpy(), return_inverse=True)
    counts = np.bincount(codes).astype(float)
    correct = {
        name: np.bincount(
            codes,
            weights=(wide[name] == wide["y_true"]).to_numpy().astype(float),
            minlength=len(groups),
        )
        for name in config["models"]
    }
    rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
    repetitions = config["evaluation"]["bootstrap_repetitions"]
    sampled = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    denominators = counts[sampled].sum(axis=1)
    bootstrap = {
        name: values[sampled].sum(axis=1) / denominators
        for name, values in correct.items()
    }
    comparisons = [
        ("baseline_vs_dummy", "logistic_cabin", "dummy_prior"),
        ("cabin_features", "logistic_cabin", "logistic_core"),
        ("missing_indicators", "logistic_missing", "logistic_cabin"),
        ("log_spending", "logistic_log_spend", "logistic_missing"),
    ]
    result = {
        "method": (
            "Percentile resampling of development travel groups, "
            "paired for model differences"
        ),
        "repetitions": repetitions,
        "seed": config["evaluation"]["bootstrap_seed"],
        "limitations": (
            "Conditional on fitted OOF models. Does not retrain models "
            "or account fully for shared training sets across folds. "
            "These are development intervals, not final holdout evidence."
        ),
        "accuracy": {},
        "contrasts": {},
    }
    for name, values in correct.items():
        result["accuracy"][name] = {
            "estimate": float(values.sum() / counts.sum()),
            "low": float(np.quantile(bootstrap[name], 0.025)),
            "high": float(np.quantile(bootstrap[name], 0.975)),
        }
    for label, candidate, reference in comparisons:
        difference = bootstrap[candidate] - bootstrap[reference]
        result["contrasts"][label] = {
            "candidate": candidate,
            "reference": reference,
            "estimate": float(
                (correct[candidate] - correct[reference]).sum() / counts.sum()
            ),
            "low": float(np.quantile(difference, 0.025)),
            "high": float(np.quantile(difference, 0.975)),
        }
    return result


def summarize_results(metrics, primary, config):
    """Свести готовые метрики и OOF-прогнозы, проверить правило H1."""
    threshold = config["logistic"]["threshold"]
    primary_metrics = metrics[
        (metrics["protocol"] == "grouped")
        & (metrics["seed"] == config["split"]["primary_seed"])
    ]
    paired = primary_metrics.pivot(
        index="fold", columns="model", values="accuracy"
    )
    differences = paired[config["primary_model"]] - paired["dummy_prior"]
    hypothesis = {
        "criterion": (
            "Positive mean paired fold accuracy difference "
            "and at least 4/5 improved primary folds"
        ),
        "mean_accuracy_difference": float(differences.mean()),
        "improved_folds": int((differences > 0).sum()),
        "supported_by_operational_rule": bool(
            differences.mean() > 0
            and (differences > 0).sum()
            >= config["evaluation"]["hypothesis_min_improved_folds"]
        ),
        "is_statistical_significance_test": False,
    }
    summary = []
    for name in config["models"]:
        subset = primary_metrics[primary_metrics["model"].eq(name)]
        oof = primary[primary["model"].eq(name)]
        summary.append(
            {
                "model": name,
                "fold_accuracy_mean": float(subset["accuracy"].mean()),
                "fold_accuracy_sd": float(subset["accuracy"].std(ddof=1)),
                **score_predictions(
                    oof["y_true"], oof["probability"], threshold
                ),
            }
        )
    return {
        "primary_seed": config["split"]["primary_seed"],
        "primary_model": config["primary_model"],
        "models": summary,
        "hypothesis": hypothesis,
        "holdout_evaluated": False,
    }


def run_experiment(root, config_path, output=None):
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output).resolve() if output else root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    config = read_config(config_path)
    started = time.perf_counter()
    provenance_path = root / "data/PROVENANCE.json"
    if provenance_path.exists():
        expected_hash = json.loads(provenance_path.read_text())["files"][
            "train.csv"
        ]["sha256"]
        if sha256(root / config["data"]["train"]) != expected_hash:
            raise ValueError(
                "Training data does not match recorded provenance"
            )
    frame = read_training(
        root / config["data"]["train"], config["data"]["expected_rows"]
    )
    assignment, manifest = freeze_splits(frame, config, root)
    write_json(
        output / "data_audit.json",
        audit_development(frame, assignment, manifest),
    )
    dev_mask = assignment["role"].eq("development")
    dev = frame.loc[dev_mask].reset_index(drop=True)
    split = assignment.loc[dev_mask].reset_index(drop=True)
    X = dev.drop(columns="Transported")
    y = dev["Transported"].to_numpy()
    threshold = config["logistic"]["threshold"]
    metrics, predictions, run_log = [], [], []
    jobs = [
        ("grouped", seed, name)
        for seed in config["split"]["seeds"]
        for name in config["models"]
    ]
    if config["evaluation"]["random_split_diagnostic"]:
        jobs.append(
            (
                "random_passenger",
                config["split"]["primary_seed"],
                config["primary_model"],
            )
        )
    for protocol, seed, name in jobs:
        fold_column = (
            f"fold_{seed}" if protocol == "grouped" else "random_fold"
        )
        for fold in range(config["split"]["n_splits"]):
            val = split[fold_column].eq(fold).to_numpy()
            train = ~val
            if protocol == "grouped" and set(split.loc[train, "group"]) & set(
                split.loc[val, "group"]
            ):
                raise AssertionError("Group overlap before model fit")
            model = make_model(name, config)
            timer = time.perf_counter()
            with threadpool_limits(limits=config["runtime"]["blas_threads"]):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(X.loc[train], y[train])
                probability = model.predict_proba(X.loc[val])[:, 1]
            elapsed = time.perf_counter() - timer
            result = score_predictions(y[val], probability, threshold)
            metrics.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "model": name,
                    "fold": fold,
                    "n_train": int(train.sum()),
                    "n_validation": int(val.sum()),
                    **result,
                }
            )
            table = pd.DataFrame(
                {
                    "PassengerId": dev.loc[val, "PassengerId"].to_numpy(),
                    "group": split.loc[val, "group"].to_numpy(),
                    "protocol": protocol,
                    "seed": seed,
                    "fold": fold,
                    "model": name,
                    "y_true": y[val],
                    "probability": probability,
                    "y_pred": (probability >= threshold).astype(int),
                }
            )
            predictions.append(table)
            classifier = model.named_steps["classifier"]
            run_log.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "fold": fold,
                    "model": name,
                    "fit_and_predict_seconds": elapsed,
                    "iterations": [
                        int(x)
                        for x in np.atleast_1d(
                            getattr(classifier, "n_iter_", [])
                        )
                    ],
                }
            )
        print(f"Завершено: {protocol}, seed={seed}, модель={name}", flush=True)
    metrics = pd.DataFrame(metrics)
    predictions = pd.concat(predictions, ignore_index=True)
    if set(predictions["PassengerId"]) & set(
        assignment.loc[~dev_mask, "PassengerId"]
    ):
        raise AssertionError("Holdout present in evaluation predictions")
    grouped = predictions[predictions["protocol"].eq("grouped")]
    expected = len(dev) * len(config["split"]["seeds"]) * len(config["models"])
    if (
        len(grouped) != expected
        or grouped.duplicated(["PassengerId", "seed", "model"]).any()
    ):
        raise AssertionError(
            "Every development passenger must receive exactly one "
            "OOF prediction per model/seed"
        )
    metrics.to_csv(
        output / "fold_metrics.csv", index=False, float_format="%.17g"
    )
    predictions.to_csv(
        output / "predictions.csv", index=False, float_format="%.17g"
    )
    primary = predictions[
        (predictions["protocol"] == "grouped")
        & (predictions["seed"] == config["split"]["primary_seed"])
    ]
    main = primary[primary["model"].eq(config["primary_model"])].copy()
    main.to_csv(output / "oof_primary.csv", index=False, float_format="%.17g")
    intervals = paired_group_intervals(predictions, config)
    write_json(output / "group_bootstrap.json", intervals)
    write_json(output / "summary.json",
               summarize_results(metrics, primary, config))
    with threadpool_limits(limits=config["runtime"]["blas_threads"]):
        fitted = make_model(config["primary_model"], config).fit(X, y)
    artifact_dir = (
        root / "artifacts" if output == root / "reports" else output / "models"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, artifact_dir / "baseline.joblib")
    write_json(
        artifact_dir / "baseline_metadata.json",
        {
            "model": config["primary_model"],
            "training_scope": "development only",
            "data_sha256": manifest["data_sha256"],
            "config_sha256": sha256(config_path),
            "threshold": threshold,
            "trained_rows": len(dev),
            "holdout_evaluated": False,
        },
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "numpy", "pandas", "scipy", "scikit-learn",
            "matplotlib", "PyYAML", "joblib", "threadpoolctl",
        ]
    }
    write_json(
        output / "run_metadata.json",
        {
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": versions,
            "config": config,
            "config_sha256": sha256(config_path),
            "data_sha256": manifest["data_sha256"],
            "split_manifest_sha256": sha256(
                root / "data/splits/manifest.json"
            ),
            "source_files_sha256": {
                str(p.relative_to(root)): sha256(p)
                for p in sorted((root / "src").rglob("*.py"))
            },
            "total_seconds": time.perf_counter() - started,
            "runs": run_log,
            "final_holdout_evaluated": False,
        },
    )
    return config, frame, assignment, metrics, predictions


def audit_saved_metrics(folder, threshold=0.5):
    """Независимо пересчитать метрики фолдов по сохраненным прогнозам."""
    folder = Path(folder)
    predictions = pd.read_csv(
        folder / "predictions.csv",
        dtype={"PassengerId": "string", "group": "string"},
    )
    saved = pd.read_csv(folder / "fold_metrics.csv")
    keys = ["protocol", "seed", "model", "fold"]
    recomputed = []
    for values, subset in predictions.groupby(keys):
        recomputed.append(
            {
                **dict(zip(keys, values)),
                **score_predictions(
                    subset["y_true"], subset["probability"], threshold
                ),
            }
        )
    reconstructed = pd.DataFrame(recomputed).set_index(keys).sort_index()
    recorded = saved.set_index(keys).sort_index()[reconstructed.columns]
    if not reconstructed.index.equals(recorded.index):
        raise AssertionError(
            "Saved folds do not match prediction-ledger folds"
        )
    np.testing.assert_allclose(
        recorded.to_numpy(), reconstructed.to_numpy(), rtol=0, atol=1e-10
    )
    return {
        "passed": True,
        "folds_recomputed": len(reconstructed),
        "absolute_tolerance": 1e-10,
    }
