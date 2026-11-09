"""Проверка входа и фиксированное разбиение без пересечения групп."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .features import RAW_FEATURES


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_config(path):
    with Path(path).open() as f:
        config = yaml.safe_load(f)
    if config["primary_model"] not in config["models"]:
        raise ValueError("Primary model is absent from experiment list")
    if config["split"]["primary_seed"] not in config["split"]["seeds"]:
        raise ValueError("Primary split seed is absent from experiment seeds")
    return config


def read_training(path, expected_rows=None):
    frame = pd.read_csv(path, dtype={"PassengerId": "string"})
    required = {"PassengerId", "Transported", *RAW_FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Input is missing fields {sorted(required - set(frame.columns))}"
        )
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} training rows, got {len(frame)}"
        )
    if frame["PassengerId"].isna().any() or not frame["PassengerId"].is_unique:
        raise ValueError("PassengerId must be present and unique")
    if not frame["PassengerId"].str.fullmatch(r"\d{4}_\d{2}").all():
        raise ValueError("Unexpected PassengerId format")
    target = (
        frame["Transported"]
        .astype("string")
        .str.lower()
        .map({"false": 0, "true": 1})
    )
    if target.isna().any():
        raise ValueError("Target must be complete Boolean True/False")
    frame["Transported"] = target.astype(int)
    return frame.sort_values("PassengerId").reset_index(drop=True)


def group_ids(frame):
    return frame["PassengerId"].astype(str).str.split("_").str[0]


def holdout_group(group, salt, fraction):
    value = int.from_bytes(
        hashlib.sha256(f"{salt}:{group}".encode()).digest()[:8], "big"
    )
    return value / 2**64 < fraction


def build_assignments(frame, config):
    settings = config["split"]
    groups = group_ids(frame)
    held = groups.map(
        lambda g: holdout_group(
            g, settings["holdout_salt"], settings["holdout_fraction"]
        )
    )
    assignment = pd.DataFrame(
        {
            "PassengerId": frame["PassengerId"],
            "group": groups,
            "role": np.where(held, "final_holdout", "development"),
        }
    )
    dev = np.flatnonzero(~held.to_numpy())
    y = frame.loc[dev, "Transported"].to_numpy()
    g = groups.loc[dev].to_numpy()
    if (
        len(np.unique(g)) < settings["n_splits"]
        or held.all()
        or not held.any()
    ):
        raise ValueError("Not enough groups for the frozen holdout/CV design")
    for seed in settings["seeds"]:
        assignment[f"fold_{seed}"] = -1
        splitter = StratifiedGroupKFold(
            n_splits=settings["n_splits"], shuffle=True, random_state=seed
        )
        for fold, (train, val) in enumerate(
            splitter.split(np.zeros(len(dev)), y, g)
        ):
            if set(g[train]) & set(g[val]):
                raise AssertionError("Group leakage inside a fold")
            assignment.loc[dev[val], f"fold_{seed}"] = fold
    assignment["random_fold"] = -1
    splitter = StratifiedKFold(
        n_splits=settings["n_splits"],
        shuffle=True,
        random_state=settings["primary_seed"],
    )
    for fold, (_, val) in enumerate(splitter.split(np.zeros(len(dev)), y)):
        assignment.loc[dev[val], "random_fold"] = fold
    return assignment


def freeze_splits(frame, config, root):
    folder = root / "data/splits"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path, assignment_path = (
        folder / "manifest.json",
        folder / "assignments.csv",
    )
    fingerprint = hashlib.sha256(
        json.dumps(config["split"], sort_keys=True).encode()
    ).hexdigest()
    data_hash = sha256(root / config["data"]["train"])
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if (
            saved["data_sha256"] != data_hash
            or saved["split_config_sha256"] != fingerprint
        ):
            raise ValueError(
                "Frozen data/split changed: "
                "do not overwrite a used evaluation protocol"
            )
        if sha256(assignment_path) != saved["assignments_sha256"]:
            raise ValueError("Frozen assignments were modified")
        assignment = pd.read_csv(
            assignment_path, dtype={"PassengerId": "string", "group": "string"}
        )
        if assignment["PassengerId"].tolist() != frame["PassengerId"].tolist():
            raise ValueError("Frozen PassengerId order does not match dataset")
    else:
        assignment = build_assignments(frame, config)
        assignment.to_csv(assignment_path, index=False)
        dev = assignment["role"].eq("development")
        saved = {
            "data_sha256": data_hash,
            "split_config_sha256": fingerprint,
            "assignments_sha256": sha256(assignment_path),
            "total_rows": len(frame),
            "development_rows": int(dev.sum()),
            "final_holdout_rows": int((~dev).sum()),
            "development_groups": int(assignment.loc[dev, "group"].nunique()),
            "holdout_groups": int(assignment.loc[~dev, "group"].nunique()),
            "holdout_evaluated": False,
        }
        manifest_path.write_text(json.dumps(saved, indent=2) + "\n")
    dev = assignment["role"].eq("development")
    if set(assignment.loc[dev, "group"]) & set(assignment.loc[~dev, "group"]):
        raise AssertionError("Holdout and development share groups")
    if (
        (
            assignment.loc[
                ~dev, [f"fold_{s}" for s in config["split"]["seeds"]]
            ]
            != -1
        )
        .any()
        .any()
    ):
        raise AssertionError("Holdout assigned to development folds")
    return assignment, saved
