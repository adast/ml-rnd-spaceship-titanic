"""Сравнить M и MC при переносе на целый диапазон номеров групп."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from spaceship_hw1.data import read_config
from spaceship_hw1.experiment import score_predictions
from spaceship_hw2 import load_development, make_hw2_model


def evaluate_blocks(dev, split, baseline, boosting):
    group_id = split["group"].astype(int)
    bands = np.array_split(np.sort(group_id.unique()), 5)
    features = dev.drop(columns="Transported")
    target = dev["Transported"]
    threshold = baseline["logistic"]["threshold"]
    metrics, predictions = [], []
    for block, groups in enumerate(bands, start=1):
        valid = group_id.isin(groups)
        train = ~valid
        if set(group_id[valid]) & set(group_id[train]):
            raise ValueError("Диапазон пересекается с обучающими группами")
        rows = split.loc[valid, ["PassengerId", "group"]].copy()
        rows["block"] = block
        rows["y_true"] = target[valid]
        row = {
            "block": block, "min_group_id": groups.min(),
            "max_group_id": groups.max(), "n_groups": len(groups),
            "n_train": int(train.sum()), "n_validation": len(rows),
            "positive_rate": rows["y_true"].mean(),
        }
        correct = {}
        for name in ("M", "MC"):
            model = make_hw2_model(name, baseline, boosting)
            model.fit(features.loc[train], target[train])
            probability = model.predict_proba(features.loc[valid])[:, 1]
            rows[f"p_{name}"] = probability
            scores = score_predictions(rows["y_true"], probability, threshold)
            row[f"accuracy_{name}"] = scores["accuracy"]
            row[f"log_loss_{name}"] = scores["log_loss"]
            correct[name] = rows["y_true"].eq(probability >= threshold)

        fixed = int((correct["MC"] & ~correct["M"]).sum())
        introduced = int((correct["M"] & ~correct["MC"]).sum())
        metrics.append({
            **row, "errors_fixed": fixed, "errors_introduced": introduced,
            "net_correct": fixed - introduced,
            "delta_pp": 100 * (fixed - introduced) / len(rows),
        })
        predictions.append(rows)
    oof = pd.concat(predictions, ignore_index=True)
    if (not oof["PassengerId"].is_unique
            or set(oof["PassengerId"]) != set(dev["PassengerId"])):
        raise ValueError("Нарушено покрытие пассажиров рабочей выборки")
    return pd.DataFrame(metrics), oof


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "reports/hw2")
    folder = parser.parse_args().output
    baseline = read_config(root / "configs/baseline.yaml")
    boosting = yaml.safe_load((root / "configs/hw2.yaml").read_text())[
        "boosting"
    ]
    dev, split, _ = load_development(root, baseline)
    with threadpool_limits(limits=1):
        metrics, oof = evaluate_blocks(dev, split, baseline, boosting)
    folder.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(folder / "cabin_blocks.csv", index=False,
                   float_format="%.12g")
    oof.to_csv(
        folder / "cabin_block_predictions.csv.gz", index=False,
        float_format="%.17g", compression={"method": "gzip", "mtime": 0},
    )
