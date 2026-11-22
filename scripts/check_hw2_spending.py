"""Проверить сумму и состав расходов на тех же фолдах, что и модель MC."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss
from summarize_hw2 import bootstrap_scores
from threadpoolctl import threadpool_limits

from spaceship_hw1.data import read_config
from spaceship_hw2 import load_development, spending_model

VARIANTS = {
    "total": ["TotalSpend"],
    "share": ["SpendShare"],
    "total_share": ["TotalSpend", "SpendShare"],
}


def comparisons(folder, split, seeds):
    """Сопоставить сохранённые прогнозы MC с разбиениями рабочей выборки."""
    options = {"dtype": {"PassengerId": str, "group": str}}
    grouped = pd.read_csv(
        folder / "predictions.csv.gz", **options,
        converters={"probability": float},
    ).query("model == 'MC'").rename(columns={"probability": "MC"})
    for seed in seeds:
        yield (f"group_{seed}", split[f"fold_{seed}"],
               grouped.loc[grouped.seed.eq(seed)])
    archived = pd.read_csv(
        folder / "cabin_block_predictions.csv.gz", **options,
        converters={"p_MC": float},
    ).rename(columns={"block": "fold", "p_MC": "MC"})
    groups = split.group.astype(int)
    blocks = np.array_split(np.sort(groups.unique()), 5)
    fold = np.searchsorted([block[-1] for block in blocks], groups) + 1
    yield "blocks", pd.Series(fold, index=split.index), archived


def run(root, output):
    baseline = read_config(root / "configs/baseline.yaml")
    boosting = yaml.safe_load((root / "configs/hw2.yaml").read_text())[
        "boosting"
    ]
    dev, split, _ = load_development(root, baseline)
    split = split.set_index("PassengerId")
    features = dev.set_index("PassengerId")
    target = features.pop("Transported")
    threshold = baseline["logistic"]["threshold"]
    ledgers, scores = [], []
    for protocol, fold, archive in comparisons(
        output, split, baseline["split"]["seeds"]
    ):
        archive = archive.set_index("PassengerId")
        if (not archive.index.is_unique
                or set(archive.index) != set(split.index)):
            raise ValueError(f"Invalid passenger IDs in {protocol}")
        keys = ["group", "fold", "y_true", "MC"]
        rows = archive.loc[split.index, keys].copy()
        if (not rows.group.eq(split.group).all()
                or not rows.fold.eq(fold).all()
                or not rows.y_true.eq(target).all()
                or not rows.MC.between(0, 1).all()
                or fold.groupby(split.group).nunique().max() != 1):
            raise ValueError(f"Inconsistent MC reference in {protocol}")
        rows.insert(0, "protocol", protocol)
        for name, columns in VARIANTS.items():
            for held in sorted(fold.unique()):
                valid = fold.eq(held)
                model = spending_model(baseline, boosting, columns)
                model.fit(features.loc[~valid], target.loc[~valid])
                rows.loc[valid, name] = model.predict_proba(
                    features.loc[valid]
                )[:, 1]
        models = ["MC", *VARIANTS]
        outcomes = rows[models].ge(threshold).eq(target, axis="index")
        groups = outcomes.groupby(rows.group)
        sampled = bootstrap_scores(groups.sum().to_numpy(),
                                   groups.size().to_numpy())
        deltas = 100 * (sampled - sampled[:, [0]])
        intervals = np.percentile(deltas, [2.5, 97.5], axis=0).T
        before = outcomes.MC
        for name, (low, high) in zip(models, intervals, strict=True):
            correct = outcomes[name]
            by_fold = correct.groupby(fold).mean()
            scores.append({
                "protocol": protocol, "model": name, "reference": "MC",
                "passengers": len(rows), "correct": int(correct.sum()),
                "mean_accuracy": by_fold.mean(), "sd_accuracy": by_fold.std(),
                "log_loss": log_loss(target, rows[name]),
                "fixed": int((correct & ~before).sum()),
                "introduced": int((~correct & before).sum()),
                "ci_low_pp": low, "ci_high_pp": high,
            })
        ledgers.append(rows.reset_index())
    with TemporaryDirectory(dir=output) as directory:
        stage = Path(directory)
        pd.concat(ledgers, ignore_index=True).to_csv(
            stage / "spending_predictions.csv.gz", index=False,
            float_format="%.17g", compression={"method": "gzip", "mtime": 0},
        )
        pd.DataFrame(scores).to_csv(stage / "spending.csv", index=False,
                                    float_format="%.12g")
        for path in stage.iterdir():
            path.replace(output / path.name)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "reports/hw2")
    with threadpool_limits(limits=1):
        run(root, parser.parse_args().output)
