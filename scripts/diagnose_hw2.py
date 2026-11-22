"""Проверить порядок номеров кают и разобрать ошибки по палубам и криосну."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from summarize_hw2 import SEEDS, read_ledgers


def development_passengers(root):
    manifest = json.loads((root / "data/splits/manifest.json").read_text())
    split_path = root / "data/splits/assignments.csv"
    train_path = root / "data/raw/train.csv"
    for path, key in (
        (split_path, "assignments_sha256"), (train_path, "data_sha256")
    ):
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != manifest[key]:
                raise ValueError(f"Изменился исходный файл {path}")

    split = pd.read_csv(
        split_path, dtype={"group": str}, index_col="PassengerId"
    )
    dev = split.loc[split["role"].eq("development")].copy()
    held = split.loc[split["role"].eq("final_holdout")]
    if set(dev["group"]) & set(held["group"]):
        raise ValueError("Рабочая и отложенная выборки имеют общие группы")
    raw = pd.read_csv(
        train_path, usecols=["PassengerId", "Cabin", "CryoSleep"],
        dtype={"CryoSleep": str}, index_col="PassengerId"
    ).loc[dev.index]
    cabin = raw["Cabin"].str.extract(
        r"^(?P<deck>[^/]+)/(?P<number>[0-9]+)/(?P<side>[^/]+)$"
    )
    cabin["number"] = pd.to_numeric(cabin["number"])
    dev["CryoSleep"] = raw["CryoSleep"].fillna("missing")
    return dev.join(cabin, validate="one_to_one")


def cabin_order(dev):
    rows = []
    for (deck, side), block in dev.groupby(["deck", "side"]):
        pairs = block[["group", "number"]].drop_duplicates()
        group_order = pairs["group"].astype(int)
        # Точное равенство рангов учитывает и порядок, и совпадающие номера.
        n_groups = group_order.nunique()
        comparable = n_groups > 1 and pairs["number"].nunique() > 1
        ranks_equal = group_order.rank().equals(pairs["number"].rank())
        rows.append({
            "deck": deck, "side": side, "n_passengers": len(block),
            "n_groups": n_groups, "n_group_cabin_pairs": len(pairs),
            "equal_ranks": ranks_equal if len(pairs) > 1 else pd.NA,
            "spearman": (group_order.corr(pairs["number"], method="spearman")
                         if comparable else float("nan")),
        })
    return pd.DataFrame(rows)


def cabin_errors(dev, predictions):
    rows = []
    for seed in SEEDS:
        selected = predictions.loc[predictions["seed"].eq(seed)]
        if set(selected["PassengerId"]) != set(dev.index):
            raise ValueError("Состав проверочных пассажиров изменился")
        actual = dev.loc[selected["PassengerId"]]
        for name, column in (("group", "group"), ("fold", f"fold_{seed}")):
            expected = actual[column].to_numpy()
            if not (selected[name].to_numpy() == expected).all():
                raise ValueError("Группы или фолды расходятся с данными HW1")
        pair = selected.loc[selected["model"].isin(["M", "MC"])]
        correct = pair.pivot(index="PassengerId", columns="model",
                             values="correct")
        scores = pair.pivot(index="PassengerId", columns="model",
                            values="probability")
        truth = pair.loc[pair["model"].eq("M")].set_index("PassengerId")
        paired = pd.DataFrame({
            "deck": dev["deck"].fillna("missing"), "y_true": truth["y_true"],
            "fixed": correct["MC"] & ~correct["M"],
            "introduced": correct["M"] & ~correct["MC"],
        })
        by_deck = paired.groupby("deck")
        counts = by_deck.agg(
            n_passengers=("deck", "size"), n_positive=("y_true", "sum"),
            prevalence=("y_true", "mean"), errors_fixed=("fixed", "sum"),
            errors_introduced=("introduced", "sum"),
        )
        for deck, block in by_deck:
            # При одном классе сравнить ранжирование двух классов нельзя.
            for model in ("M", "MC"):
                counts.loc[deck, f"ap_{model}"] = (
                    average_precision_score(block["y_true"],
                                            scores.loc[block.index, model])
                    if block["y_true"].nunique() == 2 else float("nan")
                )
        counts = counts.reset_index()
        counts.insert(0, "seed", seed)
        counts["net_correct"] = (
            counts["errors_fixed"] - counts["errors_introduced"]
        )
        counts["delta_pp"] = (
            100 * counts["net_correct"] / counts["n_passengers"]
        )
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def hw1_errors(dev, predictions):
    """Сравнить B1 и M на одних пассажирах для каждого значения CryoSleep."""
    pair = predictions.loc[predictions["model"].isin(["B1", "M"])]
    correct = pair.pivot(
        index=["seed", "PassengerId"], columns="model", values="correct"
    )
    correct = correct.join(dev["CryoSleep"], on="PassengerId")
    counts = correct.groupby(["seed", "CryoSleep"]).agg(
        n_passengers=("B1", "size"), correct_B1=("B1", "sum"),
        correct_M=("M", "sum"),
    )
    counts["net_correct"] = counts["correct_M"] - counts["correct_B1"]
    return counts.loc[list(SEEDS)].reset_index()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "reports/hw2",
                        help="Каталог с прогнозами и метриками фолдов")
    folder = parser.parse_args().output
    dev = development_passengers(root)
    _, predictions = read_ledgers(folder)
    order = cabin_order(dev)
    errors = cabin_errors(dev, predictions)
    for table, name in ((order, "cabin_order"), (errors, "cabin_errors"),
                        (hw1_errors(dev, predictions), "hw1_errors")):
        table.to_csv(folder / f"{name}.csv", index=False, float_format="%.12g")
