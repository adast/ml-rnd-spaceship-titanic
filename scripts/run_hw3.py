"""Разобрать ошибки выбранной модели ДЗ2 по группам пассажиров."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from hw3_errors import run as analyze_errors
from summarize_hw2 import read_ledgers

from spaceship_hw1.data import read_config
from spaceship_hw2 import load_development

ROOT = Path(__file__).resolve().parents[1]


def reference_predictions(folder, dev, split, seed):
    """Сверить пассажиров, метки и фолды по двум архивам ДЗ2."""
    _, predictions = read_ledgers(folder)
    mc = predictions.loc[predictions["model"].eq("MC")]
    wrong = mc["y_pred"].ne(mc["y_true"])
    mc = mc.assign(
        error_repeats=wrong.groupby(mc["PassengerId"]).transform("sum")
    ).set_index("PassengerId", drop=False)
    selected = mc.loc[mc["seed"].eq(seed)]
    spending = pd.read_csv(
        folder / "spending_predictions.csv.gz",
        dtype={"PassengerId": str, "group": str},
        converters={"MC": float, "share": float},
    )
    spending = spending.loc[
        spending["protocol"].eq(f"group_{seed}")
    ].set_index("PassengerId")
    ids = pd.Index(dev["PassengerId"])
    for run_seed, archive in [*mc.groupby("seed"), (seed, spending)]:
        if not archive.index.sort_values().equals(ids.sort_values()):
            raise ValueError("Состав пассажиров расходится с архивом ДЗ2")
        aligned = archive.loc[ids]
        for name, expected in (
            ("group", split["group"]), ("fold", split[f"fold_{run_seed}"]),
            ("y_true", dev["Transported"]),
        ):
            if not np.array_equal(aligned[name], expected):
                raise ValueError(f"Расхождение с архивом ДЗ2 в {name}")
    selected = selected.loc[ids].copy()
    spending = spending.loc[ids]
    np.testing.assert_allclose(
        selected["probability"], spending["MC"], rtol=0, atol=1e-12
    )
    if not spending["share"].between(0, 1).all():
        raise ValueError("Некорректные вероятности варианта с долей расходов")
    selected["share_probability"] = spending["share"]
    return selected.reset_index(drop=True), summarize_confidence(mc)


def summarize_confidence(reference):
    """Сопоставить уверенность в выбранном классе с частотой верных ответов."""
    probability = reference["probability"]
    confidence = probability.where(reference["y_pred"].eq(1), 1 - probability)
    ranges = pd.cut(
        confidence, [0.5, 0.65, 0.8, 0.9, np.inf], right=False,
        labels=["0.50-0.65", "0.65-0.80", "0.80-0.90", "0.90-1"],
    )
    return reference.assign(
        range=ranges, confidence=confidence,
        wrong=~reference["correct"],
        changed=reference["error_repeats"].between(
            1, reference["seed"].nunique() - 1
        ),
        predicted_stayed=reference["y_pred"].eq(0),
        fp=reference["y_pred"].eq(1) & ~reference["correct"],
        fn=reference["y_pred"].eq(0) & ~reference["correct"],
        always_wrong=reference["error_repeats"].eq(3),
    ).groupby(["seed", "range"], observed=True).agg(
        passengers=("PassengerId", "size"), errors=("wrong", "sum"),
        mean_confidence=("confidence", "mean"), accuracy=("correct", "mean"),
        predicted_stayed=("predicted_stayed", "sum"),
        predicted_moved=("y_pred", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
        errors_all_splits=("always_wrong", "sum"),
        changed_across_splits=("changed", "sum"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/hw3")
    output = parser.parse_args().output
    baseline = read_config(ROOT / "configs/baseline.yaml")
    dev, split, _ = load_development(ROOT, baseline)
    reference, confidence = reference_predictions(
        ROOT / "reports/hw2", dev, split, baseline["split"]["primary_seed"]
    )
    output.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output) as directory:
        stage = Path(directory)
        confidence.to_csv(stage / "confidence.csv", float_format="%.12g")
        analyze_errors(dev, split, reference, stage)
        for path in stage.iterdir():
            path.replace(output / path.name)
    print(f"Таблицы сохранены в {output}")


if __name__ == "__main__":
    main()
