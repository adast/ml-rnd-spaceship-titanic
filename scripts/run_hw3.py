"""Разобрать ошибки выбранной модели ДЗ2 и проверить её устойчивость."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from hw3_errors import run as analyze_errors
from hw3_robustness import run as check_robustness
from matplotlib.figure import Figure
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


def plot_robustness(folder):
    results = pd.read_csv(folder / "robustness.csv")
    fig = Figure(figsize=(9, 3.8), layout="constrained")
    left, right = fig.subplots(1, 2, sharey=True)
    for ax, check, title, xlabel in (
        (left, "training_groups", "Размер обучающей выборки",
         "Доля групп, оставленных для обучения, %"),
        (right, "missing_bill", "Пропуск одного счёта",
         "Пассажиры со скрытым счётом,\n"
         "% от числа пассажиров с пятью известными счетами"),
    ):
        runs = results.loc[results["check"].eq(check)]
        values = 100 * runs.groupby("percent")["accuracy"].mean()
        ax.plot(values.index, values, "-o", color="#136b63", linewidth=2)
        if check == "training_groups":
            repeats = runs.loc[runs["subset_seed"].notna()]
            ax.scatter(repeats["percent"], 100 * repeats["accuracy"],
                       s=22, color="#566978", zorder=3)
        for x, y in values.items():
            top = 100 * runs.loc[runs["percent"].eq(x), "accuracy"].max()
            ax.annotate(f"{y:.2f}".replace(".", ","), (x, top),
                        xytext=(0, 9), textcoords="offset points",
                        ha="center", color="#16483e")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set(xlabel=xlabel, xticks=values.index)
        ax.xaxis.label.set_fontsize(9)
        ax.margins(x=0.12, y=0.22)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(
            lambda value, _: f"{value:g}".replace(".", ",")
        )
        ax.grid(axis="y", color="#dfe5e5")
        ax.spines[["top", "right"]].set_visible(False)
    left.set_ylabel("Доля верных ответов, %")
    fig.savefig(folder / "robustness.png", dpi=170)


def plot_bill_errors(folder):
    table = pd.read_csv(folder / "bill_services.csv").set_index("service")
    table = table.drop("uniform")
    fig = Figure(figsize=(7.6, 3.2), layout="constrained")
    ax = fig.subplots()
    for column, sign, color, label in (
        ("new_fp", -1, "#a04b27", "Ошибочный прогноз «переместился»"),
        ("new_fn", 1, "#136b63", "Пропущенное перемещение"),
    ):
        bars = ax.barh(table.index, sign * table[column], height=0.58,
                       color=color, label=label)
        ax.bar_label(bars, labels=table[column].astype(int),
                     padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="#394a53", linewidth=0.8)
    ax.set(xlabel="Число новых ошибок", xticks=range(-300, 301, 100))
    ax.xaxis.set_major_formatter(lambda value, _: f"{abs(value):g}")
    ax.margins(x=0.1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2,
              frameon=False)
    fig.savefig(folder / "bill_errors.png", dpi=170)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/hw3")
    output = parser.parse_args().output
    baseline = read_config(ROOT / "configs/baseline.yaml")
    boosting = yaml.safe_load((ROOT / "configs/hw2.yaml").read_text())[
        "boosting"
    ]
    dev, split, _ = load_development(ROOT, baseline)
    reference, confidence = reference_predictions(
        ROOT / "reports/hw2", dev, split, baseline["split"]["primary_seed"]
    )
    output.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output) as directory:
        stage = Path(directory)
        confidence.to_csv(stage / "confidence.csv", float_format="%.12g")
        case_id = analyze_errors(dev, split, reference, stage)
        check_robustness(dev, split, baseline, boosting, reference, stage,
                         case_id=case_id)
        plot_robustness(stage)
        plot_bill_errors(stage)
        for path in stage.iterdir():
            path.replace(output / path.name)
    print(f"Таблицы и рисунки сохранены в {output}")


if __name__ == "__main__":
    main()
