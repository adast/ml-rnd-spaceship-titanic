"""Сравнить модели ДЗ2 по сохранённым проверочным прогнозам."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

SEEDS = (42, 17, 2026)
PRIMARY = SEEDS[0]
FOLDS = tuple(range(5))
MODELS = ("B1", "C", "M", "MC")
COMPARISONS = (
    ("MC", "B1"), ("C", "B1"), ("M", "B1"), ("MC", "M"), ("MC", "C")
)
BOOTSTRAP_SEED = 9187
REPETITIONS = 2000
FOLD_KEY = ["seed", "model", "fold"]


def read_ledgers(folder):
    folds = pd.read_csv(folder / "folds.csv")
    data = pd.read_csv(
        folder / "predictions.csv.gz",
        dtype={"PassengerId": str, "group": str},
        converters={"probability": float},
    )
    required = [
        "PassengerId", "group", *FOLD_KEY, "y_true", "probability", "y_pred"
    ]
    if data[required].isna().any().any():
        raise ValueError("Missing values in the prediction ledger")
    if data.duplicated(["seed", "model", "PassengerId"]).any():
        raise ValueError("Duplicate passenger in a model/seed pair")
    if not data["y_true"].isin([0, 1]).all():
        raise ValueError("Expected binary labels")
    if not data["probability"].between(0, 1).all():
        raise ValueError("Invalid predicted probability")
    if not data["y_pred"].eq(data["probability"] >= 0.5).all():
        raise ValueError("Prediction differs from the 0.5 decision rule")
    passengers = data.groupby("PassengerId")
    if not passengers[["group", "y_true"]].nunique().eq(1).all().all():
        raise ValueError("Passenger groups or labels differ between runs")
    if not passengers.size().eq(len(SEEDS) * len(MODELS)).all():
        raise ValueError("Models and seeds must predict the same passengers")
    if not data.groupby(["seed", "group"])["fold"].nunique().eq(1).all():
        raise ValueError("Travel group assigned to multiple folds in one seed")

    expected = pd.MultiIndex.from_product(
        [SEEDS, MODELS, FOLDS], names=FOLD_KEY
    ).sort_values()
    folds = folds.set_index(FOLD_KEY).sort_index()
    data["correct"] = data["y_pred"] == data["y_true"]
    probability = data["probability"]
    p_true = np.where(data["y_true"], probability, 1 - probability)
    eps = np.finfo(float).eps
    data["loss"] = -np.log(np.clip(p_true, eps, 1 - eps))
    measured = data.groupby(FOLD_KEY).agg(
        n_validation=("correct", "size"),
        accuracy=("correct", "mean"),
        log_loss=("loss", "mean"),
    )
    for index in (folds.index, measured.index):
        if not index.equals(expected):
            raise ValueError("Unexpected set of seeds, models or folds")
    measured["n_train"] = passengers.ngroups - measured["n_validation"]
    for name in measured:
        if not np.allclose(
            folds[name], measured[name], rtol=1e-9, atol=1e-10
        ):
            raise ValueError(f"Fold ledger disagrees with predictions: {name}")
    return folds, data


def bootstrap_intervals(primary):
    """Оценить интервалы для разностей, повторно выбирая целые группы."""
    by_group = primary.groupby(["group", "model"])["correct"]
    successes = by_group.sum().unstack()[list(MODELS)].to_numpy()
    counts = by_group.size().unstack()[MODELS[0]].to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_scores = []
    for start in range(0, REPETITIONS, 64):
        batch = min(64, REPETITIONS - start)
        sample = rng.integers(len(counts), size=(batch, len(counts)))
        sampled_scores.append(
            successes[sample].sum(axis=1) / counts[sample].sum(axis=1)[:, None]
        )
    scores = pd.DataFrame(np.concatenate(sampled_scores), columns=MODELS)
    return {
        (a, b): np.percentile(100 * (scores[a] - scores[b]), [2.5, 97.5])
        for a, b in COMPARISONS
    }


def summarize(folds, data):
    keys = ["seed", "model"]
    summary = folds.groupby(keys).agg(
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std")
    ).join(data.groupby(keys).agg(
        n_passengers=("correct", "size"), pooled_accuracy=("correct", "mean"),
        pooled_log_loss=("loss", "mean"),
    ))
    order = pd.MultiIndex.from_product([SEEDS, MODELS], names=keys)
    summary = summary.loc[order]
    intervals = bootstrap_intervals(data.loc[data["seed"] == PRIMARY])
    rows = []
    for seed in SEEDS:
        accuracy = folds.loc[seed, "accuracy"].unstack("model")
        correct = data.loc[data["seed"] == seed].pivot(
            index="PassengerId", columns="model", values="correct"
        )
        for candidate, reference in COMPARISONS:
            delta = 100 * (accuracy[candidate] - accuracy[reference])
            fixed = int((correct[candidate] & ~correct[reference]).sum())
            introduced = int((~correct[candidate] & correct[reference]).sum())
            low, high = (np.nan, np.nan)
            if seed == PRIMARY:
                low, high = intervals[candidate, reference]
            rows.append({
                "seed": seed, "candidate": candidate, "reference": reference,
                "mean_delta_pp": delta.mean(),
                "folds_improved": int((delta > 0).sum()),
                **{f"fold_{f}_delta_pp": delta.loc[f] for f in FOLDS},
                "n_passengers": len(correct), "errors_fixed": fixed,
                "errors_introduced": introduced,
                "pooled_delta_pp": 100 * (fixed - introduced) / len(correct),
                "pooled_ci95_low_pp": low, "pooled_ci95_high_pp": high,
            })
    return summary.reset_index(), pd.DataFrame(rows)


def plot_ablations(primary, path):
    fig = Figure(figsize=(11, 5.1))
    left, right = fig.subplots(1, 2, sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.15, right=0.97, bottom=0.29, top=0.76, wspace=0.11
    )
    fig.suptitle(
        "Вклад номера каюты и градиентного бустинга",
        x=0.04, y=0.97, ha="left", weight="bold", fontsize=18, color="#1b2833",
    )
    fig.text(0.04, 0.885,
             "Разность долей верных ответов, п. п.   ·   Разбиение 42",
             color="#435362", fontsize=11)
    y = np.arange(len(primary))
    offsets = np.linspace(-0.19, 0.19, len(FOLDS))
    for offset, f in zip(offsets, FOLDS, strict=True):
        left.scatter(primary[f"fold_{f}_delta_pp"], y + offset,
                     s=25, color="#8293a1", zorder=3)
    navy, teal = "#234b6c", "#08786b"
    left.scatter(primary["mean_delta_pp"], y, marker="D", s=47, color=navy,
                 edgecolor="white", linewidth=0.8, zorder=4)
    right.hlines(
        y, primary["pooled_ci95_low_pp"], primary["pooled_ci95_high_pp"],
        color=teal, linewidth=2.2, zorder=3,
    )
    right.scatter(primary["pooled_delta_pp"], y, s=50, color=teal,
                  edgecolor="white", linewidth=0.8, zorder=4)
    left.set_yticks(y, primary["candidate"] + " − " + primary["reference"])
    left.set_ylim(len(y) - 0.45, -0.48)
    left.set_title("По частям выборки", loc="left",
                   fontsize=13, weight="bold")
    right.set_title("По всем пассажирам", loc="left",
                    fontsize=13, weight="bold")
    left.xaxis.set_major_formatter(
        lambda value, _: f"{value:g}".replace(".", ",").replace("-", "−")
    )
    for axis in (left, right):
        axis.grid(axis="x", color="#dce2e5", linewidth=0.8)
        axis.axvline(0, color="#667680", linewidth=1.1, linestyle="--")
        axis.tick_params(axis="both", length=0, pad=5, labelsize=10)
        axis.spines[:].set_visible(False)
        axis.set_axisbelow(True)
        axis.margins(x=0.08)
    left.legend(
        [Line2D([], [], linestyle="", marker="o", color="#8293a1"),
         Line2D([], [], linestyle="", marker="D", color=navy)],
        ["Одна часть", "Среднее по пяти частям"],
        frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.11),
        ncols=2, handletextpad=0.45, columnspacing=1, fontsize=9,
    )
    right.legend([Line2D([], [], color=teal, marker="o")],
                 ["Разность и 95%-й интервал"], frameon=False,
                 loc="upper left", bbox_to_anchor=(-0.02, -0.11), fontsize=9)
    fig.text(0.04, 0.108,
             "B1 и C используют логистическую регрессию, M и MC используют "
             "бустинг. В C и MC добавлен номер каюты.",
             fontsize=9)
    fig.text(0.04, 0.057,
             "Для оценки интервалов 2000 раз выбирали группы с возвращением. "
             "Модели сравнивали на одних и тех же пассажирах.", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_folder = Path(__file__).resolve().parents[1] / "reports/hw2"
    parser.add_argument("--output", type=Path, default=default_folder,
                        help="Папка с folds.csv и predictions.csv.gz")
    folder = parser.parse_args().output
    folds, predictions = read_ledgers(folder)
    summary, ablations = summarize(folds, predictions)
    summary.to_csv(folder / "summary.csv", index=False, float_format="%.12g")
    ablations.to_csv(
        folder / "ablations.csv", index=False, float_format="%.12g"
    )
    primary = ablations.loc[ablations["seed"] == PRIMARY]
    plot_ablations(primary, folder / "ablations.png")
    print(summary.to_string(index=False))
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
