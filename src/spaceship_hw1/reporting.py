"""Generate figures and tables directly from saved out-of-fold evidence."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)
from threadpoolctl import threadpool_limits

from .experiment import write_json
from .features import RAW_FEATURES, SPENDING, make_model

NAMES = {
    "dummy_prior": "Частоты классов",
    "logistic_core": "ЛР без каюты",
    "logistic_cabin": "B1: ЛР с каютой",
    "logistic_missing": "ЛР + индикаторы",
    "logistic_log_spend": "ЛР + log1p",
}


CONTRASTS = {
    "cabin_features": "Палуба и борт каюты",
    "missing_indicators": "Индикаторы числовых пропусков",
    "log_spending": "Преобразование расходов log1p",
}
ERROR_NAMES = {
    "False positive": "Ложноположительная (FP)",
    "False negative": "Ложноотрицательная (FN)",
}
ERROR_FIELDS = [
    "PassengerId",
    "fold",
    "y_true",
    "y_pred",
    "probability",
    "missing_count",
    *RAW_FEATURES,
]


def markdown_table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(clean, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(map(clean, row)) + " |" for row in rows],
        ]
    )


def save_figure(fig, figures, name):
    fig.savefig(figures / (name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def primary_records(dev, predictions, config):
    primary_seed = config["split"]["primary_seed"]
    selected = predictions[
        (predictions["protocol"] == "grouped")
        & (predictions["seed"] == primary_seed)
        & (predictions["model"] == config["primary_model"])
    ]
    records = selected.merge(
        dev,
        on="PassengerId",
        how="left",
        validate="one_to_one",
        suffixes=("", "_raw"),
    )
    if len(records) != len(dev) or records["Transported"].isna().any():
        raise AssertionError(
            "Reporting evidence does not cover development rows"
        )
    records["missing_count"] = records[RAW_FEATURES].isna().sum(axis=1)
    records["group_size"] = records["group"].map(
        records.groupby("group").size()
    )
    records["correct"] = records["y_true"].eq(records["y_pred"])
    records["confidence"] = np.maximum(
        records["probability"], 1 - records["probability"]
    )

    return records


def plot_data(dev, figures):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    counts = dev["Transported"].value_counts().sort_index()
    axes[0].bar(["Не перемещен (0)", "Перемещен (1)"], counts.to_numpy(),
                color=["#577590", "#43aa8b"])
    axes[0].set(title=f"Классы рабочей выборки (n={len(dev)})",
                ylabel="Число пассажиров")
    missing = dev[RAW_FEATURES].isna().mean().sort_values()
    axes[1].barh(missing.index, missing.to_numpy() * 100, color="#577590")
    axes[1].set(xlabel="Доля пропусков (%)", title="Признаки рабочей выборки")
    save_figure(fig, figures, "01_data_audit")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for ax, name in zip(axes.flat, ["Age", *SPENDING]):
        values = dev[name].dropna().to_numpy()
        ax.hist(np.log1p(values) if name in SPENDING else values,
                bins=30, color="#577590")
        ax.set(title=name + (" (после log1p)" if name in SPENDING else ""),
               ylabel="Число пассажиров")
    save_figure(fig, figures, "02_numeric_distributions")


def plot_model_comparison(summary, figures):
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    means = [x["fold_accuracy_mean"] * 100 for x in summary["models"]]
    deviations = [x["fold_accuracy_sd"] * 100 for x in summary["models"]]
    ax.bar([NAMES[x["model"]] for x in summary["models"]], means,
           yerr=deviations, capsize=5,
           color=["#aaaaaa", "#577590", "#43aa8b", "#90be6d", "#f9c74f"])
    for i, value in enumerate(means):
        ax.text(i, value + deviations[i] + 0.5, f"{value:.2f}", ha="center")
    ax.set(ylim=(0, 100), ylabel="Доля верных ответов (%)",
           title="Пять групповых фолдов; отрезки: стандартное отклонение")
    save_figure(fig, figures, "03_model_comparison")


def plot_diagnostics(records, figures):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        records["y_true"], records["y_pred"],
        display_labels=["Нет (0)", "Да (1)"], cmap="Blues", colorbar=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set(title="Матрица ошибок B1 по OOF-прогнозам",
                   xlabel="Предсказанный класс", ylabel="Истинный класс")
    fpr, tpr, _ = roc_curve(records["y_true"], records["probability"])
    axes[0, 1].plot(fpr, tpr, color="#43aa8b")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 1].set(xlabel="Доля ложноположительных ответов",
                   ylabel="Доля найденных положительных объектов",
                   title="ROC по OOF-прогнозам рабочей выборки")
    precision, recall, _ = precision_recall_curve(
        records["y_true"], records["probability"]
    )
    axes[1, 0].plot(recall, precision, color="#43aa8b")
    axes[1, 0].axhline(records["y_true"].mean(), color="gray", linestyle="--")
    axes[1, 0].set(xlabel="Полнота (recall)", ylabel="Точность (precision)",
                   title="Точность и полнота")
    observed, predicted = calibration_curve(
        records["y_true"], records["probability"],
        n_bins=10, strategy="quantile",
    )
    axes[1, 1].plot(predicted, observed, "o-", color="#43aa8b")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 1].set(xlabel="Средняя предсказанная вероятность",
                   ylabel="Наблюдаемая доля класса 1",
                   title="Калибровка: 10 квантильных групп")
    save_figure(fig, figures, "04_primary_diagnostics")


def plot_seed_sensitivity(seed_means, figures):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    values = seed_means.to_numpy() * 100
    image = ax.imshow(values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(seed_means.columns)),
                  [str(x) for x in seed_means.columns])
    ax.set_yticks(range(len(seed_means)), [NAMES[x] for x in seed_means.index])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                    color="white" if values[i, j] > 70 else "black")
    ax.set(xlabel="Начальное число генератора разбиений (seed)",
           title="Средняя доля верных ответов по пяти фолдам (%)")
    fig.colorbar(image, ax=ax)
    save_figure(fig, figures, "05_seed_sensitivity")


def plot_ablations(intervals, figures):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for i, label in enumerate(CONTRASTS):
        x = intervals["contrasts"][label]
        error = np.array([[x["estimate"] - x["low"]],
                          [x["high"] - x["estimate"]]]) * 100
        ax.errorbar(x["estimate"] * 100, i, xerr=error,
                    fmt="o", color="#577590", capsize=5)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(range(len(CONTRASTS)), list(CONTRASTS.values()))
    ax.set(xlabel="Парная разность долей верных OOF-ответов (п.п.)",
           title="Условные 95%-е интервалы бутстрэпа без переобучения")
    save_figure(fig, figures, "06_paired_ablations")


def summarize_slices(records, output, figures):
    dimensions = {
        "HomePlanet": records["HomePlanet"].fillna("Missing").astype(str),
        "CryoSleep": records["CryoSleep"].fillna("Missing").astype(str),
        "Missing fields": records["missing_count"].map(
            lambda x: "0" if x == 0 else "1" if x == 1 else "2+"
        ),
        "Travel group": records["group_size"].map(
            lambda x: "Singleton" if x == 1 else "Multiple passengers"
        ),
    }
    names = {
        "HomePlanet": "Исходная планета (HomePlanet)",
        "CryoSleep": "Криосон (CryoSleep)",
        "Missing fields": "Число пропусков",
        "Travel group": "Группа путешественников",
        "Missing": "Пропуск",
        "True": "Да",
        "False": "Нет",
        "Singleton": "Один пассажир",
        "Multiple passengers": "Несколько",
    }
    slices = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, (dimension, labels_series) in zip(axes.flat, dimensions.items()):
        table = (
            records.assign(slice=labels_series)
            .groupby("slice")
            .agg(
                n=("correct", "size"),
                accuracy=("correct", "mean"),
                positive_rate=("y_true", "mean"),
            )
        )
        ax.bar(
            [names.get(str(x), str(x)) for x in table.index],
            table["accuracy"] * 100,
            color="#577590",
        )
        for index, (_, row) in enumerate(table.iterrows()):
            ax.text(
                index,
                row["accuracy"] * 100 + 1,
                f"n={int(row['n'])}",
                ha="center",
                fontsize=8,
            )
        ax.set(
            ylim=(0, 110),
            ylabel="Доля верных ответов (%)",
            title=names[dimension],
        )
        for label, row in table.iterrows():
            slices.append(
                {
                    "dimension": dimension,
                    "slice": str(label),
                    "n": int(row["n"]),
                    "accuracy": float(row["accuracy"]),
                    "positive_rate": float(row["positive_rate"]),
                }
            )
    save_figure(fig, figures, "07_error_slices")
    pd.DataFrame(slices).to_csv(output / "slices.csv", index=False)

    return slices


def plot_protocol(metrics, config, figures):
    primary_seed = config["split"]["primary_seed"]
    grouped = metrics[metrics["protocol"].eq("grouped")]
    random = metrics[metrics["protocol"].eq("random_passenger")]
    if len(random):
        main = grouped[
            (grouped["seed"] == primary_seed)
            & grouped["model"].eq(config["primary_model"])
        ]
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.bar(
            ["Новые группы", "Случайные пассажиры"],
            [main["accuracy"].mean() * 100, random["accuracy"].mean() * 100],
            yerr=[
                main["accuracy"].std() * 100,
                random["accuracy"].std() * 100,
            ],
            capsize=5,
            color=["#43aa8b", "#577590"],
        )
        ax.set(
            ylim=(0, 100),
            ylabel="Доля верных ответов (%)",
            title="Разные сценарии; отрезки показывают стандартное отклонение",
        )
        save_figure(fig, figures, "08_protocol_sensitivity")

    return random["accuracy"].mean() if len(random) else None


def analyze_errors(records, output):
    errors = records[~records["correct"]]
    # Анализ добавлен после экспериментов; примеры характерны по уверенности.
    typical_examples, error_types = [], []
    for label, target in [("False positive", 0), ("False negative", 1)]:
        subset = errors[errors["y_true"].eq(target)].copy()
        if subset.empty:
            continue
        median = float(subset["confidence"].median())
        subset["distance_to_median"] = (subset["confidence"] - median).abs()
        row = subset.sort_values(["distance_to_median", "PassengerId"]).iloc[0]
        error_types.append(
            {
                "error_type": label,
                "count": len(subset),
                "share_of_errors": len(subset) / len(errors),
                "class_error_rate": len(subset)
                / int(records["y_true"].eq(target).sum()),
                "median_confidence": median,
            }
        )
        typical_examples.append(
            {
                "error_type": label,
                "selection_rule": (
                    "Ближайшая к медиане уверенность внутри типа ошибки; "
                    "при равенстве выбирается минимальный PassengerId"
                ),
                "error_type_count": len(subset),
                "median_confidence": median,
                **json.loads(row[ERROR_FIELDS].to_json()),
            }
        )
    analysis = {
        "scope": (
            "Описательный анализ OOF-ошибок рабочей выборки добавлен "
            "при проверке после экспериментов; без выбора модели "
            "и обращения к окончательной отложенной выборке"
        ),
        "selection_limitation": (
            "Примеры характерны по уверенности внутри FP/FN; "
            "представительность по всем признакам не утверждается"
        ),
        "total_errors": len(errors),
        "error_types": error_types,
        "typical_examples": typical_examples,
    }
    write_json(output / "error_analysis.json", analysis)
    confident = errors.sort_values(
        ["confidence", "PassengerId"], ascending=[False, True]
    )
    selections = [
        ("Самая уверенная FP", confident[confident["y_pred"].eq(1)]),
        ("Самая уверенная FN", confident[confident["y_pred"].eq(0)]),
        ("Ошибка с наибольшим числом пропусков", errors.sort_values(
            ["missing_count", "confidence", "PassengerId"],
            ascending=[False, False, True],
        )),
    ]
    examples = []
    used = set()
    for reason, subset in selections:
        subset = subset[~subset["PassengerId"].isin(used)]
        if len(subset):
            row = subset.iloc[0]
            used.add(row["PassengerId"])
            record = {
                "selection_rule": reason,
                **json.loads(row[ERROR_FIELDS].to_json()),
            }
            examples.append(record)
    write_json(output / "error_examples.json", examples)

    return analysis, examples


def explain_errors(dev, assignment, config, examples, output):
    primary_seed = config["split"]["primary_seed"]
    # Refit only the already specified training fold to explain each OOF error.
    # This is an analysis of the fixed model, not a model-selection experiment.
    explanations, fitted_folds = [], {}
    dev_assignment = assignment.loc[
        assignment["role"].eq("development")
    ].reset_index(drop=True)
    dev_ordered = dev.reset_index(drop=True)
    for example in examples:
        fold = int(example["fold"])
        if fold not in fitted_folds:
            train = ~dev_assignment[f"fold_{primary_seed}"].eq(fold).to_numpy()
            with threadpool_limits(limits=config["runtime"]["blas_threads"]):
                fitted_folds[fold] = make_model(
                    config["primary_model"], config
                ).fit(
                    dev_ordered.loc[train].drop(columns="Transported"),
                    dev_ordered.loc[train, "Transported"],
                )
        model = fitted_folds[fold]
        raw = dev_ordered[
            dev_ordered["PassengerId"].eq(example["PassengerId"])
        ].drop(columns="Transported")
        features = model.named_steps["features"].transform(raw)
        transformed = model.named_steps["preprocess"].transform(features)[0]
        names = model.named_steps["preprocess"].get_feature_names_out()
        classifier = model.named_steps["classifier"]
        terms = transformed * classifier.coef_[0]
        probability = float(model.predict_proba(raw)[0, 1])
        np.testing.assert_allclose(
            probability, example["probability"], rtol=0, atol=1e-9
        )
        contributions = [
            {
                "feature": str(name),
                "transformed_value": float(value),
                "log_odds_contribution": float(term),
            }
            for name, value, term in zip(names, transformed, terms)
        ]
        contributions.sort(
            key=lambda x: abs(x["log_odds_contribution"]), reverse=True
        )
        explanations.append(
            {
                "PassengerId": example["PassengerId"],
                "fold": fold,
                "reproduced_oof_probability": probability,
                "intercept": float(classifier.intercept_[0]),
                "logit_sum": float(classifier.intercept_[0] + terms.sum()),
                "contributions": contributions,
                "interpretation": (
                    "Это разложение прогноза модели, "
                    "а не объяснение причины события."
                ),
            }
        )
    write_json(output / "error_explanations.json", explanations)
    return explanations


def plot_contributions(explanations, figures):
    if not explanations:
        return
    fig, axes = plt.subplots(
        len(explanations),
        1,
        figsize=(11, 3.3 * len(explanations)),
        layout="constrained",
        squeeze=False,
    )
    for ax, explanation in zip(axes.flat, explanations):
        terms = explanation["contributions"][:6]
        values = [x["log_odds_contribution"] for x in terms]
        ax.barh(
            [x["feature"] for x in terms],
            values,
            color=["#43aa8b" if x >= 0 else "#f9844a" for x in values],
        )
        ax.axvline(0, color="gray", linewidth=1)
        ax.invert_yaxis()
        ax.set(
            xlabel="Вклад в логарифм шансов (без константы и прочих членов)",
            title=(
                f"OOF-ошибка {explanation['PassengerId']}; "
                f"p={explanation['reproduced_oof_probability']:.4f}; "
                f"фолд {explanation['fold']}"
            ),
        )
    save_figure(fig, figures, "09_error_contributions")


def example_table(examples, selection):
    headers = ["PassengerId", "Фолд", "Истинный", "Прогноз", "P(Transported)"]
    rows = []
    for item in examples:
        label = item[selection]
        row = [ERROR_NAMES.get(label, label),
               item["PassengerId"], item["fold"],
               item["y_true"], item["y_pred"], f"{item['probability']:.4f}"]
        if selection == "selection_rule":
            row.append(item["missing_count"])
        rows.append(row)
    if selection == "selection_rule":
        return markdown_table(["Правило отбора", *headers, "Пропуски"], rows)
    return markdown_table(["Тип ошибки", *headers], rows)


def write_tables(summary, intervals, analysis, examples, output):
    error_type_table = markdown_table(
        ["Тип ошибки", "Число", "Доля всех ошибок",
         "Доля ошибок в истинном классе"],
        [[ERROR_NAMES[x["error_type"]], x["count"],
          f"{x['share_of_errors'] * 100:.2f}%",
          f"{x['class_error_rate'] * 100:.2f}%"]
         for x in analysis["error_types"]],
    )
    typical_table = example_table(analysis["typical_examples"], "error_type")
    table = markdown_table(
        ["Модель", "Средняя accuracy ± SD",
         "OOF F1", "OOF AUC", "OOF log loss"],
        [[x["model"],
          f"{x['fold_accuracy_mean'] * 100:.3f}% ± "
          f"{x['fold_accuracy_sd'] * 100:.3f} п.п.",
          f"{x['f1']:.4f}", f"{x['roc_auc']:.4f}", f"{x['log_loss']:.4f}"]
         for x in summary["models"]],
    )
    contrasts = intervals["contrasts"]
    contrast_table = markdown_table(
        ["Изменение", "Разность accuracy (п.п.)",
         "Условный 95%-й интервал (п.п.)"],
        [[title, f"{contrasts[label]['estimate'] * 100:+.3f}",
          f"[{contrasts[label]['low'] * 100:+.3f}, "
          f"{contrasts[label]['high'] * 100:+.3f}]"]
         for label, title in CONTRASTS.items()],
    )
    error_table = example_table(examples, "selection_rule")
    sections = [
        "# Результаты на рабочей выборке", table,
        "## Фиксированные парные абляции", contrast_table,
        "Интервалы рассчитаны по прогнозам уже обученных моделей: "
        "целые группы путешественников повторно выбирались с возвращением. "
        "Полная вариативность повторного обучения и зависимость между "
        "пересекающимися обучающими частями в них не учтены.",
        "## Частоты ошибок и примеры с медианной уверенностью",
        error_type_table, typical_table,
        "Описательный анализ добавлен при проверке после экспериментов. "
        "Примеры характерны по уверенности внутри FP/FN, а не по всем "
        "признакам. Полные записи и правила: `error_analysis.json`.",
        "## Ошибки с крайними значениями уверенности или числа пропусков",
        error_table,
        "Эти примеры не являются типичными и не оценивают частоты ошибок. "
        "Полные записи: `error_examples.json`; точные вычисления модели: "
        "`error_explanations.json`.",
    ]
    (output / "RESULTS.md").write_text("\n\n".join(sections) + "\n")
    return {
        "results_table": table,
        "contrast_table": contrast_table,
        "error_table": error_table,
        "error_type_table": error_type_table,
        "typical_error_table": typical_table,
    }


def generate_report(config, frame, assignment, metrics, predictions, output):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 120}
    )
    summary = json.loads((output / "summary.json").read_text())
    intervals = json.loads((output / "group_bootstrap.json").read_text())
    dev = frame.loc[assignment["role"].eq("development")].copy()
    records = primary_records(dev, predictions, config)
    grouped = metrics[metrics["protocol"].eq("grouped")]
    seed_means = (
        grouped.groupby(["model", "seed"])["accuracy"].mean().unstack()
    )
    seed_means = seed_means.reindex(config["models"])
    seed_means.reset_index().to_csv(output / "seed_accuracy.csv", index=False)
    plot_data(dev, figures)
    plot_model_comparison(summary, figures)
    plot_diagnostics(records, figures)
    plot_seed_sensitivity(seed_means, figures)
    plot_ablations(intervals, figures)
    slices = summarize_slices(records, output, figures)
    random_mean = plot_protocol(metrics, config, figures)
    analysis, examples = analyze_errors(records, output)
    explanations = explain_errors(dev, assignment, config, examples, output)
    plot_contributions(explanations, figures)
    fragments = write_tables(summary, intervals, analysis, examples, output)
    write_json(
        output / "report_fragments.json",
        {
            **fragments,
            "slice_summary": slices,
            "development_rows": len(dev),
            "primary_errors": analysis["total_errors"],
            "random_mean_accuracy": float(random_mean)
            if random_mean is not None
            else None,
            "seed_means": json.loads(
                seed_means.reset_index().to_json(orient="records")
            ),
            "final_holdout_evaluated": False,
        },
    )
    return summary
