"""Проверить MC при уменьшении обучающей выборки и пропуске одного счёта."""

from hashlib import sha256

import numpy as np
import pandas as pd
from hw3_errors import spa_response
from sklearn.metrics import roc_auc_score
from summarize_hw2 import bootstrap_scores
from threadpoolctl import threadpool_limits

from spaceship_hw1.features import SPENDING
from spaceship_hw2 import make_hw2_model


def bill_loss(features):
    """По идентификаторам определить порядок пассажиров и счёт для пропуска."""
    complete = features[SPENDING].notna().all(axis=1)
    ids = features.loc[complete, "PassengerId"]
    digests = ids.map(lambda x: sha256(f"hw3-bills-v1:{x}".encode()).digest())
    ordered = digests.map(lambda d: d[:8]).sort_values(kind="stable").index
    rank = pd.Series(len(ids), index=features.index)
    rank.loc[ordered] = np.arange(len(ids))
    bills = digests.map(
        lambda d: int.from_bytes(d[8:16], "big") % len(SPENDING)
    )
    damaged = features.copy()
    for number, column in enumerate(SPENDING):
        damaged.loc[bills.index[bills.eq(number)], column] = np.nan
    return damaged, rank, complete


def comparison(target, predicted, original, complete):
    correct = predicted == target
    before = original == target
    fixed = (correct & ~before).sum()
    introduced = (before & ~correct).sum()
    return {
        "n_validation": len(target), "accuracy": correct.mean(),
        "delta_pp": 100 * (fixed - introduced) / len(target),
        "fixed": fixed, "introduced": introduced, "flips": fixed + introduced,
        "n_complete": complete.sum(),
        "complete_correct_before": before[complete].sum(),
        "complete_correct_after": correct[complete].sum(),
    }


def service_summary(changes, features, complete, groups, folds, before, truth):
    """Сравнить услуги на тех же пассажирах, включая неизменённые записи."""
    bills = features.loc[complete, SPENDING]
    correct = changes.add(before.astype(int), axis="index")
    summary = pd.DataFrame({
        "n_positive_masked": bills.gt(0).sum(),
        "fixed": changes.eq(1).sum(),
        "introduced": changes.eq(-1).sum(),
        "new_fp": changes.loc[truth.eq(0)].eq(-1).sum(),
        "new_fn": changes.loc[truth.eq(1)].eq(-1).sum(),
        "zero_amount_flips": (changes.loc[complete].ne(0) & bills.eq(0)).sum(),
        "mean_fold_accuracy": correct.groupby(folds).mean().mean(),
        "pooled_delta_pp": 100 * changes.mean(),
    })
    # Для каждого пассажира все пять вариантов пропуска равновероятны.
    uniform = summary.mean().rename("uniform").to_frame().T
    summary = pd.concat([summary, uniform])
    grouped = changes.groupby(groups)
    samples = 100 * bootstrap_scores(
        grouped.sum().to_numpy(), grouped.size().to_numpy()
    )
    samples = np.column_stack((samples, samples.mean(axis=1)))
    summary[["ci_low_pp", "ci_high_pp"]] = np.percentile(
        samples, [2.5, 97.5], axis=0
    ).T
    summary["n_validation"] = len(features)
    summary["n_complete"] = complete.sum()
    return summary.rename_axis("service").reset_index()


def best_threshold_accuracy(truth, scores):
    """Верхняя граница точности при переборе порогов по известным меткам."""
    # Повышая порог, переводим все равные оценки в класс 0 одновременно.
    _, bins = np.unique(scores, return_inverse=True)
    change = np.bincount(bins, weights=1 - 2 * truth.astype(int)).cumsum()
    return (truth.sum() + max(0, change.max())) / len(truth)


def bill_scores(scores, truth, folds, threshold):
    """Сопоставить сдвиг вероятностей и изменение порядка прогнозов."""
    summary = pd.DataFrame({
        "mean_probability": scores.mean(),
        "positive_predictions": scores.ge(threshold).sum(),
    }).astype(float)
    summary.loc["uniform"] = summary.loc[SPENDING].mean()
    summary["mean_probability_change"] = (
        summary.mean_probability - summary.loc["clean", "mean_probability"]
    )
    # Для AUC сравниваем только пары, оценённые одной обученной моделью.
    fold_auc, fold_best = [], []
    for _, part in scores.groupby(folds):
        y = truth.loc[part.index].to_numpy()
        scenarios = {name: (y, part[name].to_numpy()) for name in part}
        # Для пары из разных классов есть 25 равновероятных сочетаний счетов.
        scenarios["uniform"] = (
            np.repeat(y, len(SPENDING)),
            part[SPENDING].to_numpy().ravel(),
        )
        both_classes = np.any(y != y[0])
        fold_auc.append({
            name: roc_auc_score(labels, values) if both_classes else np.nan
            for name, (labels, values) in scenarios.items()
        })
        fold_best.append({
            name: best_threshold_accuracy(labels, values)
            for name, (labels, values) in scenarios.items()
        })
    summary["mean_fold_auc"] = pd.DataFrame(fold_auc).mean(skipna=False)
    summary["best_threshold_accuracy"] = pd.DataFrame(fold_best).mean()
    summary["delta_mean_fold_auc"] = (
        summary.mean_fold_auc - summary.loc["clean", "mean_fold_auc"]
    )
    summary["n_validation"] = len(scores)
    return summary.rename_axis("service").reset_index()


def spa_cases(model, training, validation, reference, threshold,
              case_id, output):
    """Проверить замену счёта за Spa при прогнозе «переместился»."""
    missing_count = validation[SPENDING].isna().sum(axis=1)
    only_spa = validation.Spa.isna() & missing_count.eq(1)
    selected = validation.CryoSleep.eq(False) & only_spa
    rows = []
    for i, case in validation.loc[selected].iterrows():
        prediction = reference.loc[case.PassengerId]
        if prediction.probability < threshold:
            continue
        missing = case[["HomePlanet", "Cabin"]].isna()
        row = {
            "PassengerId": case.PassengerId, "fold": int(prediction.fold),
            "y_true": int(prediction.y_true),
            "missing_probability": prediction.probability,
            "missing_match_fields": ",".join(missing.index[missing]),
            "n_donors": 0, "changed_donors": 0,
        }
        if not missing.any():
            response = spa_response(model, training, validation.loc[[i]])
            changed = response.probability.lt(threshold)
            row["n_donors"] = response.n_donors.sum()
            row["changed_donors"] = response.loc[changed, "n_donors"].sum()
            if case.PassengerId == case_id:
                response.to_csv(output / "spa_case.csv", index=False,
                                float_format="%.12g")
        rows.append(row)
    return rows


def run(dev, split, baseline, boosting, reference, output, case_id=None):
    features = dev.drop(columns="Transported")
    target = dev["Transported"]
    threshold = baseline["logistic"]["threshold"]
    probabilities = reference.set_index("PassengerId")
    if not probabilities.index.is_unique:
        raise ValueError("В архиве повторяются идентификаторы пассажиров")
    archive = probabilities.loc[dev["PassengerId"], "probability"].to_numpy()
    folds = split[f"fold_{baseline['split']['primary_seed']}"]
    groups = split["group"]
    damaged, rank, complete = bill_loss(features)
    scores = pd.DataFrame(index=dev.index, columns=["clean", *SPENDING],
                          dtype=float)
    lost_positive = (features[SPENDING].gt(0) & damaged[SPENDING].isna()).any(
        axis=1
    )
    rows = []
    spa_rows = []
    with threadpool_limits(limits=baseline["runtime"]["blas_threads"]):
        for fold in range(baseline["split"]["n_splits"]):
            valid = folds.eq(fold)
            train = ~valid
            training_groups = np.sort(groups[train].unique())
            if groups[valid].isin(training_groups).any():
                raise ValueError("Группа попала в обучение и проверку")
            model = make_hw2_model("MC", baseline, boosting)
            model.fit(features.loc[train], target[train])
            clean = model.predict_proba(features.loc[valid])[:, 1]
            np.testing.assert_allclose(
                clean, archive[valid], rtol=0, atol=1e-9
            )
            clean_class = clean >= threshold
            truth = target[valid].to_numpy()
            eligible = complete[valid].to_numpy()
            scores.loc[valid, "clean"] = clean
            if case_id is not None:
                spa_rows.extend(spa_cases(
                    model, features.loc[train], features.loc[valid],
                    probabilities, threshold, case_id, output,
                ))
            for service in SPENDING:
                masked = features.loc[valid].copy()
                masked.loc[eligible, service] = np.nan
                scores.loc[valid, service] = model.predict_proba(masked)[:, 1]
            common = {
                "fold": fold, "n_train": int(train.sum()),
                "train_groups": len(training_groups), "subset_seed": None,
            }
            rows.append({
                **common, "check": "training_groups", "percent": 100,
                "n_masked": 0, "n_masked_positive": 0,
                **comparison(truth, clean_class, clean_class, eligible),
            })
            missing_prob = model.predict_proba(damaged.loc[valid])[:, 1]
            for percent in (0, 25, 50, 100):
                cutoff = round(complete.sum() * percent / 100)
                mask = rank[valid].to_numpy() < cutoff
                predicted = np.where(mask, missing_prob >= threshold,
                                     clean_class)
                rows.append({
                    **common, "check": "missing_bill", "percent": percent,
                    "n_masked": mask.sum(),
                    "n_masked_positive": (mask & lost_positive[valid]).sum(),
                    **comparison(truth, predicted, clean_class, eligible),
                })
            for seed in (101, 202, 303):
                rng = np.random.default_rng([seed, fold])
                order = rng.permutation(training_groups)
                for percent in (20, 50):
                    n_groups = round(len(order) * percent / 100)
                    selected = groups.isin(order[:n_groups])
                    model = make_hw2_model("MC", baseline, boosting)
                    model.fit(features.loc[selected], target[selected])
                    predicted = model.predict_proba(features.loc[valid])[:, 1]
                    rows.append({
                        **common, "check": "training_groups",
                        "percent": percent, "subset_seed": seed,
                        "train_groups": n_groups, "n_train": selected.sum(),
                        "n_masked": 0, "n_masked_positive": 0, **comparison(
                            truth, predicted >= threshold, clean_class,
                            eligible,
                        ),
                    })
    detail = pd.DataFrame(rows).astype({"subset_seed": "Int64"})
    summary = detail.groupby(["check", "percent", "subset_seed"],
                             dropna=False, as_index=False).agg(
        n_validation=("n_validation", "sum"), accuracy=("accuracy", "mean"),
        delta_pp=("delta_pp", "mean"), n_masked=("n_masked", "sum"),
        n_masked_positive=("n_masked_positive", "sum"),
        fixed=("fixed", "sum"), introduced=("introduced", "sum"),
        flips=("flips", "sum"), n_complete=("n_complete", "sum"),
        complete_correct_before=("complete_correct_before", "sum"),
        complete_correct_after=("complete_correct_after", "sum"),
    )
    correct = scores.ge(threshold).eq(target, axis="index").astype("int8")
    changes = correct[SPENDING].sub(correct["clean"], axis="index")
    services = service_summary(changes, features, complete, groups, folds,
                               correct["clean"], target)
    if spa_rows:
        pd.DataFrame(spa_rows).to_csv(
            output / "spa_case_controls.csv", index=False, float_format="%.12g"
        )
    for filename, table in (
        ("robustness_folds", detail), ("robustness", summary),
        ("bill_services", services),
        ("bill_scores", bill_scores(scores, target, folds, threshold)),
    ):
        table.to_csv(output / f"{filename}.csv", index=False,
                     float_format="%.12g")
