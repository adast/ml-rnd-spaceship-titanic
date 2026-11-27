"""Разобрать ошибки модели из ДЗ2 и сравнить группы пассажиров."""

import pandas as pd
from summarize_hw2 import bootstrap_scores

from spaceship_hw1.features import RAW_FEATURES, SPENDING


def slice_metrics(frame, axis, level):
    truth, predicted = frame.y_true, frame.y_pred
    correct = predicted.eq(truth)
    share_correct = frame.share_probability.ge(0.5).eq(truth)
    return {
        "axis": axis, "level": level, "passengers": len(frame),
        "booking_groups": frame.group.nunique(), "positives": truth.sum(),
        "fp": (predicted.eq(1) & truth.eq(0)).sum(),
        "fn": (predicted.eq(0) & truth.eq(1)).sum(),
        "errors_all_splits": frame.error_repeats.eq(3).sum(),
        "accuracy": correct.mean(), "share_accuracy": share_correct.mean(),
        "share_fixed": (share_correct & ~correct).sum(),
        "share_introduced": (correct & ~share_correct).sum(),
    }


def deck_comparisons(frame):
    """Сравнить палубы при совместном повторном выборе групп бронирования."""
    awake_earth = frame.origin.eq("Earth") & frame.cryo.eq("awake")
    cohorts = pd.DataFrame({deck: frame.deck.eq(deck) for deck in "BFG"})
    pairs = [("B", "G"), ("F", "G")]
    for suffix, mask in (
        ("earth_awake", awake_earth),
        ("earth_awake_moved", awake_earth & frame.y_true.eq(1)),
        ("earth_awake_stayed", awake_earth & frame.y_true.eq(0)),
    ):
        left, right = f"F_{suffix}", f"G_{suffix}"
        cohorts[left] = cohorts["F"] & mask
        cohorts[right] = cohorts["G"] & mask
        pairs.append((left, right))
    sizes = cohorts.groupby(frame.group).sum()
    correct = cohorts.mul(frame.y_pred.eq(frame.y_true), axis="index")
    successes = correct.groupby(frame.group).sum()
    # Числители и знаменатели берём для одних и тех же выбранных групп.
    totals = pd.concat({"correct": successes, "eligible": sizes}, axis=1)
    draws = bootstrap_scores(totals.to_numpy(),
                             frame.groupby("group").size().to_numpy())
    draws = pd.DataFrame(draws, columns=totals.columns)
    rates = draws["correct"] / draws["eligible"]
    n_correct, n = successes.sum(), sizes.sum()
    accuracy = n_correct / n
    # Для обеих палуб берём общую долю переместившихся.
    matched = awake_earth & frame.deck.isin(["F", "G"])
    weight = frame.loc[matched, "y_true"].mean()
    for deck in ("F", "G"):
        cohort = f"{deck}_earth_awake"
        for table in (accuracy, rates):
            table[f"{cohort}_standardized"] = (
                weight * table[f"{cohort}_moved"]
                + (1 - weight) * table[f"{cohort}_stayed"]
            )
    pairs.append(("F_earth_awake_standardized", "G_earth_awake_standardized"))
    rows = []
    for left, right in pairs:
        low, high = (100 * (rates[left] - rates[right])).quantile(
            [0.025, 0.975]
        )
        rows.append({
            "left": left, "right": right,
            "left_correct": n_correct.get(left), "left_n": n.get(left),
            "right_correct": n_correct.get(right), "right_n": n.get(right),
            "left_accuracy": accuracy[left], "right_accuracy": accuracy[right],
            "difference_pp": 100 * (accuracy[left] - accuracy[right]),
            "low_pp": low, "high_pp": high,
        })
    return rows


def error_example(frame, pattern, predicted):
    """Выбрать ошибку с ближайшей к медиане оценкой вероятности."""
    decisions = frame.loc[frame.y_pred.eq(predicted)]
    errors = decisions.loc[decisions.y_true.ne(predicted)]
    median = errors.probability.median()
    example = errors.loc[(errors.probability - median).abs().idxmin()]
    same_missing = decisions[SPENDING].isna().eq(
        example[SPENDING].isna()
    ).all(axis=1)
    controls = decisions.loc[
        decisions.y_true.eq(predicted) & decisions.fold.eq(example.fold)
        & decisions.deck.eq(example.deck) & decisions.origin.eq(example.origin)
        & same_missing
    ]
    fields = ["HomePlanet", "Cabin", "CryoSleep", "Age", *SPENDING]
    row = {
        "pattern": pattern, "passengers": len(frame),
        "positives": frame.y_true.sum(),
        "predicted_in_direction": len(decisions),
        "direction_errors": len(errors), "PassengerId": example.name,
        "fold": example.fold, "y_true": example.y_true, "y_pred": predicted,
        "error_repeats": example.error_repeats,
        "probability": example.probability,
        **example[fields].to_dict(), "control_candidates": len(controls),
    }
    if not controls.empty:
        distance = (controls.probability - example.probability).abs()
        control = controls.loc[distance.idxmin()]
        row.update(control_id=control.name,
                   control_probability=control.probability)
        row.update(control[fields].add_prefix("control_").to_dict())
    return row


def child_age_pairs(frame):
    """Найти пары детей одного бронирования, различающиеся только возрастом."""
    zero_bills = frame.bills.eq("complete_zero")
    eligible = frame.cryo.eq("awake") & zero_bills & frame.Age.le(12)
    keys = ["group", "fold"] + [col for col in RAW_FEATURES if col != "Age"]
    children = frame.loc[
        eligible, [*keys, "Age", "y_true", "probability"]
    ].reset_index()
    pairs = children.merge(children, on=keys, suffixes=("_younger", "_older"))
    return pairs.loc[pairs.Age_younger.lt(pairs.Age_older)].sort_values(
        ["PassengerId_younger", "PassengerId_older"]
    )


def run(dev, split, reference, output):
    """Сравнить группы пассажиров и подобрать примеры ошибок."""
    frame = dev.set_index("PassengerId").join(
        reference.set_index("PassengerId"), validate="one_to_one"
    ).sort_index()
    fold = split.set_index("PassengerId").loc[frame.index, "fold_42"]
    if not frame.fold.eq(fold).all() or not frame.y_true.eq(
        frame.Transported
    ).all():
        raise ValueError("Метки или фолды расходятся с разбиением ДЗ2")
    # Одинаковые пропуски тоже входят в определение совпавшего профиля.
    profiles = frame.groupby(RAW_FEATURES, dropna=False)
    contradictory = profiles.y_true.transform("nunique").gt(1)
    columns = [*RAW_FEATURES, "group", "fold", "y_true", "probability"]
    frame.loc[contradictory, columns].sort_values(RAW_FEATURES).to_csv(
        output / "identical_inputs.csv", float_format="%.12g"
    )
    frame["deck"] = frame.Cabin.str.extract(
        r"^([^/]+)/[0-9]+/[^/]+$", expand=False
    ).fillna("missing")
    total = frame[SPENDING].sum(axis=1, min_count=len(SPENDING))
    frame["bills"] = "complete_positive"
    frame.loc[total.eq(0), "bills"] = "complete_zero"
    frame.loc[total.isna(), "bills"] = "missing"
    cryo = frame.CryoSleep.map({True: "asleep", False: "awake"})
    frame["cryo"] = cryo.fillna("missing")
    frame["origin"] = frame.HomePlanet.fillna("missing")

    slices = [slice_metrics(frame, "overall", "all")]
    for axis in ("deck", "bills", "cryo"):
        for level, subset in frame.groupby(axis):
            slices.append(slice_metrics(subset, axis, level))

    awake = frame.cryo.eq("awake")
    sleeping_on_g = frame.cryo.eq("asleep") & frame.deck.eq("G")
    spa_vr = frame.Spa + frame.VRDeck
    patterns = (
        ("sleep_zero_g", sleeping_on_g & total.eq(0), 1),
        ("awake_zero", awake & total.eq(0), 0),
        ("awake_spa_vr", awake & total.gt(0) & spa_vr.gt(total / 2), 0),
        ("awake_missing_bills", awake & total.isna(), 1),
    )
    examples = [
        error_example(frame.loc[mask], name, predicted)
        for name, mask, predicted in patterns
    ]
    compared_decks = frame.loc[frame.deck.isin(["B", "F", "G"])]
    origins = pd.crosstab(compared_decks.deck, compared_decks.origin)
    earth = compared_decks.loc[compared_decks.origin.eq("Earth")]
    cryo_counts = pd.crosstab(earth.deck, earth.cryo).add_prefix("Earth_")
    origins.join(cryo_counts).fillna(0).astype(int).to_csv(
        output / "deck_origins.csv"
    )
    for name, rows in (("slices", slices), ("errors", examples),
                       ("child_age_pairs", child_age_pairs(frame)),
                       ("deck_comparisons", deck_comparisons(frame))):
        pd.DataFrame(rows).to_csv(
            output / f"{name}.csv", index=False, float_format="%.12g"
        )
