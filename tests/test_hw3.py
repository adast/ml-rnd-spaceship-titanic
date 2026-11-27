from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from spaceship_hw1.features import RAW_FEATURES, SPENDING

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def robustness(monkeypatch):
    monkeypatch.syspath_prepend(ROOT / "scripts")
    return import_module("hw3_robustness")


def test_lost_bill_and_selection_order_follow_the_passenger(robustness):
    frame = pd.DataFrame({
        "PassengerId": [f"{i:04d}_01" for i in range(12)],
        **{col: np.arange(12, dtype=float) for col in SPENDING},
    })
    frame.loc[1, SPENDING[0]] = np.nan
    frame.loc[3, SPENDING] = np.nan
    original = frame.copy()

    def mask(rows):
        damaged, rank, complete = robustness.bill_loss(rows)
        removed = rows[SPENDING].notna() & damaged[SPENDING].isna()
        assert removed.sum(axis=1).eq(complete.astype(int)).all()
        pd.testing.assert_frame_equal(damaged, rows.mask(removed.reindex(
            columns=rows.columns, fill_value=False
        )))
        return damaged.assign(rank=rank).set_index("PassengerId").sort_index()

    first = mask(frame)
    shuffled = mask(frame.sample(frac=1, random_state=3, ignore_index=True))
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(first, shuffled)
    complete = original.set_index("PassengerId")[SPENDING].notna().all(axis=1)
    assert sorted(first.loc[complete, "rank"]) == list(range(complete.sum()))
    assert first.loc[~complete, "rank"].ge(complete.sum()).all()


def test_nested_training_groups_and_fixed_validation(
    tmp_path, monkeypatch, robustness,
):
    groups = pd.Series(np.repeat(np.arange(20), 2))
    dev = pd.DataFrame({
        "PassengerId": [f"{i // 2:04d}_{i % 2 + 1:02d}" for i in range(40)],
        "Transported": 0, **dict.fromkeys(SPENDING, 1.0),
    })
    split = pd.DataFrame({"group": groups, "fold_42": groups % 2})
    reference = dev[["PassengerId"]].assign(probability=0.2)
    config = {"split": {"n_splits": 2, "primary_seed": 42},
              "logistic": {"threshold": 0.5}, "runtime": {"blas_threads": 1}}
    model = Mock()
    model.predict_proba.side_effect = lambda x: np.tile(
        [0.8, 0.2], (len(x), 1)
    )
    monkeypatch.setattr(robustness, "make_hw2_model", lambda *args: model)
    robustness.run(dev, split, config, {}, reference, tmp_path)

    for call in model.predict_proba.call_args_list:
        ids = call.args[0].index
        fold = split.fold_42.loc[ids[0]]
        assert set(ids) == set(split.index[split.fold_42.eq(fold)])

    details = pd.read_csv(tmp_path / "robustness_folds.csv")
    fits = details.loc[details["check"].eq("training_groups")].itertuples()
    subsets = {}
    for row, call in zip(fits, model.fit.call_args_list, strict=True):
        ids = call.args[0].index
        actual = set(groups.loc[ids])
        assert set(ids) == set(groups.index[groups.isin(actual)])
        train = set(groups[split.fold_42.ne(row.fold)])
        assert actual <= train
        assert len(actual) == round(len(train) * row.percent / 100)
        subsets[row.fold, row.subset_seed, row.percent] = actual
    for (fold, seed, percent), small in subsets.items():
        if percent == 20:
            assert small < subsets[fold, seed, 50]


def test_expected_loss_keeps_five_outcomes_of_each_passenger_together(
    robustness,
):
    # В первой группе к ошибкам ведёт пропуск Spa, во второй пропуск FoodCourt.
    groups = pd.Series([10, 10, 20, 20])
    truth = pd.Series([0, 0, 1, 1])
    features = pd.DataFrame(1.0, index=groups.index, columns=SPENDING)
    complete = features.notna().all(axis=1)
    changes = pd.DataFrame(0, index=groups.index, columns=SPENDING)
    changes.loc[groups.eq(10), "Spa"] = -1
    changes.loc[groups.eq(20), "FoodCourt"] = -1
    results = robustness.service_summary(
        changes, features, complete, groups, groups, complete, truth
    ).set_index("service")
    assert results.loc["Spa", "new_fp"] == 2
    assert results.loc["FoodCourt", "new_fn"] == 2
    assert results.loc["Spa", "ci_low_pp"] < results.loc["Spa", "ci_high_pp"]
    uniform = results.loc["uniform"]
    assert uniform.n_validation == 4
    assert uniform.introduced == pytest.approx(0.8)
    np.testing.assert_allclose(
        uniform[["pooled_delta_pp", "ci_low_pp", "ci_high_pp"]], -20
    )


def test_uniform_auc_compares_independent_services_within_folds(robustness):
    # По одной паре разных классов на фолд. Диапазоны двух фолдов разнесены.
    pair = np.array([[2, 2, 2, 2, 8], [1, 1, 1, 1, 8]]) / 50
    scores = pd.DataFrame(np.vstack([pair, 0.7 + pair]), columns=SPENDING)
    scores["clean"] = [0.2, 0.1, 0.9, 0.8]
    table = robustness.bill_scores(
        scores, pd.Series([1, 0, 1, 0]), pd.Series([0, 0, 1, 1]), 0.5
    ).set_index("service")
    # Из 25 пар услуг 20 дают правильный порядок и одна даёт равенство.
    assert table.loc["uniform", "mean_fold_auc"] == pytest.approx(20.5 / 25)
    assert table.loc[SPENDING, "mean_fold_auc"].mean() == pytest.approx(0.9)
    assert table.at["uniform", "best_threshold_accuracy"] == pytest.approx(0.9)
    assert table.loc["uniform", "n_validation"] == len(scores)


def test_best_threshold_keeps_equal_scores_together(robustness):
    scores = np.array([0.4, 0.4, 0.4, 0.8])
    for labels in ([0, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]):
        truth = np.array(labels)
        expected = max(((scores >= t) == truth).mean()
                       for t in [0.4, 0.8, np.inf])
        assert robustness.best_threshold_accuracy(truth, scores) == expected


def test_spa_case_uses_the_observed_training_amounts(robustness):
    case = pd.DataFrame([{
        "PassengerId": "0001_01", "HomePlanet": "Earth", "Cabin": "F/1/P",
        "CryoSleep": False, **dict.fromkeys(SPENDING, 0.0), "Spa": np.nan,
    }])
    training = pd.concat([case] * 6, ignore_index=True)
    training["Spa"] = [0, 0, 53, 100, 200, 300]
    training.loc[3, "Cabin"] = "G/1/P"
    training.loc[4, "FoodCourt"] = np.nan
    training.loc[5, "CryoSleep"] = True
    model = Mock()
    model.predict_proba.side_effect = lambda x: np.column_stack(
        [x.Spa.fillna(0).gt(50), x.Spa.fillna(0).le(50)]
    )
    result = robustness.spa_response(model, training, case)
    assert result.spa_amount.tolist() == [0, 53]
    assert result.n_donors.tolist() == [2, 1]
    assert result.probability.tolist() == [1, 0]
    assert case.Spa.isna().all()


def test_error_counts_and_paired_changes_match_archived_predictions():
    archive = pd.read_csv(ROOT / "reports/hw2/spending_predictions.csv.gz")
    primary = archive.loc[archive.protocol.eq("group_42")]
    mc = primary.MC.ge(0.5)
    correct = mc.eq(primary.y_true)
    share_correct = primary.share.ge(0.5).eq(primary.y_true)
    expected = {
        "passengers": len(primary), "positives": primary.y_true.sum(),
        "fp": (mc & ~correct).sum(), "fn": (~mc & ~correct).sum(),
        "share_fixed": (~correct & share_correct).sum(),
        "share_introduced": (correct & ~share_correct).sum(),
    }
    slices = pd.read_csv(ROOT / "reports/hw3/slices.csv")
    assert set(slices.axis) == {"overall", "cryo", "deck", "bills"}
    for _, rows in slices.groupby("axis"):
        assert rows[list(expected)].sum().to_dict() == expected
    old = pd.read_csv(ROOT / "reports/hw2/cabin_errors.csv")
    by_deck = slices.loc[slices.axis.eq("deck")].set_index("level")
    by_deck = by_deck[["passengers", "positives"]].sort_index()
    old = old.loc[old.seed.eq(42)].set_index("deck").sort_index()
    assert old.index.tolist() == by_deck.index.tolist()
    np.testing.assert_array_equal(by_deck, old[["n_passengers", "n_positive"]])


def test_same_inputs_with_opposite_labels_against_archived_predictions():
    rows = pd.read_csv(ROOT / "reports/hw3/identical_inputs.csv")
    profiles = rows.groupby(RAW_FEATURES, dropna=False)
    assert profiles.ngroups > 0
    assert profiles.y_true.nunique().eq(2).all()
    # Внутри фолда пассажиры с одним профилем получают один прогноз.
    by_fold = rows.groupby([*RAW_FEATURES, "fold"], dropna=False)
    assert by_fold.probability.nunique().eq(1).all()
    archive = pd.read_csv(ROOT / "reports/hw2/spending_predictions.csv.gz")
    primary = archive.loc[archive.protocol.eq("group_42")].set_index(
        "PassengerId"
    ).loc[rows.PassengerId]
    np.testing.assert_array_equal(rows.y_true, primary.y_true)
    np.testing.assert_array_equal(rows.fold, primary.fold)
    np.testing.assert_allclose(
        rows.probability, primary.MC, rtol=0, atol=1e-11
    )
