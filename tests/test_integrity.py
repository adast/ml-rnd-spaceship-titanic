import copy
import json

import numpy as np
import pandas as pd
import pytest

from spaceship_hw1.data import build_assignments, freeze_splits, read_config
from spaceship_hw1.experiment import (
    audit_saved_metrics,
    run_experiment,
    score_predictions,
)
from spaceship_hw1.features import PassengerFeatures, make_model


@pytest.fixture
def config():
    return read_config("configs/baseline.yaml")


@pytest.fixture
def passengers():
    n = 120
    return pd.DataFrame(
        {
            "PassengerId": [
                f"{i // 2 + 1:04d}_{i % 2 + 1:02d}" for i in range(n)
            ],
            "HomePlanet": np.resize(["Earth", "Europa", "Mars"], n),
            "CryoSleep": np.resize([True, False], n),
            "Cabin": [f"B/{i}/P" for i in range(n)],
            "Destination": "TRAPPIST-1e",
            "Age": np.arange(n) % 70,
            "VIP": False,
            "RoomService": np.arange(n) * 3,
            "FoodCourt": 2.0,
            "ShoppingMall": 0.0,
            "Spa": np.nan,
            "VRDeck": 1.0,
            "Name": "Synthetic test fixture",
            "Transported": np.arange(n) % 2 == 0,
        }
    )


def test_target_is_rejected(passengers):
    with pytest.raises(ValueError, match="target"):
        PassengerFeatures().fit(passengers)


def test_features_are_row_local(passengers):
    X = passengers.drop(columns="Transported")
    feature = PassengerFeatures(missing_indicators=True, log_spend=True).fit(X)
    alone = feature.transform(X.iloc[[0]])
    together = feature.transform(X).iloc[[0]]
    pd.testing.assert_frame_equal(alone, together)
    assert "PassengerId" not in together and "Name" not in together


def test_imputation_does_not_learn_validation_distribution(passengers, config):
    X = passengers.drop(columns="Transported").iloc[:20].copy()
    X.loc[X.index[:3], "Age"] = np.nan
    model = make_model("logistic_cabin", config).fit(
        X, passengers["Transported"].iloc[:20].astype(int)
    )
    imputer = (
        model.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
    )
    before = imputer.statistics_.copy()
    assert before[0] == X["Age"].median()
    changed = X.copy()
    changed["Age"] = 1_000_000.0
    model.predict_proba(changed)
    np.testing.assert_array_equal(before, imputer.statistics_)


def test_unseen_category_and_empty_training_column(passengers, config):
    X = passengers.drop(columns="Transported")
    model = make_model("logistic_missing", config).fit(
        X, passengers["Transported"].astype(int)
    )
    test = X.iloc[:2].copy()
    test["HomePlanet"] = "Unseen planet"
    test["Cabin"] = "Z/99/X"
    probability = model.predict_proba(test)
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1)


def test_negative_spending_rejected(passengers):
    X = passengers.drop(columns="Transported").copy()
    X.loc[0, "Spa"] = -1.0
    with pytest.raises(ValueError, match="Negative spending"):
        PassengerFeatures(log_spend=True).fit_transform(X)


def test_holdout_is_group_disjoint_and_label_independent(passengers, config):
    first = build_assignments(passengers, config)
    dev = first["role"].eq("development")
    assert not set(first.loc[dev, "group"]) & set(first.loc[~dev, "group"])
    changed = passengers.copy()
    changed.loc[~dev, "Transported"] = ~changed.loc[~dev, "Transported"]
    pd.testing.assert_frame_equal(first, build_assignments(changed, config))
    for seed in config["split"]["seeds"]:
        for fold in range(config["split"]["n_splits"]):
            val = dev & first[f"fold_{seed}"].eq(fold)
            train = dev & ~first[f"fold_{seed}"].eq(fold)
            assert not set(first.loc[val, "group"]) & set(
                first.loc[train, "group"]
            )


def test_frozen_split_detects_tampering(tmp_path, passengers, config):
    (tmp_path / "data/raw").mkdir(parents=True)
    passengers.to_csv(tmp_path / config["data"]["train"], index=False)
    freeze_splits(passengers, config, tmp_path)
    assignments = tmp_path / "data/splits/assignments.csv"
    assignments.write_text(assignments.read_text() + "\n")
    with pytest.raises(ValueError, match="modified"):
        freeze_splits(passengers, config, tmp_path)


def test_accuracy_and_probability_metrics_hand_checked():
    metrics = score_predictions([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
    assert metrics["accuracy"] == 0.75
    assert metrics["f1"] == pytest.approx(0.8)
    expected_loss = (
        -(np.log(0.9) + np.log(0.2) + np.log(0.7) + np.log(0.9)) / 4
    )
    assert metrics["log_loss"] == pytest.approx(expected_loss)


def test_end_to_end_excludes_holdout(tmp_path, passengers, config):
    import yaml

    config = copy.deepcopy(config)
    config["data"]["expected_rows"] = len(passengers)
    config["split"]["seeds"] = [42]
    config["split"]["n_splits"] = 2
    config["evaluation"]["bootstrap_repetitions"] = 20
    config["evaluation"]["random_split_diagnostic"] = False
    (tmp_path / "data/raw").mkdir(parents=True)
    passengers.to_csv(tmp_path / config["data"]["train"], index=False)
    cfg = tmp_path / "baseline.yaml"
    cfg.write_text(yaml.safe_dump(config))
    _, _, assignment, _, predictions = run_experiment(tmp_path, cfg)
    held_ids = set(
        assignment.loc[assignment["role"].eq("final_holdout"), "PassengerId"]
    )
    assert not held_ids & set(predictions["PassengerId"])
    assert not json.loads((tmp_path / "reports/summary.json").read_text())[
        "holdout_evaluated"
    ]
    assert len(predictions) == (len(passengers) - len(held_ids)) * len(
        config["models"]
    )
    assert audit_saved_metrics(tmp_path / "reports")["passed"]
    metrics_path = tmp_path / "reports/fold_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics.loc[0, "accuracy"] += 0.1
    metrics.to_csv(metrics_path, index=False)
    with pytest.raises(AssertionError):
        audit_saved_metrics(tmp_path / "reports")
