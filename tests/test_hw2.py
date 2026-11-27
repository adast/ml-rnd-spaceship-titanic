import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from spaceship_hw1.data import (
    build_assignments,
    freeze_splits,
    read_config,
    read_training,
    sha256,
)
from spaceship_hw1.features import (
    CATEGORICAL,
    NUMERIC,
    RAW_FEATURES,
    SPENDING,
    make_model,
)
from spaceship_hw2 import (
    MODELS,
    compare_historical_b1,
    load_development,
    make_hw2_model,
    spending_features,
)

ROOT = Path(__file__).resolve().parents[1]


def test_factorial_changes_preserve_hw1_inputs_and_fold_preprocessing():
    baseline = read_config(ROOT / "configs/baseline.yaml")
    boosting = yaml.safe_load((ROOT / "configs/hw2.yaml").read_text())[
        "boosting"
    ]
    X = pd.DataFrame({
        **{k: [1.0, 12.0, 8.0, 4.0] for k in NUMERIC},
        **{k: ["True"] * 4 for k in CATEGORICAL},
        "Cabin": ["B/001/P", "C/9/S", None, "A/14/P/extra"],
    })
    probe = X.copy()
    probe["Cabin"] = [None, "bad", "A/7/P/extra", "Z/999/X"]
    hw1 = make_model("logistic_cabin", baseline)
    expected = hw1.named_steps["features"].fit_transform(X)
    for name in MODELS:
        model = make_hw2_model(name, baseline, boosting)
        classifier = model.named_steps["classifier"]
        if name in ("M", "MC"):
            assert isinstance(classifier, HistGradientBoostingClassifier)
            for key, value in boosting.items():
                assert classifier.get_params()[key] == value
        else:
            assert type(classifier) is type(hw1.named_steps["classifier"])
            assert classifier.get_params() == hw1["classifier"].get_params()

        recipe = model.named_steps["features"]
        raw = recipe.fit_transform(X)
        pd.testing.assert_frame_equal(raw[expected.columns], expected)
        if name in ("C", "MC"):
            assert list(raw.columns) == [*expected.columns, "CabinNumber"]
            np.testing.assert_allclose(
                raw["CabinNumber"], [1, 9, np.nan, np.nan], equal_nan=True
            )
            other = recipe.transform(probe)
            np.testing.assert_allclose(
                other["CabinNumber"], [np.nan, np.nan, np.nan, 999],
                equal_nan=True,
            )
            preprocess = model["preprocess"].fit(raw)
            numeric = preprocess.named_transformers_["numeric"]
            assert numeric["impute"].statistics_[-1] == 5
            preprocess.transform(other)
            assert numeric["impute"].statistics_[-1] == 5
        else:
            pd.testing.assert_frame_equal(raw, expected)


def test_spending_features_use_only_complete_bills_from_the_same_passenger():
    rows = pd.DataFrame(
        [[0, 0, 0, 0, 0], [1, 2, 3, 4, 0],
         [1, np.nan, 3, 4, 0], [1, 2, 3, np.nan, 0]],
        columns=SPENDING, index=[7, 2, 1, 9],
    )
    result = spending_features(rows)
    np.testing.assert_allclose(result.TotalSpend, [0, 10, np.nan, np.nan])
    np.testing.assert_allclose(result.SpendShare, [0, 0.5, np.nan, np.nan])
    pd.testing.assert_frame_equal(
        spending_features(rows.loc[[2]]), result.loc[[2]]
    )


def test_missing_frozen_manifest_is_never_generated(tmp_path):
    with pytest.raises(FileNotFoundError, match="Frozen HW1 manifest"):
        load_development(tmp_path, {})
    assert not (tmp_path / "data/splits").exists()


def test_development_loader_skips_holdout_labels_in_shuffled_csv(tmp_path):
    baseline = read_config(ROOT / "configs/baseline.yaml")
    baseline["data"]["expected_rows"] = 80
    passengers = pd.DataFrame({
        **dict.fromkeys(RAW_FEATURES, 1),
        "PassengerId": [f"{i:04d}_01" for i in range(80)],
        "Transported": np.arange(80) % 2 == 0,
    })
    assignments = build_assignments(passengers, baseline)
    held = assignments["role"].eq("final_holdout")
    csv = tmp_path / baseline["data"]["train"]
    csv.parent.mkdir(parents=True)
    labels = passengers["Transported"].astype(str).where(~held, "sealed")
    passengers.assign(Transported=labels).sample(
        frac=1, random_state=17
    ).to_csv(csv, index=False)
    freeze_splits(passengers, baseline, tmp_path)
    provenance = {"files": {"train.csv": {"sha256": sha256(csv)}}}
    (tmp_path / "data/PROVENANCE.json").write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="Target must be complete Boolean"):
        read_training(csv)
    dev, split, _ = load_development(tmp_path, baseline)
    assert dev["PassengerId"].equals(split["PassengerId"])
    assert dev["PassengerId"].tolist() == passengers.loc[
        ~held, "PassengerId"
    ].tolist()
    assert dev["Transported"].tolist() == passengers.loc[
        ~held, "Transported"
    ].astype(int).tolist()


def test_historical_check_joins_by_seed_and_id_and_detects_label_drift(
    tmp_path,
):
    fresh = pd.DataFrame({
        "PassengerId": ["0008_01", "0009_02"], "group": ["0008", "0009"],
        "seed": 42, "fold": [1, 0], "model": "B1", "y_true": [1, 0],
        "probability": [0.67, 0.18], "y_pred": [1, 0],
    }).astype({"PassengerId": "string", "group": "string"})
    history = fresh.assign(protocol="grouped", model="logistic_cabin")
    path = tmp_path / "history.csv"
    history.iloc[::-1].to_csv(path, index=False)
    assert compare_historical_b1(fresh, path)["prediction_rows"] == 2
    fresh.loc[0, "y_pred"] = 0
    with pytest.raises(AssertionError, match="y_pred"):
        compare_historical_b1(fresh, path)
