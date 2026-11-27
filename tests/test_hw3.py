from pathlib import Path

import numpy as np
import pandas as pd

from spaceship_hw1.features import RAW_FEATURES

ROOT = Path(__file__).resolve().parents[1]


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
