"""Row-local features and train-fold-only learned preprocessing."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

SPENDING = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC = ["Age", *SPENDING]
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
RAW_FEATURES = [*NUMERIC, *CATEGORICAL, "Cabin"]
VARIANTS = {
    "dummy_prior": (False, False, False),
    "logistic_core": (False, False, False),
    "logistic_cabin": (True, False, False),
    "logistic_missing": (True, True, False),
    "logistic_log_spend": (True, True, True),
}


class PassengerFeatures(TransformerMixin, BaseEstimator):
    """Вычислять признаки строки без статистик других наблюдений."""

    def __init__(self, cabin=True, missing_indicators=False, log_spend=False):
        self.cabin = cabin
        self.missing_indicators = missing_indicators
        self.log_spend = log_spend

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Expected a pandas DataFrame with named raw fields"
            )
        if "Transported" in X:
            raise ValueError("Remove the target Transported from model inputs")
        missing = sorted(set(RAW_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required predictor columns: {missing}")

    def fit(self, X, y=None):
        self._validate(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "fitted_")
        self._validate(X)
        result = pd.DataFrame(index=X.index)
        for name in NUMERIC:
            values = pd.to_numeric(X[name], errors="raise").astype(float)
            if np.isinf(values.to_numpy()).any():
                raise ValueError(f"Infinite predictor in {name}")
            if name in SPENDING and (values.dropna() < 0).any():
                raise ValueError(f"Negative spending in {name}")
            result[name] = (
                np.log1p(values)
                if self.log_spend and name in SPENDING
                else values
            )
            if self.missing_indicators:
                result[name + "_missing"] = values.isna().astype(float)
        for name in CATEGORICAL:
            values = X[name].astype("string")
            if name in ["CryoSleep", "VIP"]:
                values = values.str.lower().map(
                    {"true": "True", "false": "False"}
                )
            result[name] = values.astype(object).where(values.notna(), np.nan)
        if self.cabin:
            parts = (
                X["Cabin"]
                .astype("string")
                .str.extract(r"^([^/]+)/([0-9]+)/([^/]+)$")
            )
            for index, name in [(0, "CabinDeck"), (2, "CabinSide")]:
                result[name] = (
                    parts[index]
                    .astype(object)
                    .where(parts[index].notna(), np.nan)
                )
        return result


def make_model(name, config):
    if name not in VARIANTS:
        raise ValueError(f"Unknown model {name}")
    cabin, indicators, log_spend = VARIANTS[name]
    numeric = NUMERIC + (
        [x + "_missing" for x in NUMERIC] if indicators else []
    )
    categorical = CATEGORICAL + (["CabinDeck", "CabinSide"] if cabin else [])
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__",
                                 keep_empty_features=True)),
        ("encode", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop", sparse_threshold=0)
    params = {k: v for k, v in config["logistic"].items() if k != "threshold"}
    classifier = (
        DummyClassifier(strategy="prior")
        if name == "dummy_prior"
        else LogisticRegression(**params)
    )
    return Pipeline(
        [
            ("features", PassengerFeatures(cabin, indicators, log_spend)),
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )
