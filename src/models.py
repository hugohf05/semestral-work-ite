"""Baselines, linear regression and Random Forest for market value prediction.

All models predict ``log_value`` (log10 of the value in euros) and expose the
scikit-learn interface ``fit(X, y)`` / ``predict(X)``, so they can be compared
with the same code.
"""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FEATURE_COLUMNS, TARGET_COLUMN

CATEGORICAL = ["position", "league"]
NUMERIC = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL]

LAST_TRAIN_SEASON = 2023
VALIDATION_SEASON = 2024
TEST_SEASON = 2025


def time_split(table):
    """Split a player-season table by time, never at random.

    Train: seasons up to ``LAST_TRAIN_SEASON``; validation: ``VALIDATION_SEASON``;
    test: ``TEST_SEASON`` (the last complete season).

    :param table: Output of ``features.build_player_seasons``
    :return: Tuple ``(train, validation, test)`` of DataFrames
    """
    train = table[table["season"] <= LAST_TRAIN_SEASON]
    validation = table[table["season"] == VALIDATION_SEASON]
    test = table[table["season"] == TEST_SEASON]
    return train, validation, test


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Predict the median target of the player's group.

    Groups are defined by ``group_columns``. A group never seen in training
    falls back to the global median. With no group columns this is the
    "always predict the global median" baseline.
    """

    def __init__(self, group_columns=()):
        """:param group_columns: Columns that define a group, e.g. position and league."""
        self.group_columns = group_columns

    def fit(self, X, y):
        """Store the median of ``y`` for every group.

        :param X: Feature DataFrame containing ``group_columns``
        :param y: Target values (log value)
        :return: ``self``
        """
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_median_ = float(y.median())
        columns = list(self.group_columns)
        # e.g. {("Attack", "GB1"): 7.1, ("Defender", "ES1"): 6.4, ...}
        self.medians_ = y.groupby([X[c] for c in columns]).median().to_dict() if columns else {}
        return self

    def predict(self, X):
        """Look up the group median of every row.

        :param X: Feature DataFrame containing ``group_columns``
        :return: Array of predictions
        """
        columns = list(self.group_columns)
        if not columns:
            return np.full(len(X), self.global_median_)
        # One tuple per row, e.g. ("Attack", "GB1"); a single column gives 1-tuples
        keys = X[columns].itertuples(index=False, name=None)
        return np.array([self.medians_.get(key if len(columns) > 1 else key[0],
                                           self.global_median_) for key in keys])


def _preprocessor(numeric_columns, scale):
    """One-hot encode the categorical columns; optionally scale the numeric ones.

    :param numeric_columns: Names of the numeric columns to keep
    :param scale: Standardise the numeric columns (needed by linear regression only)
    :return: Unfitted ``ColumnTransformer``
    """
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        # One-hot: "league" becomes 0/1 columns (league_GB1, league_ES1, ...);
        # an unseen category becomes all zeros instead of an error
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", numeric, numeric_columns),
    ])


def make_linear_regression():
    """Linear regression with one-hot categories, scaled numbers and age squared.

    :return: Unfitted scikit-learn pipeline
    """
    return Pipeline([
        ("add_age_squared", _AgeSquared()),
        ("prepare", _preprocessor(NUMERIC + ["age_squared"], scale=True)),
        ("model", LinearRegression()),
    ])


def make_random_forest(**params):
    """Random Forest with one-hot categories (no scaling needed).

    :param params: Overrides for ``RandomForestRegressor`` hyperparameters
    :return: Unfitted scikit-learn pipeline
    """
    settings = {"n_estimators": 300, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42}
    settings.update(params)
    return Pipeline([
        ("prepare", _preprocessor(NUMERIC, scale=False)),
        ("model", RandomForestRegressor(**settings)),
    ])


class _AgeSquared(BaseEstimator):
    """Add ``age_squared`` so a straight-line model can learn a curved age effect."""

    def fit(self, X, y=None):
        """Nothing to learn.

        :return: ``self``
        """
        return self

    def transform(self, X):
        """Return a copy of ``X`` with the extra column.

        :param X: Feature DataFrame with ``age``
        :return: DataFrame with ``age_squared`` added
        """
        X = X.copy()
        X["age_squared"] = X["age"] ** 2
        return X


def split_xy(table):
    """Separate features and target.

    :param table: Player-season table
    :return: Tuple ``(X, y)``
    """
    return table[FEATURE_COLUMNS], table[TARGET_COLUMN]


PARAM_GRID = {
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", 0.5],
}


def tune_random_forest(train, validation, grid=PARAM_GRID, n_estimators=200):
    """Choose Random Forest hyperparameters with the validation season.

    Every combination of ``grid`` is trained on ``train`` and scored by MAE
    (log10 euros) on ``validation``; the test season is never used here.

    :param train: Training table
    :param validation: Validation table
    :param grid: Dict ``{hyperparameter: list of values to try}``
    :param n_estimators: Trees per forest during the search (fewer = faster)
    :return: Tuple ``(best_params, results)``; ``results`` is a DataFrame
        with one row per combination, best first
    """
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(validation)
    rows = []
    # product(...) yields every combination, e.g. (10, 1, "sqrt"), (10, 1, 0.5), ...
    for values in product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        model = make_random_forest(n_estimators=n_estimators, **params).fit(X_train, y_train)
        mae = np.abs(model.predict(X_val) - y_val).mean()
        rows.append({**params, "mae_log": mae})
    results = pd.DataFrame(rows).sort_values("mae_log").reset_index(drop=True)
    best = {key: results.loc[0, key] for key in grid}
    # pandas stores None as NaN in a mixed column; scikit-learn needs None back
    best = {key: (None if pd.isna(value) else value) for key, value in best.items()}
    # Convert numpy numbers to plain Python ints so they print and save cleanly
    for key in ("max_depth", "min_samples_leaf"):
        if best.get(key) is not None:
            best[key] = int(best[key])
    return best, results
