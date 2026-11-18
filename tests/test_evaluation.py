"""Tests for ``src.evaluation`` with values that can be checked by hand."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from src import evaluation


def test_perfect_prediction_metrics():
    y = np.array([6.0, 7.0, 8.0])
    metrics = evaluation.regression_metrics(y, y)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["mae_log"] == 0
    assert metrics["mae_eur"] == 0
    assert metrics["within_factor_2"] == 100


def test_predicting_the_mean_gives_r2_zero():
    y = np.array([6.0, 7.0, 8.0])
    assert evaluation.regression_metrics(y, np.full(3, 7.0))["r2"] == pytest.approx(0.0)


def test_metrics_in_euros():
    # Real 1 M€ and 10 M€; predicted 2 M€ and 5 M€ (both off by a factor of 2)
    y_true = np.log10([1e6, 1e7])
    y_pred = np.log10([2e6, 5e6])
    metrics = evaluation.regression_metrics(y_true, y_pred)
    assert metrics["mae_eur"] == pytest.approx((1e6 + 5e6) / 2)
    assert metrics["mae_log"] == pytest.approx(np.log10(2))
    assert metrics["median_abs_pct_error"] == pytest.approx(75.0)  # 100 % and 50 %
    assert metrics["within_factor_2"] == 100


def test_error_by_value_range_sign_shows_under_and_overestimation():
    y_true = np.log10([5e5, 1e8])
    y_pred = np.log10([1e6, 5e7])  # cheap player overestimated, star underestimated
    table = evaluation.error_by_value_range(y_true, y_pred)
    assert table.loc["< 1 M€", "mean_error_log"] > 0
    assert table.loc["≥ 50 M€", "mean_error_log"] < 0
    assert table["players"].sum() == 2


def test_value_gaps_sorted_by_ratio():
    table = pd.DataFrame({
        "name": ["A", "B"], "season": [2024, 2024], "position": ["Attack", "Defender"],
        "league": ["GB1", "ES1"], "age": [20.0, 30.0], "market_value_in_eur": [1e6, 1e6],
    })
    gaps = evaluation.value_gaps(table, np.log10([2e6, 8e6]))
    assert gaps["name"].tolist() == ["B", "A"]
    assert gaps.loc[0, "ratio"] == pytest.approx(8)


def test_plot_is_saved(tmp_path):
    path = evaluation.plot_predicted_vs_real([6.0, 7.0], [6.1, 6.9], tmp_path / "plot.png")
    assert path.is_file()


class AgeOnlyModel(RegressorMixin, BaseEstimator):
    """Fake fitted model whose prediction is ``age / 10`` (ignores the rest)."""

    def fit(self, X, y):
        """Nothing to learn."""
        return self

    def predict(self, X):
        """Return ``age / 10`` for every row."""
        return X["age"].to_numpy() / 10


def make_age_table():
    """Small table where only ``age`` explains the target of ``AgeOnlyModel``."""
    X = pd.DataFrame({"age": [20.0, 25.0, 30.0, 35.0], "minutes": [900, 1500, 2000, 3000]})
    return X, X["age"] / 10


def test_feature_importance_finds_the_only_useful_column():
    X, y = make_age_table()
    importance = evaluation.feature_importance(AgeOnlyModel(), X, y)
    assert importance.index[0] == "age"
    assert importance.loc["age", "importance"] > 0
    assert importance.loc["minutes", "importance"] == 0


def test_age_curve_changes_only_age():
    X, _ = make_age_table()
    curve = evaluation.age_curve(AgeOnlyModel(), X, ages=[20, 30])
    assert curve["age"].tolist() == [20, 30]
    assert curve["predicted_eur"].tolist() == pytest.approx([100.0, 1000.0])  # 10**2, 10**3
    assert X["age"].tolist() == [20.0, 25.0, 30.0, 35.0]  # input not modified


def test_importance_and_age_plots_are_saved(tmp_path):
    X, y = make_age_table()
    importance = evaluation.feature_importance(AgeOnlyModel(), X, y)
    curve = evaluation.age_curve(AgeOnlyModel(), X, ages=[20, 30])
    assert evaluation.plot_feature_importance(importance, tmp_path / "imp.png").is_file()
    assert evaluation.plot_age_curves({"fake": curve}, tmp_path / "age.png").is_file()
