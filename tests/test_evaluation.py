"""Tests for ``src.evaluation`` with values that can be checked by hand."""

import numpy as np
import pandas as pd
import pytest

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
