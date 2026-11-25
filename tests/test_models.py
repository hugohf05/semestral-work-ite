"""Tests for ``src.models`` using a small synthetic player-season table."""

import numpy as np
import pandas as pd
import pytest

from src import models
from src.features import FEATURE_COLUMNS, TARGET_COLUMN


def make_table(rows=200, seed=0):
    """Synthetic table where value depends on league and age (so models can learn it)."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({
        "age": rng.uniform(18, 36, rows),
        "season": rng.integers(2012, 2026, rows),
        "position": rng.choice(["Attack", "Defender", "Goalkeeper", "Midfield"], rows),
        "league": rng.choice(["GB1", "ES1"], rows),
        "minutes": rng.integers(450, 3400, rows),
        "matches": rng.integers(8, 38, rows),
        "goals_per90": rng.uniform(0, 1, rows),
        "assists_per90": rng.uniform(0, 0.5, rows),
        "yellow_cards_per90": rng.uniform(0, 0.3, rows),
        "red_cards": rng.integers(0, 2, rows),
        "team_points_per_game": rng.uniform(0.5, 2.5, rows),
        "european_minutes": rng.choice([0, 0, 360, 900], rows),
    })
    table[TARGET_COLUMN] = 6 + 0.5 * (table.league == "GB1") - 0.03 * (table.age - 18)
    return table


def test_time_split_has_no_overlap_and_is_ordered():
    train, validation, test = models.time_split(make_table())
    assert train.season.max() <= models.LAST_TRAIN_SEASON
    assert set(validation.season) == {models.VALIDATION_SEASON}
    assert set(test.season) == {models.TEST_SEASON}


def test_global_median_baseline_predicts_one_constant():
    table = make_table()
    X, y = table[FEATURE_COLUMNS], table[TARGET_COLUMN]
    prediction = models.MedianBaseline().fit(X, y).predict(X)
    assert prediction == pytest.approx(np.full(len(X), y.median()))


def test_group_baseline_predicts_group_median():
    X = pd.DataFrame({"position": ["A", "A", "B"], "league": ["x", "x", "x"]})
    y = pd.Series([1.0, 3.0, 10.0])
    model = models.MedianBaseline(["position", "league"]).fit(X, y)
    assert model.predict(X).tolist() == [2.0, 2.0, 10.0]


def test_group_baseline_unseen_group_falls_back_to_global_median():
    X = pd.DataFrame({"position": ["A", "B"], "league": ["x", "x"]})
    model = models.MedianBaseline(["position", "league"]).fit(X, pd.Series([1.0, 5.0]))
    unseen = pd.DataFrame({"position": ["C"], "league": ["y"]})
    assert model.predict(unseen)[0] == pytest.approx(3.0)


@pytest.mark.parametrize("factory", [models.make_linear_regression,
                                     lambda: models.make_random_forest(n_estimators=20)])
def test_models_fit_and_beat_global_median(factory):
    table = make_table()
    X, y = table[FEATURE_COLUMNS], table[TARGET_COLUMN]
    prediction = factory().fit(X, y).predict(X)
    baseline = models.MedianBaseline().fit(X, y).predict(X)
    assert len(prediction) == len(X)
    assert np.abs(prediction - y).mean() < np.abs(baseline - y).mean()


def test_models_handle_unseen_category():
    table = make_table()
    model = models.make_random_forest(n_estimators=10).fit(table[FEATURE_COLUMNS], table[TARGET_COLUMN])
    new = table.head(1).copy()
    new["league"] = "FR1"
    assert np.isfinite(model.predict(new[FEATURE_COLUMNS])).all()


def test_tune_random_forest_tries_every_combination():
    table = make_table(rows=300)
    train, validation, _ = models.time_split(table)
    grid = {"max_depth": [3, None], "min_samples_leaf": [1, 5], "max_features": ["sqrt"]}
    best, results = models.tune_random_forest(train, validation, grid, n_estimators=10)
    assert len(results) == 4
    assert results["mae_log"].is_monotonic_increasing  # best first
    assert set(best) == set(grid)
    assert best["max_depth"] in (3, None)  # NaN turned back into None
