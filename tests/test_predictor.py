"""Tests for ``src.predictor``: saving, loading and finding players."""

import numpy as np
import pandas as pd
import pytest

from src import predictor


def make_table():
    """Two seasons of 'Pedri' and one of 'Pedro', to test name matching."""
    return pd.DataFrame({
        "name": ["Pedri", "Pedri", "Pedro"],
        "season": [2023, 2024, 2024],
        "market_value_in_eur": [80e6, 100e6, 5e6],
    })


class ConstantModel:
    """Fake model that always predicts log10 value 7 (= 10 M€)."""

    def predict(self, X):
        """Return 7.0 for every row."""
        return np.full(len(X), 7.0)


def test_find_player_returns_latest_season_by_default():
    row = predictor.find_player(make_table(), "pedri")  # case-insensitive
    assert row["season"].item() == 2024


def test_find_player_with_season():
    assert predictor.find_player(make_table(), "Pedri", 2023)["market_value_in_eur"].item() == 80e6


def test_find_player_unknown_name_raises():
    with pytest.raises(ValueError, match="No player-season"):
        predictor.find_player(make_table(), "Messi")


def test_find_player_ambiguous_name_raises():
    with pytest.raises(ValueError, match="several players"):
        predictor.find_player(make_table(), "Ped")


def test_predict_value_converts_log_to_euros():
    rows = pd.DataFrame({column: [0] for column in predictor.FEATURE_COLUMNS})
    assert predictor.predict_value(ConstantModel(), rows)[0] == pytest.approx(10e6)


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "sub" / "model.joblib"
    predictor.save_model({"any": "object"}, {"max_depth": 10}, [2012, 2013], path)
    saved = predictor.load_model(path)
    assert saved["params"] == {"max_depth": 10}
    assert saved["seasons"] == [2012, 2013]


def test_load_missing_model_gives_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="python main.py train"):
        predictor.load_model(tmp_path / "missing.joblib")


def test_find_player_exact_name_beats_partial_matches():
    table = pd.DataFrame({"name": ["Pedri", "Pedrinho"], "season": [2024, 2024],
                          "market_value_in_eur": [100e6, 1e6]})
    assert predictor.find_player(table, "pedri")["name"].item() == "Pedri"


def test_undervalued_players_filters_and_sorts():
    table = pd.DataFrame({
        "name": ["Young cheap", "Veteran", "Too cheap", "Other season"],
        "season": [2025, 2025, 2025, 2024],
        "age": [22.0, 37.0, 22.0, 22.0],
        "market_value_in_eur": [2e6, 2e6, 1e5, 2e6],
        **{column: [0, 0, 0, 0] for column in predictor.FEATURE_COLUMNS
           if column not in ("season", "age")},
    })
    result = predictor.undervalued_players(table, ConstantModel(), season=2025)
    assert result["name"].tolist() == ["Young cheap"]  # veteran, too cheap and 2024 removed
    assert result.loc[0, "ratio"] == pytest.approx(5)  # 10 M€ predicted / 2 M€ real
