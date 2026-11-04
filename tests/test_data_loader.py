"""Tests for src.data_loader, using small fixture CSV files."""

from pathlib import Path

import pandas as pd
import pytest

from src import data_loader

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_players_returns_expected_columns():
    players = data_loader.load_players(FIXTURES)
    assert list(players.columns) == data_loader.PLAYER_COLUMNS
    assert len(players) == 3


def test_load_players_never_loads_leakage_columns():
    players = data_loader.load_players(FIXTURES)
    assert "market_value_in_eur" not in players.columns
    assert "highest_market_value_in_eur" not in players.columns


def test_load_players_strips_whitespace():
    players = data_loader.load_players(FIXTURES)
    assert "Roman Weidenfeller" in players["name"].values


def test_load_players_parses_dates_and_invalid_becomes_nat():
    players = data_loader.load_players(FIXTURES).set_index("player_id")
    assert players.loc[10, "date_of_birth"] == pd.Timestamp("1978-06-09")
    assert pd.isna(players.loc[99, "date_of_birth"])


def test_load_players_keeps_missing_contract_as_nat():
    players = data_loader.load_players(FIXTURES).set_index("player_id")
    assert pd.isna(players.loc[10, "contract_expiration_date"])
    assert players.loc[26, "contract_expiration_date"] == pd.Timestamp("2018-06-30")


def test_load_valuations_has_league_at_valuation_time():
    valuations = data_loader.load_valuations(FIXTURES)
    assert valuations["date"].dtype.kind == "M"
    assert set(valuations["player_club_domestic_competition_id"]) == {"IT1", "L1"}


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the dataset"):
        data_loader.load_players(tmp_path)


def test_missing_columns_raise_value_error(tmp_path):
    (tmp_path / "players.csv").write_text('"player_id","name"\n"1","A"\n')
    with pytest.raises(ValueError, match="missing columns"):
        data_loader.load_players(tmp_path)
