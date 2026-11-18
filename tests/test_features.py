"""Unit tests for ``src.features`` using small handmade tables."""

import pandas as pd
import pytest

from src import features


def make_games():
    """Three matches of season 2023 (two Premier League, one LaLiga) and one cup match."""
    return pd.DataFrame({
        "game_id": [1, 2, 3, 4],
        "season": [2023, 2023, 2023, 2023],
    })


def make_appearances():
    """Player 7 plays 2 Premier League matches and 1 LaLiga match; player 8 plays only a cup match."""
    return pd.DataFrame({
        "game_id": [1, 2, 3, 4],
        "player_id": [7, 7, 7, 8],
        "competition_id": ["GB1", "GB1", "ES1", "CL"],
        "minutes_played": [90, 90, 45, 90],
        "goals": [1, 0, 1, 3],
        "assists": [0, 1, 0, 0],
        "yellow_cards": [1, 0, 0, 0],
        "red_cards": [0, 0, 0, 0],
    })


def make_valuations():
    """Player 7: one valuation out of the window and two inside (the later one wins)."""
    return pd.DataFrame({
        "player_id": [7, 7, 7],
        "date": pd.to_datetime(["2024-01-10", "2024-04-15", "2024-06-01"]),
        "market_value_in_eur": [1_000_000, 2_000_000, 10_000_000],
    })


def make_players():
    """Profiles for player 7 and 8."""
    return pd.DataFrame({
        "player_id": [7, 8],
        "name": ["Test Player", "Cup Only"],
        "date_of_birth": pd.to_datetime(["2000-06-30", "1995-01-01"]),
        "position": ["Attack", "Defender"],
    })


def test_aggregate_seasons_sums_only_league_matches():
    seasons = features.aggregate_seasons(make_appearances(), make_games())
    assert len(seasons) == 1  # player 8 only played a cup match
    row = seasons.iloc[0]
    assert (row.minutes, row.matches, row.goals, row.assists) == (225, 3, 2, 1)


def test_aggregate_seasons_picks_league_with_most_minutes():
    seasons = features.aggregate_seasons(make_appearances(), make_games())
    assert seasons.iloc[0].league == "GB1"  # 180 minutes in GB1 vs. 45 in ES1


def test_season_targets_uses_latest_valuation_in_window():
    targets = features.season_targets(make_valuations())
    assert targets.to_dict("records") == [
        {"player_id": 7, "season": 2023, "market_value_in_eur": 10_000_000}
    ]


def test_season_targets_ignores_zero_and_out_of_window_values():
    valuations = pd.DataFrame({
        "player_id": [1, 2],
        "date": pd.to_datetime(["2024-05-01", "2024-09-01"]),
        "market_value_in_eur": [0, 5_000_000],
    })
    assert features.season_targets(valuations).empty


def test_add_derived_features_age_per90_and_log():
    table = pd.DataFrame({
        "season": [2023], "date_of_birth": pd.to_datetime(["2000-06-30"]),
        "minutes": [900], "goals": [2], "assists": [1], "yellow_cards": [0],
        "market_value_in_eur": [1_000_000],
    })
    row = features.add_derived_features(table).iloc[0]
    assert row.age == pytest.approx(24, abs=0.01)  # 30 June 2024
    assert row.goals_per90 == pytest.approx(0.2)
    assert row.assists_per90 == pytest.approx(0.1)
    assert row.log_value == pytest.approx(6.0)


def test_build_player_seasons_end_to_end():
    table = features.build_player_seasons(
        make_players(), make_appearances(), make_games(), make_valuations(), min_minutes=200
    )
    assert list(table.columns) == (["player_id", "name"] + features.FEATURE_COLUMNS
                                   + ["market_value_in_eur", features.TARGET_COLUMN])
    assert table.player_id.tolist() == [7]
    assert table.iloc[0].log_value == pytest.approx(7.0)


def test_build_player_seasons_applies_minimum_minutes():
    table = features.build_player_seasons(
        make_players(), make_appearances(), make_games(), make_valuations(), min_minutes=450
    )
    assert table.empty  # player 7 has only 225 minutes


def test_build_player_seasons_drops_missing_birth_date_and_position():
    players = make_players()
    players.loc[0, "position"] = "Missing"
    table = features.build_player_seasons(
        players, make_appearances(), make_games(), make_valuations(), min_minutes=0
    )
    assert table.empty


def test_no_market_value_column_is_used_as_feature():
    assert not any("value" in column for column in features.FEATURE_COLUMNS)
