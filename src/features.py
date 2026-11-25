"""Build the player-season table used to train and evaluate the models.

Each output row is one player in one season of a top-5 league, with
profile and performance features and the end-of-season market value as
the target. The decisions behind every step are documented in
``docs/data_quality.md`` and ``notebooks/eda.ipynb``.
"""

import numpy as np
import pandas as pd

from src import data_loader

TOP5_LEAGUES = {
    "GB1": "Premier League", "ES1": "LaLiga", "IT1": "Serie A",
    "L1": "Bundesliga", "FR1": "Ligue 1",
}
EUROPEAN_CUPS = ["CL", "EL", "UCOL"]  # main phases only, qualifiers excluded
MIN_MINUTES = 450
VALUATION_MONTHS = (4, 7)  # April to July, inclusive

FEATURE_COLUMNS = [
    "age", "season", "position", "league", "minutes", "matches",
    "goals_per90", "assists_per90", "yellow_cards_per90", "red_cards",
    "team_points_per_game", "european_minutes",
]
TARGET_COLUMN = "log_value"


def aggregate_seasons(appearances, games, leagues=TOP5_LEAGUES):
    """Sum each player's league statistics per season.

    Only matches of the given leagues are used. ``season`` comes from
    ``games`` and is the start year (2025 means 2025/26). If a player played
    in two leagues in one season, ``league`` and ``club_id`` are those of the
    club with most minutes.

    :param appearances: Per-match statistics from ``load_appearances``
    :param games: Matches from ``load_games``
    :param leagues: Competition ids to keep
    :return: DataFrame with one row per ``player_id`` and ``season``
    """
    league_apps = appearances[appearances["competition_id"].isin(leagues)].merge(
        games[["game_id", "season"]], on="game_id", validate="many_to_one"
    )
    seasons = league_apps.groupby(["player_id", "season"], as_index=False).agg(
        minutes=("minutes_played", "sum"),
        matches=("game_id", "nunique"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
    )
    # Main club = the club with most minutes: sum minutes per club, sort them
    # and keep only the last (largest) row of every player-season
    main_club = (
        league_apps.groupby(["player_id", "season", "competition_id", "player_club_id"],
                            as_index=False)["minutes_played"].sum()
        .sort_values("minutes_played")
        .drop_duplicates(["player_id", "season"], keep="last")
        .rename(columns={"competition_id": "league", "player_club_id": "club_id"})
    )
    return seasons.merge(main_club[["player_id", "season", "league", "club_id"]],
                         on=["player_id", "season"], validate="one_to_one")


def team_points_per_game(games, leagues=TOP5_LEAGUES):
    """Compute every club's league points per game in every season.

    Each match is turned into two rows, one per team, so home and away
    games are treated the same way. Win = 3 points, draw = 1, loss = 0.

    :param games: Matches from ``load_games`` (with clubs and goals)
    :param leagues: Competition ids to use
    :return: DataFrame with ``club_id``, ``season`` and ``team_points_per_game``
    """
    league_games = games[games["competition_id"].isin(leagues)]
    home = league_games.rename(columns={"home_club_id": "club_id", "home_club_goals": "scored",
                                        "away_club_goals": "conceded"})
    away = league_games.rename(columns={"away_club_id": "club_id", "away_club_goals": "scored",
                                        "home_club_goals": "conceded"})
    # Long format: 1 row per team and match (a match appears twice, once per side)
    teams = pd.concat([home, away])[["club_id", "season", "scored", "conceded"]]
    # np.select picks the first true condition: win -> 3, draw -> 1, otherwise 0
    teams["points"] = np.select(
        [teams["scored"] > teams["conceded"], teams["scored"] == teams["conceded"]], [3, 1], 0)
    return (teams.groupby(["club_id", "season"], as_index=False)["points"].mean()
            .rename(columns={"points": "team_points_per_game"}))


def european_minutes(appearances, games, cups=EUROPEAN_CUPS):
    """Sum each player's minutes in European cups per season.

    :param appearances: Per-match statistics from ``load_appearances``
    :param games: Matches from ``load_games`` (gives the season)
    :param cups: Competition ids counted as European cups
    :return: DataFrame with ``player_id``, ``season`` and ``european_minutes``
    """
    cup_apps = appearances[appearances["competition_id"].isin(cups)].merge(
        games[["game_id", "season"]], on="game_id", validate="many_to_one")
    return (cup_apps.groupby(["player_id", "season"], as_index=False)["minutes_played"].sum()
            .rename(columns={"minutes_played": "european_minutes"}))


def season_targets(valuations, months=VALUATION_MONTHS):
    """Pick the end-of-season market value of every player.

    A valuation dated between April and July of year ``Y`` belongs to season
    ``Y - 1`` (e.g. May 2024 closes season 2023/24). If there are several,
    the latest one is kept. Values of zero are ignored.

    :param valuations: Market value history from ``load_valuations``
    :param months: First and last month of the valuation window
    :return: DataFrame with ``player_id``, ``season`` and ``market_value_in_eur``
    """
    first, last = months
    window = valuations[
        (valuations["market_value_in_eur"] > 0)
        & valuations["date"].dt.month.between(first, last)
    ].copy()
    # A valuation in May 2024 closes season 2023/24, whose start year is 2023
    window["season"] = window["date"].dt.year - 1
    latest = window.sort_values("date").drop_duplicates(["player_id", "season"], keep="last")
    return latest[["player_id", "season", "market_value_in_eur"]].reset_index(drop=True)


def add_derived_features(table):
    """Add age, per-90 statistics and the log10 target to a player-season table.

    Age is measured on 30 June at the end of the season.

    :param table: Player-seasons with ``date_of_birth``, ``minutes``,
        ``goals``, ``assists``, ``yellow_cards`` and ``market_value_in_eur``
    :return: A copy of ``table`` with the new columns
    """
    table = table.copy()
    # 30 June of the year after the start year, e.g. season 2023 -> 2024-06-30
    season_end = pd.to_datetime((table["season"] + 1).astype(str) + "-06-30")
    table["age"] = (season_end - table["date_of_birth"]).dt.days / 365.25  # .25 = leap years
    for stat in ("goals", "assists", "yellow_cards"):
        # Rate per full match, so players with different minutes can be compared
        table[f"{stat}_per90"] = table[stat] / table["minutes"] * 90
    table[TARGET_COLUMN] = np.log10(table["market_value_in_eur"])
    return table


def build_player_seasons(players, appearances, games, valuations, min_minutes=MIN_MINUTES):
    """Join all tables into the final player-season table.

    Rows are dropped when the player has fewer than ``min_minutes``, no
    valuation in the window, no birth date or position ``Missing``.

    :param players: Profiles from ``load_players``
    :param appearances: Per-match statistics from ``load_appearances``
    :param games: Matches from ``load_games``
    :param valuations: Market value history from ``load_valuations``
    :param min_minutes: Minimum league minutes in the season
    :return: DataFrame with identifiers, ``FEATURE_COLUMNS`` and ``TARGET_COLUMN``
    """
    seasons = aggregate_seasons(appearances, games)
    seasons = seasons[seasons["minutes"] >= min_minutes]
    # validate=...: pandas raises an error if a join would duplicate rows
    table = seasons.merge(season_targets(valuations), on=["player_id", "season"],
                          validate="one_to_one")
    profile = players[["player_id", "name", "date_of_birth", "position"]]
    table = table.merge(profile, on="player_id", validate="many_to_one")
    table = table.merge(team_points_per_game(games), on=["club_id", "season"],
                        how="left", validate="many_to_one")
    table = table.merge(european_minutes(appearances, games), on=["player_id", "season"],
                        how="left", validate="one_to_one")
    # No row in the European table = the player did not play in Europe that season
    table["european_minutes"] = table["european_minutes"].fillna(0)
    table = table.dropna(subset=["date_of_birth"])
    table = table[table["position"] != "Missing"]
    table = add_derived_features(table)
    columns = ["player_id", "name"] + FEATURE_COLUMNS + ["market_value_in_eur", TARGET_COLUMN]
    return table[columns].sort_values(["season", "player_id"]).reset_index(drop=True)


def load_player_seasons(data_dir=data_loader.DEFAULT_DATA_DIR, min_minutes=MIN_MINUTES):
    """Load the raw CSV files and build the player-season table.

    :param data_dir: Folder with the Kaggle CSV files
    :param min_minutes: Minimum league minutes in the season
    :return: The table returned by ``build_player_seasons``
    """
    return build_player_seasons(
        data_loader.load_players(data_dir),
        data_loader.load_appearances(data_dir),
        data_loader.load_games(data_dir),
        data_loader.load_valuations(data_dir),
        min_minutes=min_minutes,
    )
