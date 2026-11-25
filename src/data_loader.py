"""Load the Transfermarkt CSV tables used by the project.

Only the columns needed for the model are read. The leakage columns of
``players.csv`` (``market_value_in_eur`` and ``highest_market_value_in_eur``)
are deliberately never loaded, so they cannot end up among the features.
"""

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data/raw")

PLAYER_COLUMNS = [
    "player_id", "name", "date_of_birth", "position", "sub_position",
    "foot", "height_in_cm", "contract_expiration_date",
]
APPEARANCE_COLUMNS = [
    "game_id", "player_id", "player_club_id", "date", "competition_id",
    "yellow_cards", "red_cards", "goals", "assists", "minutes_played",
]
GAME_COLUMNS = [
    "game_id", "competition_id", "season", "date", "competition_type",
    "home_club_id", "away_club_id", "home_club_goals", "away_club_goals",
]
VALUATION_COLUMNS = [
    "player_id", "date", "market_value_in_eur",
    "player_club_domestic_competition_id",
]
COMPETITION_COLUMNS = ["competition_id", "name", "type", "sub_type", "country_name"]


def _read_table(data_dir, filename, columns, date_columns=()):
    """Read selected columns of one CSV file and clean them.

    Text columns are stripped of surrounding spaces and date columns are
    converted to datetime (invalid dates become ``NaT``).

    :param data_dir: Folder containing the CSV files
    :param filename: Name of the CSV file, e.g. ``"players.csv"``
    :param columns: Names of the columns to read
    :param date_columns: Subset of ``columns`` to parse as dates
    :return: DataFrame with the requested columns
    :raises FileNotFoundError: If the file does not exist
    :raises ValueError: If some requested columns are missing from the file
    """
    path = Path(data_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset first (see README, section Data)."
        )
    available = pd.read_csv(path, nrows=0).columns
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")

    table = pd.read_csv(path, usecols=columns)[columns]
    for column in table.columns:
        if column in date_columns:
            table[column] = pd.to_datetime(table[column], errors="coerce")
        elif pd.api.types.is_string_dtype(table[column]):
            table[column] = table[column].str.strip()
    return table


def load_players(data_dir=DEFAULT_DATA_DIR):
    """Load player profiles (without any market value column).

    :param data_dir: Folder containing ``players.csv``
    :return: DataFrame with one row per player
    """
    return _read_table(
        data_dir, "players.csv", PLAYER_COLUMNS,
        date_columns=("date_of_birth", "contract_expiration_date"),
    )


def load_appearances(data_dir=DEFAULT_DATA_DIR):
    """Load per-match player statistics (minutes, goals, assists, cards).

    :param data_dir: Folder containing ``appearances.csv``
    :return: DataFrame with one row per player and match
    """
    return _read_table(data_dir, "appearances.csv", APPEARANCE_COLUMNS,
                       date_columns=("date",))


def load_games(data_dir=DEFAULT_DATA_DIR):
    """Load matches, used to find the season and competition type.

    :param data_dir: Folder containing ``games.csv``
    :return: DataFrame with one row per match
    """
    return _read_table(data_dir, "games.csv", GAME_COLUMNS, date_columns=("date",))


def load_valuations(data_dir=DEFAULT_DATA_DIR):
    """Load the market value history (the prediction target).

    :param data_dir: Folder containing ``player_valuations.csv``
    :return: DataFrame with several rows per player, one per valuation date
    """
    return _read_table(data_dir, "player_valuations.csv", VALUATION_COLUMNS,
                       date_columns=("date",))


def load_competitions(data_dir=DEFAULT_DATA_DIR):
    """Load competitions (leagues, cups, national team tournaments).

    :param data_dir: Folder containing ``competitions.csv``
    :return: DataFrame with one row per competition
    """
    return _read_table(data_dir, "competitions.csv", COMPETITION_COLUMNS)
