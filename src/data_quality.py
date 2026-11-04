"""Print data quality statistics of the Transfermarkt dataset.

Run ``python -m src.data_quality`` to reproduce the numbers reported in
``docs/data_quality.md``.
"""

import sys
from pathlib import Path

import pandas as pd

from src import data_loader


def table_overview(data_dir):
    """Print rows, columns and the columns with missing values of every CSV.

    :param data_dir: Folder containing the CSV files
    """
    for path in sorted(Path(data_dir).glob("*.csv")):
        table = pd.read_csv(path, low_memory=False)
        nulls = (table.isna().mean() * 100).round(1)
        nulls = nulls[nulls > 0].sort_values(ascending=False)
        print(f"{path.name}: {len(table)} rows, {table.shape[1]} columns, "
              f"{int(table.duplicated().sum())} duplicated rows")
        print(f"    null % (top 5): {nulls.head(5).to_dict()}")


def used_tables_checks(data_dir):
    """Print integrity and range checks for the tables used by the project.

    :param data_dir: Folder containing the CSV files
    """
    players = data_loader.load_players(data_dir)
    appearances = data_loader.load_appearances(data_dir)
    games = data_loader.load_games(data_dir)
    valuations = data_loader.load_valuations(data_dir)
    competitions = data_loader.load_competitions(data_dir)

    print("appearances.game_id not in games:",
          int((~appearances.game_id.isin(games.game_id)).sum()))
    print("appearances.player_id not in players:",
          int((~appearances.player_id.isin(players.player_id)).sum()))
    print("valuations.player_id not in players:",
          int((~valuations.player_id.isin(players.player_id)).sum()))
    print("minutes_played min/max:", appearances.minutes_played.min(),
          appearances.minutes_played.max(),
          "| > 120:", int((appearances.minutes_played > 120).sum()))
    print("valuation min/max:", valuations.market_value_in_eur.min(),
          valuations.market_value_in_eur.max(),
          "| <= 0:", int((valuations.market_value_in_eur <= 0).sum()))
    print("players without any appearance:",
          int((~players.player_id.isin(appearances.player_id)).sum()), "of", len(players))
    print("players height < 140 cm:", int((players.height_in_cm < 140).sum()))
    print("position values:", players.position.value_counts(dropna=False).to_dict())

    merged = appearances.merge(games[["game_id", "season", "competition_type"]],
                               on="game_id", how="left")
    print("appearances by competition_type:",
          merged.competition_type.value_counts(dropna=False).to_dict())
    print("appearances per season:", merged.groupby("season").size().to_dict())
    print("domestic leagues:",
          sorted(competitions[competitions.type == "domestic_league"].competition_id))

    end_of_season = valuations[valuations.date.dt.month.isin([5, 6])]
    print("valuations in May-June per year:",
          end_of_season.groupby(end_of_season.date.dt.year).size().tail(15).to_dict())


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else data_loader.DEFAULT_DATA_DIR
    table_overview(folder)
    print()
    used_tables_checks(folder)
