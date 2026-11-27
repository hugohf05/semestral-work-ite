"""Save, load and use the trained model to predict one player's value.

Used by ``main.py`` (command line) and later by the Streamlit app, so both
predict in exactly the same way.
"""

from pathlib import Path

import joblib

from src.features import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path("models/model.joblib")


def save_model(model, params, seasons, path=DEFAULT_MODEL_PATH):
    """Save the fitted model together with information about how it was trained.

    :param model: Fitted scikit-learn pipeline
    :param params: Hyperparameters chosen on the validation season
    :param seasons: Seasons used for training, e.g. ``[2012, ..., 2024]``
    :param path: Output ``.joblib`` file (its folder is created if needed)
    :return: ``path``
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # One file holds the model and its metadata, so they can never get out of sync
    joblib.dump({"model": model, "params": params, "seasons": list(seasons)}, path)
    return path


def load_model(path=DEFAULT_MODEL_PATH):
    """Load a model saved with ``save_model``.

    :param path: ``.joblib`` file
    :return: Dict with ``model``, ``params`` and ``seasons``
    :raises FileNotFoundError: If the model has not been trained yet
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Train the model first: python main.py train")
    return joblib.load(path)


def find_player(table, name, season=None):
    """Find one player-season row by (part of) the player's name.

    :param table: Player-season table from ``features.build_player_seasons``
    :param name: Full or partial name, case-insensitive (e.g. ``"pedri"``)
    :param season: Season start year; the player's latest season if ``None``
    :return: One-row DataFrame
    :raises ValueError: If no player matches, or the name matches several players
    """
    # regex=False: names such as "N'Golo" must be matched literally
    rows = table[table["name"].str.contains(name, case=False, regex=False)]
    if season is not None:
        rows = rows[rows["season"] == season]
    if rows.empty:
        raise ValueError(f"No player-season found for '{name}'"
                         + (f" in season {season}" if season is not None else ""))
    names = sorted(rows["name"].unique())
    # An exact name wins over partial matches ("Pedri" must not clash with "Pedrinho")
    exact = [candidate for candidate in names if candidate.lower() == name.lower()]
    if exact:
        rows, names = rows[rows["name"] == exact[0]], exact
    if len(names) > 1:
        shown = ", ".join(names[:5]) + (" ..." if len(names) > 5 else "")
        raise ValueError(f"'{name}' matches several players: {shown}. Be more specific.")
    return rows.sort_values("season").tail(1)


def predict_value(model, rows):
    """Predict the market value in euros.

    :param model: Fitted model that predicts log10 euros
    :param rows: DataFrame containing ``FEATURE_COLUMNS``
    :return: Array of predicted values in euros
    """
    # The model works in log10 euros; 10 ** x converts back to euros
    return 10 ** model.predict(rows[FEATURE_COLUMNS])


def undervalued_players(table, model, season, max_age=30, min_value=1e6, top=20):
    """List the players whose predicted value is furthest above their real value.

    Veterans and very cheap players are filtered out: their large gaps come
    from the model's weakness at the end of careers, not from hidden talent.

    :param table: Player-season table
    :param model: Fitted model that predicts log10 euros
    :param season: Season to analyse (use the test season for honest results)
    :param max_age: Keep players younger than this
    :param min_value: Keep players worth at least this many euros
    :param top: Number of players to return
    :return: DataFrame sorted by ``ratio`` (predicted / real), largest first
    """
    rows = table[(table["season"] == season) & (table["age"] < max_age)
                 & (table["market_value_in_eur"] >= min_value)].copy()
    rows["predicted_eur"] = predict_value(model, rows)
    rows["ratio"] = rows["predicted_eur"] / rows["market_value_in_eur"]
    columns = ["name", "age", "position", "league", "market_value_in_eur", "predicted_eur", "ratio"]
    return rows.sort_values("ratio", ascending=False)[columns].head(top).reset_index(drop=True)
