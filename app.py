"""Streamlit web app to explore the market value model.

Run from the repository root (after ``python main.py train`` and
``python main.py evaluate``)::

    streamlit run app.py

Pages: player lookup, manual estimate, model insights and top undervalued.
The app only loads the saved model; it never retrains.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src import features, predictor
from src.features import TOP5_LEAGUES

REPORTS_DIR = Path("reports")
TEST_SEASON = 2025
POSITIONS = ["Attack", "Midfield", "Defender", "Goalkeeper"]


@st.cache_data  # loaded once per session instead of on every click
def load_table():
    """Load the player-season table (about 10 s the first time).

    :return: Player-season DataFrame
    """
    return features.load_player_seasons()


@st.cache_resource  # the model is a large object: keep one shared copy
def load_saved_model():
    """Load the trained model and its metadata.

    :return: Dict with ``model``, ``params`` and ``seasons``
    """
    return predictor.load_model()


def millions(value):
    """Format euros as millions, e.g. ``12_500_000 -> "12.5 M€"``.

    :param value: Amount in euros
    :return: Formatted text
    """
    return f"{value / 1e6:,.1f} M€"


def page_player_lookup(table, saved):
    """Search a player and compare the predicted and the real value."""
    st.header("Player lookup")
    name = st.text_input("Player name (full or partial)", "Pedri")
    seasons = sorted(table["season"].unique(), reverse=True)
    season = st.selectbox("Season (start year)", seasons, index=seasons.index(TEST_SEASON))
    if season in saved["seasons"]:
        # In-sample prediction: the model memorised this row during training
        st.warning(f"The model was trained on season {season}, so this prediction is "
                   f"optimistic. Choose {TEST_SEASON} for an honest prediction.")
    try:
        row = predictor.find_player(table, name, season)
    except ValueError as error:
        st.error(str(error))
        return
    real = row["market_value_in_eur"].iloc[0]
    predicted = predictor.predict_value(saved["model"], row)[0]
    ratio = predicted / real

    st.subheader(f"{row['name'].iloc[0]} — {season}/{season + 1}")
    left, middle, right = st.columns(3)
    left.metric("Real value", millions(real))
    middle.metric("Predicted value", millions(predicted))
    right.metric("Predicted / real", f"×{ratio:.2f}")
    if ratio > 1.5:
        st.info("Statistics suggest a higher value than the market gives (possibly undervalued).")
    elif ratio < 1 / 1.5:
        st.info("The market values this player above what his statistics suggest "
                "(fame, potential or skills the model does not see).")
    # Transposed, one column mixes text ("Midfield") and numbers: show it as text
    profile = row[features.FEATURE_COLUMNS].round(2).T.rename(columns=lambda _: "value").astype(str)
    st.dataframe(profile, use_container_width=True)


def page_manual_estimate(saved):
    """Form to estimate the value of an invented or missing player."""
    st.header("Manual estimate")
    with st.form("profile"):
        left, right = st.columns(2)
        values = {
            "age": left.slider("Age", 16, 40, 24),
            "position": left.selectbox("Position", POSITIONS),
            "league": left.selectbox("League", list(TOP5_LEAGUES),
                                     format_func=TOP5_LEAGUES.get),
            "season": left.number_input("Season (start year)", 2012, TEST_SEASON, TEST_SEASON),
            "team_points_per_game": left.slider(
                "Team league points per game", 0.5, 2.7, 1.4, 0.05,
                help="About 2.3+ for a champion, 1.4 for a mid-table team, 0.9 for a relegated one"),
            "minutes": right.slider("League minutes", 450, 3420, 2000, 10),
            "matches": right.slider("League matches", 5, 38, 25),
            "european_minutes": right.slider("Minutes in European cups", 0, 1500, 0, 10),
            "goals_per90": right.slider("Goals per 90", 0.0, 1.5, 0.2, 0.01),
            "assists_per90": right.slider("Assists per 90", 0.0, 1.0, 0.1, 0.01),
            "yellow_cards_per90": right.slider("Yellow cards per 90", 0.0, 0.8, 0.15, 0.01),
            "red_cards": right.slider("Red cards", 0, 4, 0),
        }
        submitted = st.form_submit_button("Estimate value")
    if submitted:
        row = pd.DataFrame([values])
        st.metric("Estimated market value", millions(predictor.predict_value(saved["model"], row)[0]))


def page_model_insights():
    """Show the test-season results written by ``python main.py evaluate``."""
    st.header("Model insights (test season 2025)")
    metrics_file = REPORTS_DIR / "metrics_test.csv"
    if not metrics_file.is_file():
        st.error("No reports found. Run: python main.py evaluate")
        return
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv(metrics_file, index_col=0).round(3), use_container_width=True)
    st.caption("r2: share of variation explained · mae_log: mean error in log10 (0.3 ≈ ×2) · "
               "within_factor_2: % of players predicted between half and double their value")
    for title, file in [("Predicted vs. real value", "predicted_vs_real.png"),
                        ("Which features matter (permutation importance)", "feature_importance.png"),
                        ("Effect of age with everything else fixed", "age_curve.png")]:
        st.subheader(title)
        st.image(str(REPORTS_DIR / file))


def page_top_undervalued(table, saved):
    """Table of the players with the largest predicted / real ratio in the test season."""
    st.header(f"Top undervalued players ({TEST_SEASON}/{TEST_SEASON + 1})")
    max_age = st.slider("Younger than", 20, 40, 30)
    min_value = st.select_slider("Real value at least", [0.1e6, 0.5e6, 1e6, 5e6, 10e6],
                                 value=1e6, format_func=millions)
    result = predictor.undervalued_players(table, saved["model"], TEST_SEASON, max_age, min_value)
    result["league"] = result["league"].map(TOP5_LEAGUES)
    for column in ("market_value_in_eur", "predicted_eur"):
        result[column] = result[column].map(millions)
    st.dataframe(result.round({"age": 1, "ratio": 2}), use_container_width=True)
    st.caption("A high ratio means the statistics suggest more value than the market gives. "
               "The model does not see injuries, contracts or skills, so these are leads, not facts.")


def main():
    """Build the page: sidebar navigation and the selected page."""
    st.set_page_config(page_title="Market Value Predictor", page_icon="⚽", layout="wide")
    st.title("⚽ Football Player Market Value Predictor")
    try:
        saved = load_saved_model()
        table = load_table()
    except FileNotFoundError as error:
        st.error(f"{error}")
        st.stop()
    page = st.sidebar.radio("Page", ["Player lookup", "Manual estimate",
                                     "Model insights", "Top undervalued"])
    st.sidebar.caption(f"Random Forest trained on seasons {saved['seasons'][0]}–"
                       f"{saved['seasons'][-1]}. Data: Transfermarkt via Kaggle (CC0).")
    if page == "Player lookup":
        page_player_lookup(table, saved)
    elif page == "Manual estimate":
        page_manual_estimate(saved)
    elif page == "Model insights":
        page_model_insights()
    else:
        page_top_undervalued(table, saved)


if __name__ == "__main__":
    main()
