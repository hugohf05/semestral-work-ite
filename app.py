"""Streamlit web app to explore the market value model.

Run from the repository root (after ``python main.py train`` and
``python main.py evaluate``)::

    streamlit run app.py

Pages: overview, player lookup (with an explanation of each prediction),
manual estimate, model insights and top undervalued. The app only loads the
saved model; it never retrains.
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import features, predictor
from src.features import FEATURE_COLUMNS, TARGET_COLUMN, TOP5_LEAGUES

REPORTS_DIR = Path("reports")
TEST_SEASON = 2025
POSITIONS = ["Attack", "Midfield", "Defender", "Goalkeeper"]

# Chart colours (validated for colour-blind readers): series 1 blue, series 2 orange
BLUE, ORANGE, GREY = "#257de0", "#f47230", "#9c9c9c"
FEATURE_LABELS = {
    "age": "Age", "minutes": "League minutes", "goals_per90": "Goals per 90",
    "assists_per90": "Assists per 90", "team_points_per_game": "Team points per game",
    "european_minutes": "European cup minutes", "market_value_in_eur": "Market value (M€)",
    "season": "Season", "position": "Position", "league": "League", "matches": "Matches",
    "yellow_cards_per90": "Yellow cards per 90", "red_cards": "Red cards",
}


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


@st.cache_data
def test_predictions(_model, table):
    """Predict every player of the test season (cached).

    The leading underscore tells Streamlit not to hash the model (it is large).

    :param _model: Fitted model
    :param table: Player-season table
    :return: Test-season rows with ``predicted_log`` added
    """
    test = table[table["season"] == TEST_SEASON].copy()
    test["predicted_log"] = _model.predict(test[FEATURE_COLUMNS])
    return test


def read_report(name):
    """Read a CSV written by ``python main.py evaluate``, or ``None`` if it is missing.

    :param name: File name inside ``reports/``
    :return: DataFrame or ``None``
    """
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.is_file() else None


def millions(value):
    """Format euros as millions, e.g. ``12_500_000 -> "12.5 M€"``.

    :param value: Amount in euros
    :return: Formatted text
    """
    return f"{value / 1e6:,.1f} M€"


# ---------------------------------------------------------------- charts


def what_if_chart(curve, feature, current_value, title):
    """Line of the predicted value while one feature changes, with the player's point.

    :param curve: Output of ``predictor.what_if``
    :param feature: Name of the changed feature
    :param current_value: The player's real value of that feature
    :param title: Axis title
    :return: Altair chart
    """
    data = curve.assign(value_m=curve["predicted_eur"] / 1e6)
    # Index of the scenario closest to the player's real value, to mark it with a dot
    current = data.iloc[[int(np.argmin(np.abs(data[feature] - current_value)))]]
    base = alt.Chart(data).encode(
        x=alt.X(f"{feature}:Q", title=title),
        y=alt.Y("value_m:Q", title="Predicted value (M€)"),
        tooltip=[alt.Tooltip(f"{feature}:Q", title=title, format=".2f"),
                 alt.Tooltip("value_m:Q", title="Predicted (M€)", format=".1f")],
    )
    line = base.mark_line(color=BLUE, strokeWidth=2)
    hover = base.mark_point(size=120, opacity=0)  # invisible, larger hit area for the tooltip
    point = alt.Chart(current).mark_point(size=100, filled=True, color=BLUE,
                                          stroke="white", strokeWidth=2).encode(
        x=f"{feature}:Q", y="value_m:Q")
    label = alt.Chart(current).mark_text(dy=-14, fontWeight="bold").encode(
        x=f"{feature}:Q", y="value_m:Q", text=alt.value("now"))
    return (line + hover + point + label).properties(height=260)


def metrics_chart(metrics):
    """Horizontal bars: share of players predicted within x2, per model.

    :param metrics: ``metrics_test.csv`` as a DataFrame
    :return: Altair chart
    """
    order = list(metrics["model"])
    bars = alt.Chart(metrics).mark_bar(color=BLUE, cornerRadiusEnd=4, height=22).encode(
        x=alt.X("within_factor_2:Q", title="Players predicted within ×2 of their real value (%)",
                scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("model:N", sort=order, title=None),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("within_factor_2:Q", title="Within ×2 (%)", format=".1f"),
                 alt.Tooltip("r2:Q", title="R²", format=".3f"),
                 alt.Tooltip("median_abs_pct_error:Q", title="Median error (%)", format=".1f")],
    )
    labels = bars.mark_text(align="left", dx=4).encode(
        text=alt.Text("within_factor_2:Q", format=".0f"))
    return (bars + labels).properties(height=170)


def predicted_vs_real_chart(test):
    """Scatter of predicted vs. real value (log scale) with the perfect-prediction diagonal.

    :param test: Output of ``test_predictions``
    :return: Altair chart
    """
    data = test.assign(real_m=10 ** test[TARGET_COLUMN] / 1e6,
                       predicted_m=10 ** test["predicted_log"] / 1e6)
    # nice=False: keep exactly this range instead of rounding out to 0.01-1,000
    scale = alt.Scale(type="log", domain=[0.05, 300], nice=False)
    points = alt.Chart(data).mark_circle(size=30, opacity=0.45, color=BLUE).encode(
        x=alt.X("real_m:Q", title="Real value (M€, log scale)", scale=scale),
        y=alt.Y("predicted_m:Q", title="Predicted value (M€, log scale)", scale=scale),
        tooltip=[alt.Tooltip("name:N", title="Player"),
                 alt.Tooltip("league:N", title="League"),
                 alt.Tooltip("real_m:Q", title="Real (M€)", format=".1f"),
                 alt.Tooltip("predicted_m:Q", title="Predicted (M€)", format=".1f")],
    )
    diagonal = alt.Chart(pd.DataFrame({"x": [0.05, 300], "y": [0.05, 300]})).mark_line(
        color=GREY, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    return (diagonal + points).properties(height=420).interactive()  # zoom and pan


def importance_chart(importance):
    """Horizontal bars of permutation importance, most important first.

    :param importance: ``feature_importance.csv`` as a DataFrame
    :return: Altair chart
    """
    data = importance.rename(columns={importance.columns[0]: "feature"})
    data["feature"] = data["feature"].map(FEATURE_LABELS)
    # Tiny negative values (pure noise) are drawn as 0
    data["importance"] = data["importance"].clip(lower=0)
    return alt.Chart(data).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
        x=alt.X("importance:Q", title="Increase in error when the feature is shuffled"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("feature:N", title="Feature"),
                 alt.Tooltip("importance:Q", title="Importance", format=".3f")],
    ).properties(height=320)


def age_curve_chart(curves):
    """One line per model: average predicted value when only the age changes.

    :param curves: ``age_curve.csv`` as a DataFrame (``model``, ``age``, ``predicted_eur``)
    :return: Altair chart
    """
    data = curves.assign(value_m=curves["predicted_eur"] / 1e6)
    colour = alt.Color("model:N", scale=alt.Scale(domain=["Random Forest", "Linear regression"],
                                                  range=[BLUE, ORANGE]),
                       legend=alt.Legend(orient="top", title=None))
    base = alt.Chart(data).encode(
        # Extra room on the right for the line labels
        x=alt.X("age:Q", title="Age", scale=alt.Scale(domain=[18, 44], nice=False),
                axis=alt.Axis(values=list(range(18, 37, 2)))),  # ticks only where there is data
        y=alt.Y("value_m:Q", title="Average predicted value (M€)"),
        color=colour,
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("age:Q", title="Age"),
                 alt.Tooltip("value_m:Q", title="Predicted (M€)", format=".1f")],
    )
    lines = base.mark_line(strokeWidth=2) + base.mark_point(size=40, filled=True)
    # Direct labels at the right end of each line (identity is not given by colour alone)
    ends = alt.Chart(data[data["age"] == data["age"].max()]).mark_text(
        align="left", dx=6, color="#444").encode(x="age:Q", y="value_m:Q", text="model:N")
    return (lines + ends).properties(height=320)


# ---------------------------------------------------------------- pages


def page_overview(saved):
    """Landing page: what the program does, for whom, how to use it and key results."""
    st.header("Overview")
    st.markdown(
        "**What it does:** estimates a football player's **market value** "
        "(Transfermarkt) from his age, position, league, the strength of his club and "
        "one season of statistics in the top 5 European leagues.\n\n"
        "**For whom:** football fans, fantasy-league players and data-analytics students "
        "who want a quick, explainable estimate."
    )
    metrics = read_report("metrics_test.csv")
    if metrics is not None:
        forest = metrics.set_index(metrics.columns[0]).loc["Random Forest"]
        st.subheader(f"Results on the test season {TEST_SEASON}/{TEST_SEASON + 1}")
        left, middle, right = st.columns(3)
        left.metric("Variation explained (R²)", f"{forest['r2']:.2f}")
        middle.metric("Players within ×2", f"{forest['within_factor_2']:.0f} %")
        right.metric("Typical error", f"{forest['median_abs_pct_error']:.0f} %")
        st.caption("Measured once, on players the model never saw during training or tuning.")

    st.subheader("Three levels of models")
    level1, level2, level3 = st.columns(3)
    level1.markdown("**1. Baseline**\n\nMedian value of the player's position and league.\n\n"
                    "*Does machine learning add anything?*")
    level2.markdown("**2. Linear regression**\n\nA weighted sum: each feature multiplies "
                    "the value by a fixed factor.\n\n*Is a simple model enough?*")
    level3.markdown("**3. Random Forest** (used here)\n\n300 decision trees, each a series "
                    "of yes/no questions; their predictions are averaged.\n\n"
                    "*Captures curves and combinations.*")

    st.subheader("How to use this app")
    st.markdown(
        "1. **Player lookup** — search a player, compare real and predicted value and see "
        "*why*: how he compares with similar players and how the value would change.\n"
        "2. **Manual estimate** — invent a player with the sliders.\n"
        "3. **Model insights** — interactive charts of the test results.\n"
        "4. **Top undervalued** — players whose statistics suggest more than their price."
    )
    st.caption(f"Random Forest trained on seasons {saved['seasons'][0]}–{saved['seasons'][-1]}. "
               "Data: Transfermarkt via Kaggle (CC0).")


def show_explanation(table, row, model):
    """Explain one prediction: comparison with peers and two what-if curves."""
    player = row.iloc[0]
    st.subheader("Why this prediction?")

    comparison = predictor.compare_with_peers(table, row)
    peers = int(comparison["peers"].iloc[0])
    st.markdown(f"**Compared with the typical {player['position'].lower()} of "
                f"{TOP5_LEAGUES[player['league']]} in {int(player['season'])}** "
                f"(median of {peers} players)")
    shown = comparison[["player", "peers_median"]].copy()
    shown.loc["market_value_in_eur"] /= 1e6  # euros -> millions, like the label says
    shown["difference"] = np.where(shown["player"] > shown["peers_median"], "▲ higher",
                                   np.where(shown["player"] < shown["peers_median"],
                                            "▼ lower", "= same"))
    shown.index = [FEATURE_LABELS[name] for name in shown.index]
    st.dataframe(shown.rename(columns={"player": "This player", "peers_median": "Typical"})
                 .round(2), use_container_width=True)

    st.markdown("**What if?** The model's prediction when only one feature changes "
                "(everything else stays as it is).")
    left, right = st.columns(2)
    ages = np.arange(17, 38)
    left.altair_chart(what_if_chart(predictor.what_if(model, row, "age", ages), "age",
                                    player["age"], "Age"), use_container_width=True)
    points = np.round(np.arange(0.6, 2.75, 0.05), 2)
    right.altair_chart(what_if_chart(
        predictor.what_if(model, row, "team_points_per_game", points), "team_points_per_game",
        player["team_points_per_game"], "Team points per game (club strength)"),
        use_container_width=True)


def page_player_lookup(table, saved):
    """Search a player, compare the predicted and the real value and explain it."""
    st.header("Player lookup")
    left, right = st.columns([2, 1])
    name = left.text_input("Player name (full or partial)", "Pedri")
    seasons = sorted(table["season"].unique(), reverse=True)
    season = right.selectbox("Season (start year)", seasons, index=seasons.index(TEST_SEASON))
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

    player = row.iloc[0]
    st.subheader(f"{player['name']} — {season}/{season + 1}")
    st.caption(f"{player['position']} · {TOP5_LEAGUES[player['league']]} · "
               f"{player['age']:.0f} years")
    left, middle, right = st.columns(3)
    left.metric("Real value", millions(real))
    middle.metric("Predicted value", millions(predicted))
    right.metric("Predicted / real", f"×{ratio:.2f}")
    if ratio > 1.5:
        st.info("Statistics suggest a higher value than the market gives (possibly undervalued).")
    elif ratio < 1 / 1.5:
        st.info("The market values this player above what his statistics suggest "
                "(fame, potential or skills the model does not see).")
    show_explanation(table, row, saved["model"])


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


def page_model_insights(table, saved):
    """Interactive charts of the test-season results."""
    st.header(f"Model insights (test season {TEST_SEASON}/{TEST_SEASON + 1})")
    metrics = read_report("metrics_test.csv")
    importance = read_report("feature_importance.csv")
    curves = read_report("age_curve.csv")
    if metrics is None or importance is None or curves is None:
        st.error("Reports not found. Run: python main.py evaluate")
        return
    metrics = metrics.rename(columns={metrics.columns[0]: "model"})

    st.subheader("How often is each model close?")
    st.altair_chart(metrics_chart(metrics), use_container_width=True)
    with st.expander("All metrics (table)"):
        st.dataframe(metrics.set_index("model").round(3), use_container_width=True)
        st.caption("r2: share of variation explained · mae_log: mean error in log10 "
                   "(0.3 ≈ ×2) · median_abs_pct_error: error of the typical player")

    st.subheader("Predicted vs. real value")
    st.caption("Each dot is a player. On the dashed line the prediction is perfect. "
               "Hover for names; scroll to zoom.")
    st.altair_chart(predicted_vs_real_chart(test_predictions(saved["model"], table)),
                    use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Which features matter")
        st.altair_chart(importance_chart(importance), use_container_width=True)
        st.caption("Season shows 0 only because the test set has a single season.")
    with right:
        st.subheader("Effect of age")
        st.altair_chart(age_curve_chart(curves), use_container_width=True)
        st.caption("All test players get the same age; everything else is kept.")


def page_top_undervalued(table, saved):
    """Table of the players with the largest predicted / real ratio in the test season."""
    st.header(f"Top undervalued players ({TEST_SEASON}/{TEST_SEASON + 1})")
    left, right = st.columns(2)
    max_age = left.slider("Younger than", 20, 40, 30)
    min_value = right.select_slider("Real value at least", [0.1e6, 0.5e6, 1e6, 5e6, 10e6],
                                    value=1e6, format_func=millions)
    result = predictor.undervalued_players(table, saved["model"], TEST_SEASON, max_age, min_value)
    result["league"] = result["league"].map(TOP5_LEAGUES)
    for column in ("market_value_in_eur", "predicted_eur"):
        result[column] = result[column].map(millions)
    st.dataframe(result.round({"age": 1, "ratio": 2}), use_container_width=True)
    st.caption("A high ratio means the statistics suggest more value than the market gives. "
               "The model does not see injuries, contracts or skills, so these are leads, not facts.")


PAGES = ["Overview", "Player lookup", "Manual estimate", "Model insights", "Top undervalued"]


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
    page = st.sidebar.radio("Page", PAGES)
    if page == "Overview":
        page_overview(saved)
    elif page == "Player lookup":
        page_player_lookup(table, saved)
    elif page == "Manual estimate":
        page_manual_estimate(saved)
    elif page == "Model insights":
        page_model_insights(table, saved)
    else:
        page_top_undervalued(table, saved)


if __name__ == "__main__":
    main()
