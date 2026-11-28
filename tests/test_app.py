"""System test of the Streamlit app: every page runs without errors.

Needs the real data and a trained model, so it is skipped when they are missing.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.predictor import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not (DEFAULT_MODEL_PATH.is_file() and Path("data/raw/players.csv").is_file()),
    reason="needs data/raw and a trained model (python main.py train)",
)
APP_FILE = Path(__file__).resolve().parent.parent / "app.py"  # repository root
PAGES = ["Overview", "Player lookup", "Manual estimate", "Model insights", "Top undervalued"]


@pytest.fixture(scope="module")
def app():
    """Start the app once for all tests (loading the data takes several seconds)."""
    return AppTest.from_file(str(APP_FILE), default_timeout=120).run()


@pytest.mark.parametrize("page", PAGES)
def test_every_page_runs_without_exceptions(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception


def test_player_lookup_shows_real_and_predicted_value(app):
    app.sidebar.radio[0].set_value("Player lookup").run()
    app.text_input[0].set_value("Pedri").run()
    labels = [metric.label for metric in app.metric]
    assert labels == ["Real value", "Predicted value", "Predicted / real"]


def test_training_season_shows_warning(app):
    app.sidebar.radio[0].set_value("Player lookup").run()
    app.selectbox[0].set_value(2023).run()
    assert any("trained on season 2023" in warning.value for warning in app.warning)


def test_overview_shows_test_results(app):
    app.sidebar.radio[0].set_value("Overview").run()
    assert [metric.label for metric in app.metric] == [
        "Variation explained (R²)", "Players within ×2", "Typical error"]


def test_player_lookup_explains_the_prediction(app):
    app.sidebar.radio[0].set_value("Player lookup").run()
    app.text_input[0].set_value("Pedri")
    app.selectbox[0].set_value(2025).run()
    assert not app.exception
    assert any("Why this prediction" in header.value for header in app.subheader)
    comparison = app.dataframe[0].value
    assert "Team points per game" in comparison.index
