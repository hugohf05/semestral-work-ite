"""System tests for ``main.py``: train -> evaluate -> predict on a small synthetic table."""

import pytest

import main
from tests.test_models import make_table

SMALL_GRID = {"max_depth": [5], "min_samples_leaf": [5], "max_features": ["sqrt"]}


@pytest.fixture
def table():
    """Synthetic player-season table with the columns main.py needs."""
    table = make_table(rows=400)
    table["player_id"] = range(len(table))
    table["name"] = [f"Player {i}" for i in range(len(table))]
    table["market_value_in_eur"] = 10 ** table["log_value"]
    return table


@pytest.fixture
def model_path(table, tmp_path):
    """Train once with a tiny grid and return the saved model path."""
    path = tmp_path / "model.joblib"
    main.train(table, path, grid=SMALL_GRID)
    return path


def test_train_saves_model_without_test_season(model_path):
    from src import predictor
    saved = predictor.load_model(model_path)
    assert max(saved["seasons"]) == 2024  # 2025 is the test season, never trained on


def test_evaluate_writes_reports(table, model_path, tmp_path):
    reports = tmp_path / "reports"
    metrics = main.evaluate(table, model_path, reports)
    assert "Random Forest" in metrics.index and len(metrics) == 4
    for name in ("metrics_test.csv", "predicted_vs_real.png", "feature_importance.png",
                 "age_curve.png", "error_by_value_range.csv", "largest_gaps.csv"):
        assert (reports / name).is_file()


def test_predict_one_player(table, model_path):
    result = main.predict(table, "Player 7", model_path=model_path)
    assert result["name"] == "Player 7"
    assert result["predicted_eur"] > 0
    assert result["ratio"] == pytest.approx(result["predicted_eur"] / result["real_eur"])


def test_cli_reports_missing_data_without_traceback(tmp_path, capsys):
    exit_code = main.main(["--data-dir", str(tmp_path), "predict", "Pedri"])
    assert exit_code == 1
    assert "Error:" in capsys.readouterr().err
