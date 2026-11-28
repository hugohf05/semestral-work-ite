"""Command-line entry point: train, evaluate and use the market value model.

Usage (from the repository root, with the data in ``data/raw``)::

    python main.py train                 # tune on 2024, fit on 2012-2024, save the model
    python main.py evaluate              # final results on the 2025 test season -> reports/
    python main.py predict "Pedri"       # predicted vs. real value of one player
    python main.py predict "Pedri" --season 2022
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import evaluation, features, models, predictor
from src.data_loader import DEFAULT_DATA_DIR
from src.features import FEATURE_COLUMNS, TARGET_COLUMN

DEFAULT_REPORTS_DIR = Path("reports")


def train(table, model_path=predictor.DEFAULT_MODEL_PATH, grid=models.PARAM_GRID):
    """Tune the Random Forest on the validation season, refit it and save it.

    After choosing the hyperparameters, the final model is refitted on train
    and validation together (more data, more recent seasons). The test season
    is not used.

    :param table: Player-season table
    :param model_path: Where to save the model
    :param grid: Hyperparameter grid for ``models.tune_random_forest``
    :return: Tuple ``(best_params, tuning_results)``
    """
    train_set, validation, _ = models.time_split(table)
    best_params, results = models.tune_random_forest(train_set, validation, grid)
    # Final model: same hyperparameters, all seasons before the test season
    final_set = pd.concat([train_set, validation])
    model = models.make_random_forest(**best_params).fit(*models.split_xy(final_set))
    predictor.save_model(model, best_params, sorted(final_set["season"].unique()), model_path)
    return best_params, results


def evaluate(table, model_path=predictor.DEFAULT_MODEL_PATH, reports_dir=DEFAULT_REPORTS_DIR):
    """Measure every model once on the test season and write the reports.

    Baselines and linear regression are fitted on the same seasons as the
    saved Random Forest, so the comparison is fair.

    :param table: Player-season table
    :param model_path: Saved model from ``train``
    :param reports_dir: Folder for CSV tables and PNG plots
    :return: DataFrame of test metrics, one row per model
    """
    saved = predictor.load_model(model_path)
    train_set, validation, test = models.time_split(table)
    final_set = pd.concat([train_set, validation])
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    candidates = {
        "Global median": models.MedianBaseline(),
        "Median by position and league": models.MedianBaseline(["position", "league"]),
        "Linear regression": models.make_linear_regression(),
    }
    metrics = evaluation.compare_models(candidates, final_set, test, FEATURE_COLUMNS, TARGET_COLUMN)
    forest = saved["model"]
    prediction = forest.predict(test[FEATURE_COLUMNS])
    metrics.loc["Random Forest"] = evaluation.regression_metrics(test[TARGET_COLUMN], prediction)
    metrics.to_csv(reports_dir / "metrics_test.csv")

    evaluation.error_by_value_range(test[TARGET_COLUMN], prediction).to_csv(
        reports_dir / "error_by_value_range.csv")
    evaluation.value_gaps(test, prediction, top=20).to_csv(
        reports_dir / "largest_gaps.csv", index=False)
    evaluation.plot_predicted_vs_real(test[TARGET_COLUMN], prediction,
                                      reports_dir / "predicted_vs_real.png",
                                      "Random Forest, test season 2025")
    importance = evaluation.feature_importance(forest, test[FEATURE_COLUMNS], test[TARGET_COLUMN])
    evaluation.plot_feature_importance(importance, reports_dir / "feature_importance.png")
    importance.to_csv(reports_dir / "feature_importance.csv")
    # candidates["Linear regression"] was fitted inside compare_models, so it can be reused
    curves = {"Random Forest": evaluation.age_curve(forest, test[FEATURE_COLUMNS]),
              "Linear regression": evaluation.age_curve(candidates["Linear regression"],
                                                        test[FEATURE_COLUMNS])}
    evaluation.plot_age_curves(curves, reports_dir / "age_curve.png")
    # Long format (model, age, value) so the app can draw one line per model
    pd.concat(curves, names=["model"]).reset_index(level=0).to_csv(
        reports_dir / "age_curve.csv", index=False)
    return metrics


def predict(table, name, season=None, model_path=predictor.DEFAULT_MODEL_PATH):
    """Predict the value of one player and compare it with the real value.

    :param table: Player-season table
    :param name: Full or partial player name
    :param season: Season start year; latest season of the player if ``None``
    :param model_path: Saved model from ``train``
    :return: Dict with ``name``, ``season``, ``real_eur``, ``predicted_eur`` and ``ratio``
    """
    model = predictor.load_model(model_path)["model"]
    row = predictor.find_player(table, name, season)
    predicted = predictor.predict_value(model, row)[0]
    real = row["market_value_in_eur"].iloc[0]
    return {"name": row["name"].iloc[0], "season": int(row["season"].iloc[0]),
            "real_eur": real, "predicted_eur": predicted, "ratio": predicted / real}


def build_parser():
    """Define the command-line arguments.

    :return: ``argparse.ArgumentParser`` with the ``train``, ``evaluate`` and ``predict`` commands
    """
    parser = argparse.ArgumentParser(description="Football player market value predictor")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="folder with the Kaggle CSV files (default: data/raw)")
    parser.add_argument("--model", type=Path, default=predictor.DEFAULT_MODEL_PATH,
                        help="model file (default: models/model.joblib)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="tune, train and save the Random Forest")
    commands.add_parser("evaluate", help="test-season metrics and plots in reports/")
    predict_parser = commands.add_parser("predict", help="predict one player's value")
    predict_parser.add_argument("name", help="full or partial player name")
    predict_parser.add_argument("--season", type=int, help="season start year, e.g. 2024")
    return parser


def main(argv=None):
    """Run the command given on the command line.

    :param argv: Argument list (``None`` means ``sys.argv``); used by the tests
    :return: Exit code, 0 on success and 1 on a handled error
    """
    args = build_parser().parse_args(argv)
    try:
        table = features.load_player_seasons(args.data_dir)
        if args.command == "train":
            best, results = train(table, args.model)
            print(results.head(5).to_string(index=False))
            print(f"\nBest hyperparameters: {best}\nModel saved to {args.model}")
        elif args.command == "evaluate":
            metrics = evaluate(table, args.model)
            print(metrics.round(3).to_string())
            print(f"\nTables and plots saved to {DEFAULT_REPORTS_DIR}/")
        else:
            result = predict(table, args.name, args.season, args.model)
            print(f"{result['name']} ({result['season']}/{result['season'] + 1})\n"
                  f"  Real value:      {result['real_eur'] / 1e6:8.1f} M€\n"
                  f"  Predicted value: {result['predicted_eur'] / 1e6:8.1f} M€"
                  f"  (x{result['ratio']:.2f})")
    except (FileNotFoundError, ValueError) as error:
        # Expected problems (no data, no model, unknown player): short message, no traceback
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
