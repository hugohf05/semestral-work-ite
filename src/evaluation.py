"""Metrics, error analysis and plots for the market value models.

Predictions and targets are in log10 euros (``log_value``); metrics are
reported both on that scale and converted back to euros.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)

VALUE_RANGES = [0, 1e6, 5e6, 20e6, 50e6, np.inf]
VALUE_RANGE_LABELS = ["< 1 M€", "1–5 M€", "5–20 M€", "20–50 M€", "≥ 50 M€"]


def regression_metrics(y_true, y_pred):
    """Compute the comparison metrics for one model.

    :param y_true: Real values in log10 euros
    :param y_pred: Predicted values in log10 euros
    :return: Dict with ``r2``, ``mae_log``, ``rmse_log``, ``mae_eur``,
        ``median_abs_pct_error`` and ``within_factor_2``
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    real_eur, pred_eur = 10 ** y_true, 10 ** y_pred
    total_variation = ((y_true - y_true.mean()) ** 2).sum()
    return {
        "r2": 1 - (errors ** 2).sum() / total_variation,
        "mae_log": np.abs(errors).mean(),
        "rmse_log": np.sqrt((errors ** 2).mean()),
        "mae_eur": np.abs(pred_eur - real_eur).mean(),
        "median_abs_pct_error": np.median(np.abs(pred_eur - real_eur) / real_eur) * 100,
        # small tolerance: an error of exactly x2 must count despite float rounding
        "within_factor_2": (np.abs(errors) <= np.log10(2) + 1e-9).mean() * 100,
    }


def compare_models(models, train, evaluate, feature_columns, target_column):
    """Fit every model on ``train`` and measure it on ``evaluate``.

    :param models: Dict ``{name: unfitted model}``
    :param train: Training table
    :param evaluate: Validation or test table
    :param feature_columns: Names of the feature columns
    :param target_column: Name of the target column
    :return: DataFrame with one row of metrics per model
    """
    rows = {}
    for name, model in models.items():
        model.fit(train[feature_columns], train[target_column])
        prediction = model.predict(evaluate[feature_columns])
        rows[name] = regression_metrics(evaluate[target_column], prediction)
    return pd.DataFrame(rows).T


def error_by_value_range(y_true, y_pred):
    """Show how errors change with the real value of the player.

    ``mean_error_log`` below zero means the model underestimates that group.

    :param y_true: Real values in log10 euros
    :param y_pred: Predicted values in log10 euros
    :return: DataFrame with one row per value range
    """
    frame = pd.DataFrame({"true": np.asarray(y_true), "error": np.asarray(y_pred) - np.asarray(y_true)})
    frame["range"] = pd.cut(10 ** frame["true"], VALUE_RANGES, labels=VALUE_RANGE_LABELS, right=False)
    return frame.groupby("range", observed=False).agg(
        players=("error", "size"),
        mean_error_log=("error", "mean"),
        mae_log=("error", lambda e: e.abs().mean()),
    )


def value_gaps(table, y_pred, top=10):
    """List the players whose predicted value is furthest above the real one.

    :param table: Evaluation table with ``name``, ``season`` and ``market_value_in_eur``
    :param y_pred: Predicted values in log10 euros, in the same row order
    :param top: Number of players to return
    :return: DataFrame sorted by ``ratio`` (predicted / real), largest first
    """
    result = table[["name", "season", "position", "league", "age", "market_value_in_eur"]].copy()
    result["predicted_eur"] = 10 ** np.asarray(y_pred)
    result["ratio"] = result["predicted_eur"] / result["market_value_in_eur"]
    return result.sort_values("ratio", ascending=False).head(top).reset_index(drop=True)


def plot_predicted_vs_real(y_true, y_pred, path, title="Predicted vs. real value"):
    """Save a scatter plot of predicted vs. real value with the perfect diagonal.

    :param y_true: Real values in log10 euros
    :param y_pred: Predicted values in log10 euros
    :param path: Output PNG file
    :param title: Plot title
    :return: ``path``
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.4)
    limits = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(limits, limits, color="red", label="Perfect prediction")
    ax.set(xlabel="Real log10(€)", ylabel="Predicted log10(€)", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
