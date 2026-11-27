# Testing

All tests run with one command from the repository root:

```bash
python test_main.py              # 56 tests, about 15 s
python test_main.py -k features  # only tests whose name contains "features"
```

Coverage report:

```bash
python -m pytest --cov=src --cov=main --cov=app --cov-report=term
```

## Approach and rationale

A machine learning model never gives an "exact" answer, so a test like "Pedri must be predicted at 63.3 M€" would break every time the data or the model changes. The tests therefore check two kinds of things:

1. **Deterministic logic with exact answers.** Joins, aggregations, per-90 statistics, the season mapping, the log transform, metrics and player search can be checked by hand. They use **small handmade tables** (a few rows) whose results can be computed on paper, e.g. 900 minutes and 2 goals → 0.2 goals per 90.
2. **Properties of the models.** On a small synthetic table where the value depends on the league and the age, every model must produce one finite prediction per row and the linear regression and the Random Forest must beat the global median.

The real Kaggle data is never needed, except for the app test (see below). The tests are fast, run on any computer and do not depend on the dataset version.

Special attention goes to the two errors that would invalidate the results without raising any error:

- **Data leakage:** market value columns of `players.csv` are never loaded, no feature contains the value, and the model saved by `train` never contains the test season.
- **Silent duplicates in joins:** every `merge` uses `validate=...`, so pandas raises an error instead of duplicating rows.

## Test levels

| Level | File | Tests | What is checked |
| --- | --- | --- | --- |
| Unit | `test_data_loader.py` | 8 | Selected columns, whitespace, date parsing, clear errors for missing files or columns, leakage columns never loaded |
| Unit | `test_features.py` | 12 | Season aggregation, main league and club, valuation window, age, per-90, log, filters, team points per game, European minutes |
| Unit / model | `test_models.py` | 8 | Time split, median baselines, unseen categories, models beat the global median, hyperparameter search |
| Unit | `test_evaluation.py` | 9 | Every metric on hand-checked values, error by value range, gaps, importance, age curve, plots saved |
| Unit | `test_predictor.py` | 9 | Save/load the model, name search (exact, partial, ambiguous, unknown), log → euros, undervalued filter |
| System | `test_main.py` | 4 | `train` → `evaluate` → `predict` end to end on a synthetic table; CLI error message without traceback |
| System | `test_app.py` | 6 | Each of the 4 app pages runs without exceptions; lookup shows the 3 values; warning for training seasons |
| | **Total** | **56** | |

`test_app.py` runs the real app with Streamlit's `AppTest`. It needs the data in `data/raw/` and a trained model, so it is **skipped automatically** when they are missing.

## Sample input/output pairs

### Features (`src/features.py`)

| Function | Input | Expected output |
| --- | --- | --- |
| `add_derived_features` | 900 minutes, 2 goals, 1 assist | `goals_per90 = 0.2`, `assists_per90 = 0.1` |
| `add_derived_features` | Born 2000-06-30, season 2023 | `age = 24.0` (measured on 2024-06-30) |
| `add_derived_features` | Value 1,000,000 € | `log_value = 6.0` |
| `season_targets` | Valuations on 2024-01-10, 2024-04-15 and 2024-06-01 | Season 2023, value of 2024-06-01 (latest inside April–July) |
| `season_targets` | Value 0 €, or a valuation in September | Ignored |
| `aggregate_seasons` | 180 min in the Premier League, 45 min in LaLiga | `league = GB1`, 225 minutes, 3 matches |
| `aggregate_seasons` | Player with only a Champions League match | No league row |
| `team_points_per_game` | Club wins 2-1 at home and draws 1-1 away | 2.0 points per game |
| `european_minutes` | 90 min in a CL match; no European match | 90; filled with 0 |
| `build_player_seasons` | Player with 225 minutes, `min_minutes=450` | Filtered out |
| `build_player_seasons` | Position `Missing` | Filtered out |

### Models and metrics (`src/models.py`, `src/evaluation.py`)

| Function | Input | Expected output |
| --- | --- | --- |
| `MedianBaseline(["position", "league"])` | Training targets 1 and 3 for group (A, x), 10 for (B, x) | Predicts 2, 2, 10 |
| `MedianBaseline` | Group never seen in training | Global median |
| `regression_metrics` | Prediction = real value | R² = 1, MAE = 0, within ×2 = 100 % |
| `regression_metrics` | Always predicting the mean | R² = 0 |
| `regression_metrics` | Real 1 M€ and 10 M€, predicted 2 M€ and 5 M€ | MAE € = 3 M€, median error = 75 %, within ×2 = 100 % |
| `error_by_value_range` | Cheap player overestimated, star underestimated | Mean error > 0 for "< 1 M€", < 0 for "≥ 50 M€" |
| `tune_random_forest` | Grid of 2 × 2 × 1 values | 4 results sorted by MAE; `None` (not `NaN`) for no depth limit |

### Prediction and command line (`src/predictor.py`, `main.py`)

| Input | Expected output |
| --- | --- |
| `find_player(table, "pedri")` with Pedri in 2023 and 2024 | Pedri, season 2024 (latest; case-insensitive) |
| `find_player(table, "Pedri")` with "Pedri" and "Pedrinho" | Pedri (exact name wins) |
| `find_player(table, "Ped")` with "Pedri" and "Pedro" | `ValueError`: matches several players |
| `find_player(table, "Messi")` | `ValueError`: no player-season found |
| Model predicting log10 = 7 | 10,000,000 € |
| `load_model` without a trained model | `FileNotFoundError` saying `python main.py train` |
| `python main.py --data-dir <empty folder> predict Pedri` | `Error: ...` on stderr, exit code 1, no traceback |
| `main.train` on a synthetic table | Saved model trained on seasons up to 2024 only |
| `main.evaluate` | 4 models in the metrics table; 6 files in `reports/` |

## Coverage

Measured with `pytest-cov` (lines of code executed by the tests):

| Module | Coverage |
| --- | --- |
| `src/features.py`, `src/models.py`, `src/evaluation.py`, `src/predictor.py` | 100 % |
| `src/data_loader.py` | 97 % |
| `app.py` | 88 % |
| `main.py` | 83 % (the text printed by the command line is not checked) |
| `src/data_quality.py` | 0 % (one-off script that reproduces the numbers in `data_quality.md`; not part of the program) |
| **Total** | **87 %** |

## Known gaps

- The tests check that the model **works** and beats the baselines, not that its accuracy stays at R² ≈ 0.81: the real accuracy is measured once by `python main.py evaluate` (see the README).
- The app test only runs where the data and the model exist.
- The printed output of `main.py` is not compared character by character.
