# Football Player Market Value Predictor

*ITE semestral work — CTU Prague, autumn 2026 — individual project*

## Summary

**Project question:** can we estimate a football player's market value using only their profile (age, position, league, contract) and their performance in one season?

- **Task type:** supervised regression (predict a number in euros).
- **Data:** public Kaggle dataset "Football Data from Transfermarkt" (players from the main European leagues since ~2012).
- **Main method:** Random Forest, compared with linear regression and simple baselines.
- **Visual platform:** a basic web app (Streamlit) where the user selects a player, or enters a profile by hand, and sees the predicted value next to the real value.
- **Expected result:** a model that clearly beats the baseline "median value by position and league", plus an analysis of which factors matter most and which players look undervalued.

**Why this topic:** a simple rule gets market value badly wrong, so the model has real room to add value. The result is also easy to show visually in a 10-minute presentation.

## Course requirements

| ITE requirement | How this project meets it |
| --- | --- |
| Propose a program, describe inputs/outputs and logic with UML diagrams | Inputs/outputs below; activity, component and sequence diagrams in `docs/diagrams/` |
| Clean, modular code | Separate modules for loading, features, models, evaluation and app (see *Program structure*) |
| Tests in `tests/`, all runnable via `test_main.py` | `pytest` suite: unit, integration and system tests (see *Testing strategy*) |
| Docstrings in all Python files | reST-style docstrings (`:param:` / `:return:`), PEP 8, checked with `pylint` |
| README.md | Purpose, installation, usage, input/output format, limitations |
| tests.md, linked from README | `docs/tests.md` with approach, rationale and sample input/output pairs |
| License | MIT for the code; dataset license cited separately |
| Incremental Git history, repo `semestral-work-ite`, GitHub Release | Small commits every work session; release `v1.0.0` at the end |
| Presentation (10 min, diagrams) | What it does, target audience, live app demo, test coverage |
| Must be able to explain AI-generated code | Methods chosen because they are explainable; see *Methods* |

## Inputs and outputs

| | Description |
| --- | --- |
| **Input (training)** | Transfermarkt CSV files downloaded from Kaggle into `data/raw/` |
| **Input (app)** | Player name from the dataset, **or** a manual profile: age, position, league, minutes, goals, assists, contract years left |
| **Output (training)** | Trained model file (`models/model.joblib`), metrics table, plots in `reports/` |
| **Output (app)** | Predicted market value (€), real value if known, the difference (under/overvalued) and charts |
| **Errors** | Clear messages for missing data files, an unknown player or invalid input values |

## Target audience

Football fans, fantasy-league players and scouting/data-analytics students who want a quick, explainable estimate of a player's value.

## The data

Source: Kaggle **"Football Data from Transfermarkt"** (`davidcariboo/player-scores`), updated often. It is a set of related CSV tables, like a relational database.

| Table | Content | Join key |
| --- | --- | --- |
| `players.csv` | Profile: birth date, position, sub-position, foot, height, nationality, current club, contract end | `player_id` |
| `player_valuations.csv` | Market value history with dates (several per player) | `player_id`, `date` |
| `appearances.csv` | One row per player per match: minutes, goals, assists, cards | `player_id`, `game_id` |
| `games.csv` | Matches: date, competition, season | `game_id` |
| `clubs.csv` | Clubs and their league | `club_id` |
| `competitions.csv` | Competitions (LaLiga, Premier League, Champions League…) | `competition_id` |

**Joins:** `appearances` + `games` (season and competition) → aggregate per player and season → join with `players` (profile) and `player_valuations` (target).

**To check after download:** exact column names, available seasons and row counts, which change between versions. Record the dataset version and download date in the README.

**Not committed to Git:** the raw data is large, so `data/` goes in `.gitignore` and the README explains how to download it. Tests use small handmade CSV files in `tests/fixtures/`.

## Problem formulation

Each row of the final dataset is **one player in one season**. The target is their market value at the end of that season.

- **Features:** player statistics for that season plus their profile at that moment.
- **Target:** the `player_valuations` entry closest to the end of the season (e.g. May–July).
- **Expected size:** several thousand rows, since each player appears in several seasons.

### Key decision: do not use the previous value

No previous valuation and no "highest value ever" column are used as features.

- **Reason:** this year's value is very close to last year's. With that feature, the rule "value = previous value" is almost unbeatable and the model learns nothing interesting.
- **Data leakage:** the all-time highest value can include valuations *after* the predicted season.
- **Optional experiment:** train a version *with* the previous value and compare.

## Candidate features

The final list is decided after the exploratory data analysis (EDA).

| Group | Feature | Source | Why it should matter |
| --- | --- | --- | --- |
| Profile | Age | Birth date vs. season date | Value rises until ~25–27, then drops |
| Profile | Position and sub-position | `players` | Forwards are usually worth more than goalkeepers |
| Profile | Height, preferred foot | `players` | Probably small effect; to check |
| Performance | Minutes and matches played | Sum of `appearances` | Shows whether the player is a starter |
| Performance | Goals and assists per 90 min | `appearances` | Compares players with different minutes |
| Performance | Cards | `appearances` | Probably small effect |
| Context | Club's league | `clubs` / `competitions` | The Premier League inflates values |
| Context | Plays in a European competition | `games` | Signal of a strong, visible club |
| Contract | Years of contract left | `players` | Short contract → lower value |

**Contract warning:** `players.csv` only stores the *current* contract, which is wrong for older seasons. Either use it only for the latest season or drop it, and justify the choice.

## Preprocessing

### Why use the log of the value

Few players are worth €100M and many are worth under €1M. If the model minimises the error in euros, it only cares about the stars.

The logarithm turns multiplication into addition: +1 in log₁₀ means ×10 in value.

| Value | log₁₀(value) |
| --- | --- |
| €0.5M | 5.7 |
| €5M | 6.7 |
| €100M | 8.0 |

So a 20 % error counts the same for every player. The model is trained on the log and the result is converted back to euros:

$$y = \log(\text{value}) \qquad \widehat{\text{value}} = e^{\hat{y}}$$

### Steps

1. Join `appearances` with `games` and aggregate per player and season (minutes, goals, assists, cards).
2. Join each player-season with its end-of-season valuation (the trickiest join, by date).
3. Compute age for that season and per-90 statistics.
4. Filter out players with very few minutes (e.g. < 450), because their per-90 stats are noise.
5. Handle missing values (height, foot, contract): impute with the median or drop, and justify.
6. Review outliers (extreme goals per 90 in players with few minutes).
7. One-hot encode categorical features (position, league).
8. Scale numeric features only for linear regression; Random Forest does not need it.
9. After every join, check that the row count makes sense (no duplicated players). These checks become automated tests.

## Methods

Three levels are compared: simple baselines, linear regression and Random Forest (main method).

### 1. Baselines

A baseline is a naive rule used as a yardstick: if the model does not beat it, it adds nothing.

- **Baseline 1:** always predict the global median value.
- **Baseline 2 (the serious one):** predict the median value for the player's position and league.

### 2. Linear regression

The value is a weighted sum of the features; the model learns one weight per feature:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n$$

- Training finds the weights that minimise the sum of squared errors.
- **Advantage:** very interpretable. Since we predict log value, a weight of 0.3 on "goals per 90" means each extra goal per 90 multiplies the value by e^0.3 ≈ 1.35 (+35 %).
- **Limitation:** only straight-line effects. For the age curve, add age² as an extra feature.

### 3. Random Forest (main method)

**A decision tree** is a series of yes/no questions, e.g. "plays in the Premier League?" → "under 24?" → "more than 2,000 minutes?". Each leaf predicts the mean value of the training players that reached it. A single tree tends to memorise the data (overfitting).

**A Random Forest is many different trees that vote:**

1. Train e.g. 300 trees.
2. Each tree sees a random sample of rows (*bootstrap*) and, at each split, only a random subset of features.
3. The final prediction is the mean of all trees.

Each tree makes different mistakes, so averaging cancels them out.

**Why it fits:** it captures interactions (young *and* Premier League) and curves (age) automatically, needs no scaling and gives feature importances.

| Hyperparameter (scikit-learn) | Controls | Values to try |
| --- | --- | --- |
| `n_estimators` | Number of trees | 100, 300, 500 |
| `max_depth` | Max tree depth (lower = less overfitting) | 5, 10, 20, None |
| `min_samples_leaf` | Min players per leaf | 1, 5, 10 |
| `max_features` | Features considered per split | `sqrt`, 0.5, all |

Optional: a gradient boosting model for comparison.

### Why three levels

Only the **Random Forest** is used by the app to predict. The baseline and the linear regression are **benchmarks**:

- Baseline vs. models → does machine learning add anything?
- Linear regression vs. Random Forest → is a complex model needed, or is a simple one enough?

If the Random Forest does not beat the linear regression, that is also a valid conclusion.

**To do (presentation + app "Model insights" page):** explain how each of the three levels works, with visuals:
- Diagram of the three levels (baseline → linear → Random Forest) and the question each comparison answers.
- Baseline: bar chart of median value by position and league.
- Linear regression: fitted line or coefficient chart (e.g. "+1 goal per 90 → ×1.35 value").
- Random Forest: one small example tree + "many trees vote → average" diagram.
- Bar chart comparing the metrics of the three levels.

## Experiments and metrics

### Data split (by time)

Train on older seasons, evaluate on the most recent ones, as the model would be used in reality.

| Set | Seasons (approx.) | Use |
| --- | --- | --- |
| Train | First available → 3 seasons before the last | Fit the models |
| Validation | Second-to-last full season | Choose hyperparameters |
| Test | Last full season | Final result, measured once |

Hyperparameter search uses time-based cross-validation (`TimeSeriesSplit`) or grouped by player (`GroupKFold`), so the same player is never in both train and validation.

### Metrics

| Metric | Meaning | Scale |
| --- | --- | --- |
| R² | Share of value variation explained (1 = perfect, 0 = same as predicting the mean) | Log value |
| MAE | Mean absolute error | Euros |
| Relative MAE | "On average we are X % off" | Percent |
| RMSE | Like MAE but punishes large errors more | Log value and euros |

### Required outputs

- Predicted vs. real value plot (log scale) with the perfect-prediction diagonal.
- Learning curve: error vs. training set size (detects overfitting).
- Comparison table of all models and baselines with the same metrics.

### Further analysis

- **Feature importance:** with *permutation importance*.
- **Age curve:** partial dependence plot; should peak around 25–27.
- **Undervalued / overvalued players:** largest gap between predicted and real value in the test set; discuss 3–5 cases.
- **Error by group:** does the model fail more for goalkeepers, a given league or stars?

## Visual platform (Streamlit app)

A simple web app run locally with `streamlit run app.py`. It loads the trained model; it does not retrain.

| Page | What it shows |
| --- | --- |
| Player lookup | Search a player → profile, real vs. predicted value, under/overvalued label |
| Manual estimate | Form (age, position, league, minutes, goals…) → predicted value |
| Model insights | Predicted vs. real plot, feature importance, age curve, model comparison table |
| Top undervalued | Table of players with the largest positive gap in the test season |

## Program structure

```
semestral-work-ite/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Read and validate the CSV tables
│   ├── features.py         # Joins, aggregation, age, per-90 stats, filters
│   ├── models.py           # Baselines, linear regression, Random Forest
│   ├── evaluation.py       # Metrics, plots, undervalued players
│   └── predictor.py        # Load the saved model and predict for one player
├── app.py                  # Streamlit visual platform
├── main.py                 # CLI: train / evaluate / predict
├── tests/
│   ├── __init__.py
│   ├── fixtures/           # Small handmade CSV files
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_predictor.py
├── docs/
│   ├── diagrams/           # PlantUML sources + PNG exports
│   └── tests.md
├── test_main.py            # Run all tests
├── README.md
├── requirements.txt        # pandas, scikit-learn, matplotlib, streamlit, joblib, pytest
├── .gitignore              # data/, models/, __pycache__/, .venv/
└── LICENSE                 # MIT
```

**Diagrams (UML):**
- Activity diagram: training pipeline (load → join → features → train → evaluate → save).
- Component diagram: modules and their dependencies.
- Sequence diagram: user → app → predictor → model when requesting a prediction.

## Testing strategy

ML results are not exact, so tests check **deterministic logic** and **properties** of the model, not exact predicted values.

| Level | What is tested | Example |
| --- | --- | --- |
| Unit | Age, per-90 stats, log transform and its inverse, valuation-date matching, baselines | 900 min and 2 goals → 0.2 goals per 90 |
| Unit (boundary) | 0 minutes, missing birth date, player with no valuation, unknown position | Raises a clear error or is filtered out |
| Integration | Fixture CSVs → final player-season table | Expected row count, no duplicated players, no future valuations (no leakage) |
| Model | Fixed `random_state` on a tiny dataset | Prediction is positive and finite; Random Forest beats baseline 1 |
| System | `main.py predict` and the predictor from end to end | Known fixture player → output in the expected format |
| Smoke | The app module imports and loads the model | No crash on start |

All tests run with `python test_main.py`; coverage is measured with `pytest-cov` and shown in the presentation.

## Risks and limitations

| Risk or limitation | Consequence | Mitigation |
| --- | --- | --- |
| Transfermarkt value is an expert estimate | The "truth" is subjective | State it in the README |
| Wrong date join between season and valuation | Wrong or duplicated data | Row-count tests and manual checks of a few players |
| Goalkeepers and defenders rarely score | Attacking stats do not describe them | Use position as a feature or exclude goalkeepers |
| No advanced stats (passes, dribbles, defending) | Accuracy ceiling | Future work: join with FBref |
| Market inflation over the years | Same performance worth more recently | Add the season year as a feature |
| Contract only known for the present | Invalid feature for old seasons | Use only where valid or drop |
| Data leakage (max value, future values) | Misleadingly good results | Only use information available at season end; tested |
| Project scope larger than a "very simple program" | Risk of not finishing on time | Core (data → Random Forest → app) first; optional parts last |

## Plan (5 weeks)

| Week | Goal | Deliverable |
| --- | --- | --- |
| 1 | Repo setup, data download, UML diagrams, data quality report, EDA | README skeleton, `docs/diagrams/`, `docs/data_quality.md` (committed before the code), `data_loader.py` + tests |
| 2 | Player-season table: joins, aggregation, features, filters | `features.py` + tests |
| 3 | Baselines, linear regression, Random Forest, tuning, evaluation | `models.py`, `evaluation.py`, `main.py` + tests |
| 4 | Streamlit app and predictor | `predictor.py`, `app.py` + tests |
| 5 | Docs, tests.md, coverage, license, release `v1.0.0`, slides | Final release + presentation |

### Next steps

- [x] Confirm with the teacher that this scope is acceptable for the semestral work (approved 2026-10-21)
- [ ] Prepare the visual explanation of the three method levels (see *Why three levels*)
- [x] Create the `semestral-work-ite` repository with README and weekly plan
- [x] Download the Kaggle dataset and record version and date
- [x] Check the real columns of each table and the season range (see `docs/data_quality.md`)
- [ ] Export the UML diagrams in `docs/diagrams/` to PNG and reference them in the README
- [ ] Decide whether to exclude goalkeepers or keep them with position as a feature
