# Football Player Market Value Predictor

*ITE semestral work — CTU Prague, autumn 2026 — individual project*

## Summary

**Project question:** can we estimate a football player's market value using only their profile (age, position, league) and their performance in one season?

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
| **Input (app)** | Player name from the dataset, **or** a manual profile: age, position, league, minutes, goals, assists |
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
| Profile | Age | Birth date vs. season date | EDA: correlation −0.39 with log value; median value falls after 23 (selection bias makes teenagers look expensive) |
| Profile | Position and sub-position | `players` | Forwards are usually worth more than goalkeepers |
| Profile | Height, preferred foot | `players` | Probably small effect; to check |
| Performance | Minutes and matches played | Sum of `appearances` | Shows whether the player is a starter |
| Performance | Goals and assists per 90 min | `appearances` | Compares players with different minutes |
| Performance | Cards | `appearances` | Probably small effect |
| Context | Club's league | `clubs` / `competitions` | The Premier League inflates values |
| Context | Plays in a European competition | `games` | Signal of a strong, visible club |

**Contract:** dropped. `players.csv` only stores the *current* contract, which is wrong for older seasons.

**Extra features** (starts, captain, team goals conceded, clean sheets) come from `game_lineups` and `club_games`; see [feature_sources.md](feature_sources.md). Decisions on leagues, valuation window and minimum minutes are in [data_quality.md](data_quality.md).

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
5. Handle missing values (height, foot): impute with the median or drop, and justify.
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

**Idea:** every feature pushes the predicted value up or down by a fixed amount, and the effects are added.

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n$$

- $\hat{y}$ is the predicted log10 value, $x_i$ are the features and $w_i$ their weights (coefficients).
- **Training** finds the weights that minimise the sum of squared errors $\sum (y - \hat{y})^2$ (ordinary least squares). There is one exact solution; nothing random is involved.
- **Preparation in our pipeline** (`make_linear_regression`): position and league are one-hot encoded (one 0/1 column per category); numeric features are standardised (mean 0, standard deviation 1) so the weights are comparable; `age²` is added so that a straight-line model can bend the age effect into a curve.

**How to read a weight.** We predict log10(value), so adding $w$ to the log means **multiplying the value by $10^{w}$**. Effects estimated on seasons 2012–2024, all other features fixed:

| Change | Effect on value |
| --- | --- |
| Playing in the Premier League instead of LaLiga | ×1.96 |
| Team gets +0.5 points per game | ×1.69 |
| +1,000 league minutes | ×2.07 |
| +900 minutes in European cups (about 10 games) | ×2.06 |
| +0.1 goals per 90 | ×1.12 |
| One season later (market inflation) | ×1.09 |
| Goalkeeper instead of forward | ×0.76 |

**Limitations:**
- Effects are **additive**: +0.1 goals per 90 is worth ×1.12 for every player, whether he plays for the champion or for a relegated club.
- Curves must be added by hand (`age²`), and the shape is then forced: a parabola.

### 3. Random Forest (main method)

#### Step 1: one decision tree

A decision tree predicts with a series of yes/no questions about the features. Each final box (leaf) predicts the **mean log value of the training players that ended there**. A real tree trained on our data and limited to 3 levels (for illustration):

```
                    team_points_per_game <= 1.54 ?
                   /                               \
                 yes                                no  (strong club)
        age <= 32.0 ?                        age <= 31.6 ?
        /          \                         /            \
 matches <= 23.5?   age <= 33.9?     minutes <= 1,481?    age <= 34.6?
   /       \          /     \          /        \           /      \
 6.42     6.75      6.17   5.83      6.97      7.33       6.72     6.24
 2.6 M€   5.6 M€   1.5 M€  0.7 M€    9.3 M€   21.4 M€    5.2 M€   1.7 M€
```

**How the tree is built:** at each box, the algorithm tries every feature and every cut point and keeps the question that makes the two groups as **homogeneous** as possible (the smallest squared error around each group's mean). Then it repeats inside each group. Club strength is the first question because it separates expensive from cheap players best.

**Problem:** a deep tree keeps splitting until each leaf holds one or two players, so it memorises the training data (**overfitting**). Small changes in the data produce a very different tree.

#### Step 2: many different trees, averaged

A Random Forest trains **300 trees** and makes them different on purpose:

1. **Bootstrap:** each tree is trained on a random sample of the 23,903 training rows, drawn with replacement (some players appear twice, about a third not at all).
2. **Random features:** at every question, the tree may only choose among a random half of the features (`max_features=0.5`), so not all trees start with club strength.
3. **Average:** each tree has thousands of leaves (about 11,000 in our model), but a given player follows **one path** in each tree (every question sends him left *or* right) and lands in **exactly one leaf**. That leaf's value is the tree's prediction for him. The forest's prediction is the mean of these 300 leaf values.

```
 player ──► tree 1 → 7.10 ┐
       ──► tree 2 → 7.35 │
       ──► ...           ├──► mean = 7.22 ──► 10^7.22 ≈ 16.6 M€
       ──► tree 300 → 7.18┘
```

Each tree overfits in its own way; the errors point in different directions and **cancel out in the average**, while the real pattern, shared by all trees, remains.

#### Linear regression vs. Random Forest

| | Linear regression | Random Forest |
| --- | --- | --- |
| Shape of the model | One weighted sum | 300 trees of yes/no questions, averaged |
| Curves (e.g. age) | Only if added by hand (`age²`) | Learned automatically |
| Interactions (e.g. young **and** strong club) | No, each effect is fixed | Yes: a question inside another question |
| Needs scaling | Yes | No: a question like "age ≤ 32" works with any unit |
| Interpretation | One readable weight per feature | Permutation importance and age curve |
| Predicts outside the training range | Yes (can extrapolate) | No: a leaf is always a mean of real players, so it never predicts above the most expensive ones seen |
| Test R² (season 2025) | 0.776 | **0.809** |

The last two rows explain our results: the forest wins because it captures interactions, but only by 0.03 in R², so most of the signal is additive. The forest cannot predict beyond the values it has seen and averages towards typical players, which is one reason why stars are underestimated.

#### Hyperparameters and how they were chosen

Hyperparameters are settings chosen **before** training (the model does not learn them). We tested every combination of the grid in `models.PARAM_GRID` (18 forests of 200 trees), trained on 2012–2023 and scored by MAE (log10) on the validation season 2024. The test season was not used.

| Hyperparameter | What it controls | Values tried | Chosen |
| --- | --- | --- | --- |
| `max_depth` | Maximum number of questions from top to leaf; fewer = simpler trees | 10, 20, None (no limit) | **20** |
| `min_samples_leaf` | Minimum players in a leaf; more = smoother predictions | 1, 5, 10 | **1** |
| `max_features` | Share of features each question may choose from; fewer = more different trees | `sqrt` (≈ 4 of 19 columns), 0.5 | **0.5** |
| `n_estimators` | Number of trees; more = stabler mean, slower | not searched | **300** (200 during the search) |

Results (validation MAE in log10; full table printed by `python main.py train`):

| Setting | Best MAE | Worst MAE |
| --- | --- | --- |
| `max_features=0.5` | 0.1826 | 0.1980 |
| `max_features=sqrt` | 0.1932 | 0.2225 |
| `max_depth=10` | 0.1963 | 0.2225 |
| `max_depth=20` or None | 0.1826 | 0.2059 |

- `max_features` mattered most: with `sqrt` a tree often cannot ask about club strength or age at all.
- Depth 10 is too shallow; 20 and no limit are practically equal (0.1826 vs. 0.1829).
- `min_samples_leaf` changed little: averaging 300 trees already controls overfitting.
- Best vs. worst combination: 0.183 vs. 0.223. Adding the club features (week 4) improved the MAE from 0.275 to 0.183, much more than any tuning: **better data beat better settings**.
- `n_estimators` was not searched: more trees never make a forest worse, only slower; 300 is a usual value.

### Why three levels

Only the **Random Forest** is used by the app to predict. The baseline and the linear regression are **benchmarks**:

- Baseline vs. models → does machine learning add anything?
- Linear regression vs. Random Forest → is a complex model needed, or is a simple one enough?

If the Random Forest does not beat the linear regression, that is also a valid conclusion.

**To do (presentation + app "Model insights" page):** explain how each of the three levels works, with visuals:
- Diagram of the three levels (baseline → linear → Random Forest) and the question each comparison answers.
- Baseline: bar chart of median value by position and league.
- Linear regression: coefficient chart (e.g. "Premier League → ×1.96 value").
- Random Forest: one small example tree + "many trees vote → average" diagram.
- Bar chart comparing the metrics of the three levels.

## Experiments and metrics

### Data split (by time)

Train on older seasons, evaluate on the most recent ones, as the model would be used in reality. A random split would let the model see 2025 players while being tested on 2024 ones.

| Set | Seasons | Rows | Use |
| --- | --- | --- | --- |
| Train | 2012–2023 | 22,038 | Fit the models during the hyperparameter search |
| Validation | 2024 | 1,865 | Choose hyperparameters and features |
| Test | 2025 | 1,964 | Final result, measured **once** (`python main.py evaluate`) |

After the search, the final model is refitted on 2012–2024 (train + validation) with the chosen hyperparameters. A single validation season was used instead of cross-validation, to keep the search simple and fast.

### Metrics

For each player $i$ of the $n$ evaluated: $y_i$ = real log10 value, $\hat{y}_i$ = predicted log10 value, error $e_i = \hat{y}_i - y_i$. In euros: $v_i = 10^{y_i}$ and $\hat{v}_i = 10^{\hat{y}_i}$. All metrics are in `src/evaluation.py` (`regression_metrics`).

| Metric | Formula | What it measures | Test 2025 (RF) | How to read it |
| --- | --- | --- | --- | --- |
| **R²** (coefficient of determination) | $1 - \dfrac{\sum e_i^2}{\sum (y_i - \bar{y})^2}$ | Share of the variation in (log) value explained by the model, compared with always predicting the mean $\bar{y}$ | 0.809 | 1 = perfect, 0 = no better than the mean, < 0 = worse than the mean. The model removes 81 % of the error of the "always the mean" guess |
| **MAE log** (mean absolute error) | $\dfrac{1}{n}\sum \lvert e_i \rvert$ | Average size of the error in log10 | 0.186 | $10^{0.186} \approx 1.53$: on average the prediction is off by a factor of about 1.5 |
| **RMSE log** (root mean squared error) | $\sqrt{\dfrac{1}{n}\sum e_i^2}$ | Like MAE, but squaring makes large errors count more | 0.236 | Always ≥ MAE; the bigger the gap with MAE, the more the error comes from a few big misses |
| **MAE €** | $\dfrac{1}{n}\sum \lvert \hat{v}_i - v_i \rvert$ | Average error in euros | 5.8 M€ | Easy to understand but dominated by stars (a 50 M€ miss on one star outweighs dozens of small misses); not the main metric |
| **Median absolute % error** | $\text{median}\left(\dfrac{\lvert \hat{v}_i - v_i \rvert}{v_i}\right) \times 100$ | Relative error of the typical player | 32.8 % | Half of the players are predicted within ±33 % of their real value. The median ignores the few extreme cases |
| **Within ×2** | $\dfrac{1}{n}\,\#\{\, i : \lvert e_i \rvert \le \log_{10} 2 \,\} \times 100$ | Share of players with prediction between half and double the real value | 80.9 % | 4 out of 5 players are predicted within a factor of 2 |

**Why the main metrics are on the log scale:** an error of 2 M€ is huge for a 1 M€ player and negligible for a 100 M€ one. On the log scale an error of 0.3 means "×2 off" for every player, which matches how the model is trained.

**Error by value range** (`error_by_value_range`): the mean signed error per group shows *direction*. Positive = overestimated, negative = underestimated. Cheap players are overestimated and stars underestimated (regression to the mean).

### Required outputs

- Predicted vs. real value plot (log scale) with the perfect-prediction diagonal.
- Comparison table of all models and baselines with the same metrics.

### Further analysis

- **Feature importance:** with *permutation importance*.
- **Age curve:** partial dependence plot; the raw EDA median falls with age (selection bias), so check what the model learns once minutes and league are taken into account.
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
| Contract only known for the present | Invalid feature for old seasons | Dropped (decided) |
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
- [x] Export the UML diagrams in `docs/diagrams/` to PNG and reference them in the README
- [ ] Decide whether to exclude goalkeepers or keep them with position as a feature
