# Football Player Market Value Predictor

Estimate a football player's market value from their profile (age, position, league, contract) and one season of performance, and explore the results in a simple web app.

> **Status:** in development (ITE semestral work, CTU Prague, autumn 2026). Sections marked *TBD* are completed as the project grows.

## Purpose

Transfermarkt market values are hard to guess with a simple rule. This program:

1. Builds a player-season table from the public Transfermarkt dataset.
2. Trains and compares three levels of prediction methods:
   - **Baseline:** median value by position and league.
   - **Linear regression:** simple, interpretable benchmark.
   - **Random Forest:** main model, used by the app.
3. Shows predictions, real values and model insights in a Streamlit app.

**Target audience:** football fans, fantasy-league players and data-analytics students.

## Data

Kaggle dataset [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) (`davidcariboo/player-scores`).
The data is **not** included in this repository. Version and download date: *TBD*.

## Installation

*TBD* — Python version, virtual environment and `pip install -r requirements.txt`.

## Usage

*TBD* — training (`python main.py ...`), app (`streamlit run app.py`), input/output format.

## Tests

*TBD* — run all tests with `python test_main.py`. Testing approach: [docs/tests.md](docs/tests.md).

## Limitations

*TBD*

## Development plan (5 weeks)

| Week | Goal | Done when |
| --- | --- | --- |
| 1 | Setup and data | Repo structure, data downloaded and explored, UML diagrams, `data_loader.py` + tests |
| 2 | Feature engineering | `features.py` builds the player-season table (joins, age, per-90 stats, filters) + tests, incl. no-leakage checks |
| 3 | Models and evaluation | Baselines, linear regression, tuned Random Forest, metrics and plots, `main.py` CLI + tests |
| 4 | Visual platform | `predictor.py` and Streamlit `app.py` (player lookup, manual estimate, model insights, top undervalued) + tests |
| 5 | Finish and present | `docs/tests.md`, coverage report, final README, release `v1.0.0`, slides incl. visual explanation of the three method levels |

## License

[MIT](LICENSE). The dataset has its own license; see the Kaggle page.
