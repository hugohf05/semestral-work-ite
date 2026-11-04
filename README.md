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
License: CC0 (Public Domain). The data is **not** included in this repository.

- **Data coverage:** updates are paused; the dataset is current to 2026-07-06 (`player_valuations` ends on 2026-06-12, `appearances` on 2026-06-28). Last complete season: 2025/26.
- **Download date:** *TBD*

### Getting the data

**Option A — Kaggle API (used in this project).** The token is a secret: never commit it or share it.

1. Install the client (inside the virtual environment): `pip install kaggle`
2. On kaggle.com go to *Settings → API → Create New Token* and copy the token (it is shown only once).
3. Save it outside the repository with restricted permissions:
   ```bash
   mkdir -p ~/.kaggle
   pbpaste > ~/.kaggle/access_token   # macOS: pastes the clipboard; on Linux use: echo "<token>" > ...
   chmod 600 ~/.kaggle/access_token
   ```
   Alternative: set the `KAGGLE_API_TOKEN` environment variable.
4. Download and unzip into `data/raw/`:
   ```bash
   kaggle datasets download davidcariboo/player-scores -p data/raw --unzip
   ```

**Option B — manual.** Download the ZIP from the Kaggle page and run `unzip ~/Downloads/archive.zip -d data/raw`.

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
