# Project log

Problems found during the project and what was decided. Newest entries at the bottom; add one every time something unexpected appears.

| When | Problem | Decision / status |
| --- | --- | --- |
| 2026-10-14 | The first project idea was written for another course (pairs, LaTeX report, no tests). ITE asks for a program with tests, docs, license and a release | Rewrote the proposal for ITE (individual, README, tests, UML, Streamlit app) |
| 2026-10-21 | Scope (machine learning + web app) is larger than the "very simple program" in the brief | The teacher approved the project. Risk kept in the project description: core first, optional parts last |
| Week 1 | Kaggle dataset updates are paused (current to 2026-07-06; valuations end 2026-06-12) | The dataset is frozen; last complete season is 2025/26 |
| Week 1 | Kaggle API token format changed (`~/.kaggle/access_token` instead of `kaggle.json`) | README documents the current method |
| Week 1 | `players.csv` contains the current and the all-time highest market value | Data leakage: columns are never loaded (tested) |
| Week 1 | `appearances` only covers 2012–2025 while valuations start in 2000 | Player-seasons start in 2012 |
| Week 1 | 31 domestic leagues, not only the main European ones | **Decided:** top 5 (`ES1`, `GB1`, `IT1`, `L1`, `FR1`) |
| Week 1 | Valuations in May–June are scarce in 2014 and 2020 (COVID) | **Decided:** last valuation between 1 April and 31 July; rows without one are dropped (coverage per window in `data_quality.md`) |
| Week 1 | A July valuation can reflect a summer transfer (new club and league); 14.2 % of valuations have no league | **Decided:** league taken from `appearances` (league with most minutes in the season) |
| Week 1 | `contract_expiration_date` only holds the current contract | **Decided:** contract feature dropped; listed as a limitation |
| Week 1 | Per-90 statistics are unstable with few minutes | **Decided:** keep players with at least 450 minutes (about 1,950 per season in the top 5); threshold configurable and compared with 200 and 900 in the EDA |
| Week 1 | Only goals, assists, cards and minutes are available; other positions need other statistics. FBref removed its advanced stats in January 2026 | See [feature_sources.md](feature_sources.md): core = extra tables of the same dataset, Understat optional |
| Week 1 | PlantUML is not installed locally, so the `.puml` files could not be checked before exporting | Exported to PNG with an external tool; all three compiled without errors; images referenced in the README |
| Week 1 | Terminal prompt shows `(base)` (conda) and `(.venv)` | Harmless; always install with `pip` inside `.venv` |
| Week 1 (EDA) | Median value peaks at 17–18 years, not at 25–27 as assumed: only exceptional teenagers reach 450 minutes in a top-5 league (selection bias) | Hypothesis rejected; age kept as a feature. Season added as a feature (median value 3 M€ in 2012 → 8 M€ in 2025). See `notebooks/eda.ipynb` |
| Week 2 (features) | Season 2013 has only 1,043 player-seasons (about 1,900 in the others), the same sparse May–June valuations seen for 2014 | Kept; listed as a limitation |
| Week 3 (EDA) | Values look low compared with headline transfers: median 5 M€ (8 M€ in 2025); only 3.8 % of rows are ≥ 50 M€ and 0.5 % ≥ 100 M€. Market value is not a transfer fee, and the table includes every player with ≥ 450 minutes | **Decided:** keep as is (`season` is a feature, target in log). The model will probably underestimate stars (few examples); measure error by value range and list it as a limitation |
