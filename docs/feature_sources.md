# Position-specific features and data sources

## Problem

The model predicts the market value of players in every position, but the same statistic means different things per position (goals matter for a forward, not for a goalkeeper). The core dataset only has **goals, assists, cards and minutes** per player.

## What should be measured per position

Legend: ●●● very relevant, ●● relevant, ● minor, – not applicable. These are **hypotheses**; they will be checked with feature importance per position.

| Position | Goals and assists /90 | Shots, xG, xA, key passes | Passing and dribbling | Defensive actions (tackles, interceptions, aerial duels) | Goalkeeping (saves, PSxG) | Team goals conceded / clean sheets | Minutes and starts | Cards |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Goalkeeper | – | – | ● | ● | ●●● | ●●● | ●●● | ● |
| Defender | ● | ● | ●● | ●●● | – | ●●● | ●●● | ●● |
| Midfielder | ●● | ●● | ●●● | ●● | – | ● | ●●● | ● |
| Forward | ●●● | ●●● | ●● | ● | – | – | ●●● | ● |
| **Available?** | ✅ `appearances` | 🔴 external | 🔴 external | 🔴 external | 🔴 external | 🟡 derivable from `club_games` | ✅ minutes; 🟡 starts from `game_lineups` | ✅ `appearances` |

✅ already loaded · 🟡 computable from tables of the same dataset · 🔴 needs another source.

## Options to get more statistics

| Option | Adds | Problems |
| --- | --- | --- |
| **A. Extra tables of the same dataset** (🟡) | From `game_lineups`: starts vs. substitute appearances, captain, position played. From `club_games`: team goals conceded, clean sheets, wins, points per game in the matches the player appeared in | Team-level proxies, not individual stats (a clean sheet depends on the whole team). Same IDs, so **no name matching** needed |
| **B. Understat** (external) | xG, xA, shots, key passes for the top 5 leagues, from season 2014/15 | No defensive or goalkeeping stats. No official API (third-party scrapers only); terms of use not verified. Needs matching players by name and club (accents, nicknames). Seasons 2012–2013 would have no values |
| **C. FBref** | Was the full source | Opta advanced stats (xG, xA, PSxG, progressive actions…) were removed on 2026-01-20; only basic stats remain, which we already have |
| **D. Video-game ratings (FIFA / EA FC)** | Attribute ratings, including goalkeeping | They summarise player reputation and are close to the market value itself (circular); **rejected** |

## Decision (agreed 2026-11-04)

1. **Core: option A.** Fits the "very simple program" scope, is reproducible and explainable.
2. **Optional extension: option B**, only if time remains and the terms of use allow it, only for forwards and midfielders.
3. **Documented limitation:** no free source provides individual defensive and goalkeeping statistics for 2012–2025.
4. **One Random Forest with position as a feature** (it learns position-specific use of the statistics through its splits). Error is reported per position; if goalkeepers are clearly worse, exclude them or add team-level features and say so in the limitations.

Sources: [FBref & Stathead data update](https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/), [Understat scraper listing](https://apify.com/mirthful_radish/understat-xg-football-scraper).
