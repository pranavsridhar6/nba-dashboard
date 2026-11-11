# Methodology

How every number in the dashboard is built, how it was checked, and where it falls short.
All data is public. Seasons covered: 2023-24 through 2026-27 (in progress).

---

## 1. Data sources

| Data | Source | Refresh |
|---|---|---|
| Games, box scores, play-by-play, schedule | NBA Stats API via `nba_api` | Daily, incremental |
| Shot locations | NBA play-by-play coordinates (validated against the official shot chart endpoint) | Daily |
| Closest-defender shot splits | NBA player tracking (`PlayerDashPtShots`) | Daily for the current season |
| Contracts, cap holds, exceptions | ShamSports team salary pages (robots.txt checked, 6-second delay, max one download per team per day) | Daily |
| Cap thresholds | NBA official 2026-27 figures (Bi-annual from Hoops Rumors) | Yearly |
| College player stats | Barttorvik (public CSV export) | Daily during the college season |
| Draft history, NBA season totals | NBA Stats API | Daily |

Source mistakes are corrected in `reference/contract_overrides.csv` so fixes survive every refresh.

---

## 2. Validation checks (run after every refresh)

| Check | Result |
|---|---|
| 76ers: regular-season games per finished season | 82 / 82 / 82 |
| Player points add up to team points (76ers, every game) | 100% |
| Shot rows = box-score FGA, every team, every game | 100% (about 925,000 shots) |
| League: regular-season games per team, finished seasons | 82 for all 30 teams (after fixing neutral-site games, see below) |
| Stint seconds add up to game length | 100% of games |
| Stint points add up to final score | 99.5-99.9% of games (games that don't are excluded from RAPM) |
| Stints with a full 5-on-5 lineup | 97.8-99.3% |
| Contracts: parsed 2026-27 total = the source page's own total | 30 / 30 teams |
| Draft: drafted college players matched to their college season | 813 / 819 (99.3%) |

**A bug the checks caught:** the first league download paired home and away teams using "vs." and "@"
in the matchup text. Neutral-site games (Paris, Mexico City, NBA Cup semifinals) break that pattern and were
silently dropped, about 5 games per season. Oklahoma City showed 67-14 instead of 68-14. Fixed by pairing
the two team rows of each game by game ID.

---

## 3. Stints (who was on the floor)

Play-by-play logs substitutions but not who starts each quarter.
- Period 1 starters come from the box score.
- Later periods: a player started the period if his first appearance in it is anything other than being subbed in.
- Fallback: players on the floor at the end of the previous period who did nothing all period.
- Substitution names are matched to player IDs by exact name, first-name prefix ("Ja. Green"), first name,
  close spelling ("Pöltl" / "Poeltl"), and, when two teammates share a last name, by who appears next in the game.

Each stint records the 10 players, seconds, points, and a possession estimate (FGA - OREB + TOV + 0.44 x FTA).

---

## 4. RAPM (Regularized Adjusted Plus-Minus)

**Model.** Every stint becomes two rows (each team on offense). Target: points per 100 possessions.
Columns: +1 for each offensive player, +1 for each defensive player (separate columns), plus home court.
Ridge regression, weighted by possessions.

**Choices**
- Window: selected season + the season before it; the earlier season counts half.
- Regular season only; garbage time removed (Cleaning the Glass rule: 4th quarter, lead of 25 / 20 / 10
  depending on time left, and no more than two starters on the floor for either team). About 3% of minutes.
- Regularization strength chosen by 5-fold cross-validation, split by game.
- **Box-score prior:** instead of shrinking every player toward 0 (average), each player is shrunk toward what
  his box score predicts. The prior model is fit on box stats per 100 possessions, themselves pulled toward the
  league average for small samples. Fit quality: R² 0.47 for offense, 0.27 for defense.
- Uncertainty: 40 bootstrap refits (games resampled), shown as a 90% **model range**. Because ridge shrinks
  estimates, this range is narrower than a true confidence interval.
- "Reliable" = at least 4,000 possessions (offense + defense).

**Known weakness: the box prior underrates defense-first role players.** Box scores capture defense poorly
(R² 0.27), so players whose value is defensive and off the ball get a prior near 0, and the final rating is
pulled down. Examples: OG Anunoby (2023-24: basic RAPM +5.3, with prior +2.4), Josh Hart, Alex Caruso,
Herbert Jones. The Player Metrics page shows **RAPM Basic** (no prior) next to the final RAPM so the
difference is visible. Treat a large gap between the two as a flag, not an answer.

**Other limits**
- Teammates who share most of their minutes split credit unpredictably (shown as "Top teammate" and share).
- Regular season only, so playoff impact isn't measured.
- Opponent three-point luck is not adjusted.

---

## 5. Lineups, rim protection and spacing

- **Lineup simulator:** stints summed for every 2-5 player combination, by season type, opponent defense tier
  (top / middle / bottom 10 by defensive rating) and garbage time. Ratings are computed from summed points and
  possessions, never by averaging ratings.
- **Rim on/off (deterrence proxy):** opponents' share of shots within 4 feet, and their FG% there, with the
  player on vs off the floor. Raw, not adjusted for teammates.
- **Spacing proxy:** teammates' rim-attempt share and eFG% with the player on vs off (his own shots excluded),
  plus "tightly guarded %": the share of his 10+ ft jumpers with a defender within 4 ft. That last number mixes
  how much defenses respect a shooter with how well he creates separation; creators score low for the second reason.
- All on/off numbers: games the player played, regular season, no garbage time, full lineups,
  at least 1,000 possessions. True defender-position tracking is proprietary; these are public proxies.

---

## 6. Salary cap

- Contracts for all 30 teams parsed from ShamSports: salary per season, options, partial guarantees, two-way
  and Exhibit 10 deals.
- Tier from official thresholds: under cap / over cap / over tax / over first apron / over second apron.
- **Hard caps are inferred** (not listed by the source) from exception use, following the CBA: using the
  bi-annual, or more of the non-taxpayer mid-level than the taxpayer amount, hard-caps a team at the first apron;
  using the taxpayer mid-level hard-caps it at the second apron. Hand overrides in `reference/team_cap_settings.csv`
  always win (e.g. a hard cap from a sign-and-trade the exceptions can't reveal).
- Not modeled yet: luxury-tax bill, cap holds in next summer's space, trade matching ("what-if" tools).

---

## 7. Draft translator

- **Target:** years 1-3 NBA box-score impact (same scale as RAPM: 0 = average player), possession-weighted,
  pulled toward replacement level (-2) with 500 pseudo-possessions so players who barely played count as busts
  instead of being dropped.
- **Inputs:** college BPM (overall / offense / defense), usage, true shooting, rebound / assist / turnover /
  steal / block rates, free-throw rate, 3-point rate and %, free-throw %, height, exact age at the draft,
  power conference, minutes share, high school recruiting rank.
- **Model:** ridge regression (chosen over gradient boosting on within-class ranking), validated one draft class
  at a time: each class predicted by a model that never saw it.

| 2010-2025 drafts, 759 players | Model | Draft order |
|---|---|---|
| Correlation with actual years 1-3 impact | 0.40 | 0.21 |
| Average error (pts / 100) | 0.64 | 0.73 |
| Ranking within each class (Spearman) | 0.38 | 0.29 |

- **What it values:** steals, blocks, defensive rebounding, efficiency, low turnovers, lower usage.
  It underrates high-usage creators, because the box-score impact it predicts does.
- Recruiting rank (available for 88% of draftees) added nothing once college stats were known.
- Choosing the better of two models on the same validation slightly flatters the winner.
- College players only (no international or G League). 2027 prospects use last season's stats until they've
  played 8 games this season.

---

## 8. What I'd like to build next

- What-if cap tools: trade salary matching, "can we sign X", next summer's cap space with holds.
- Opponent three-point luck adjustment in RAPM.
- A longer-history prior for RAPM (fit on more seasons than the four used here).
- International prospects in the draft translator.
