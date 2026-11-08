"""League overview: one row per team per season (regular season), for the main page.

Record and points from league games, ratings and defense rank from team_ratings,
and for the current season, payroll from reference/team_payroll.csv (hand-curated, ShamSports).

Output: data/league/overview/season=<season>/team_overview.parquet
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import SEASONS, CURRENT_SEASON        # noqa: E402
from utils import table_path                       # noqa: E402


def record(games):
    reg = games[games["season_type"] == "Regular Season"]
    home = pd.DataFrame({"team_id": reg["home_team_id"], "team": reg["home_team"],
                         "pf": reg["home_pts"], "pa": reg["away_pts"]})
    away = pd.DataFrame({"team_id": reg["away_team_id"], "team": reg["away_team"],
                         "pf": reg["away_pts"], "pa": reg["home_pts"]})
    t = pd.concat([home, away])
    t["win"] = (t["pf"] > t["pa"]).astype(int)
    out = t.groupby(["team_id", "team"]).agg(games=("win", "size"), wins=("win", "sum"),
                                             ppg=("pf", "mean"), opp_ppg=("pa", "mean")).reset_index()
    out["losses"] = out["games"] - out["wins"]
    out["win_pct"] = out["wins"] / out["games"]
    return out


def main():
    th = pd.read_csv(os.path.join("reference", "cap_thresholds.csv")).set_index("season")
    pay = pd.read_csv(os.path.join("reference", "team_payroll.csv"))
    pay["total_salary"] = pd.to_numeric(pay["total_salary"], errors="coerce")

    for season in SEASONS:
        games = pd.read_parquet(table_path("league/games", "games.parquet", season))
        ratings = pd.read_parquet(table_path("league/team_ratings", "team_ratings.parquet", season),
                                  columns=["team_id", "ortg", "drtg", "net", "def_rank", "def_tier"])
        t = record(games).merge(ratings, on="team_id", how="left")
        t["net_rank"] = t["net"].rank(ascending=False, method="min").astype("Int64")
        t["season"] = season

        if season in th.index:
            line = th.loc[season]
            p = pay[pay["season"] == season][["team", "total_salary"]]
            t = t.merge(p, on="team", how="left")
            t["vs_tax"] = t["total_salary"] - line["tax_level"]
            t["cap_tier"] = np.select(
                [t["total_salary"].isna(), t["total_salary"] < line["salary_cap"],
                 t["total_salary"] < line["tax_level"], t["total_salary"] < line["first_apron"],
                 t["total_salary"] < line["second_apron"]],
                ["Unknown", "Under cap", "Over cap", "Over tax", "Over first apron"], "Over second apron")
        else:
            t["total_salary"], t["vs_tax"], t["cap_tier"] = np.nan, np.nan, None

        t = t.sort_values(["win_pct", "net"], ascending=False).reset_index(drop=True)
        out = table_path("league/overview", "team_overview.parquet", season)
        t.round({"win_pct": 3, "ppg": 1, "opp_ppg": 1}).to_parquet(out, index=False)

        missing = t.loc[t["total_salary"].isna(), "team"].tolist() if season == CURRENT_SEASON else []
        print(f"{season}: {len(t)} teams, top: {t.iloc[0]['team']} {t.iloc[0]['wins']}-{t.iloc[0]['losses']}"
              + (f" | payroll missing for {len(missing)}: {missing}" if missing else ""))


if __name__ == "__main__":
    main()