"""Step 9a: team-perspective tables for all 30 teams, from league data.

  data/league/team_games/season=<s>/team_games.parquet   every game twice (once per team), plus upcoming games
  data/league/game_shots/season=<s>/game_shots.parquet   both teams' field goal attempts in every game
These replace the 76ers-only games, schedule and game_shots tables on the team pages.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import SEASONS, CURRENT_SEASON        # noqa: E402
from utils import table_path                       # noqa: E402

PBP_COLS = ["game_id", "actionNumber", "period", "clock", "teamId", "teamTricode", "personId",
            "xLegacy", "yLegacy", "shotDistance", "actionType", "subType", "shotValue"]


def both_sides(df, h_id, h, a_id, a, h_pts, a_pts):
    """One row per game -> two rows: home team's view and away team's view."""
    home = pd.DataFrame({"team_id": df[h_id], "team": df[h], "opponent_id": df[a_id], "opponent": df[a],
                         "is_home": True, "team_pts": df[h_pts], "opp_pts": df[a_pts]})
    away = pd.DataFrame({"team_id": df[a_id], "team": df[a], "opponent_id": df[h_id], "opponent": df[h],
                         "is_home": False, "team_pts": df[a_pts], "opp_pts": df[h_pts]})
    keep = [c for c in ("game_id", "game_date", "season", "season_type") if c in df.columns]
    return pd.concat([pd.concat([df[keep].reset_index(drop=True), home.reset_index(drop=True)], axis=1),
                      pd.concat([df[keep].reset_index(drop=True), away.reset_index(drop=True)], axis=1)],
                     ignore_index=True)


def team_games(season):
    g = pd.read_parquet(table_path("league/games", "games.parquet", season))
    played = both_sides(g, "home_team_id", "home_team", "away_team_id", "away_team", "home_pts", "away_pts")
    played["status"] = "Final"
    played["result"] = np.where(played["team_pts"] > played["opp_pts"], "W", "L")
    parts = [played]
    sp = table_path("league/schedule", "schedule.parquet", season)
    if season == CURRENT_SEASON and os.path.exists(sp):
        s = pd.read_parquet(sp)
        s = s[~s["game_id"].isin(set(g["game_id"]))]             # not yet in the played list
        up = both_sides(s, "home_team_id", "home_team", "away_team_id", "away_team", "home_score", "away_score")
        up["status"] = up["game_id"].map(s.set_index("game_id")["game_status"])
        up.loc[up["status"] != "Final", ["team_pts", "opp_pts"]] = np.nan
        up["result"] = None
        parts.append(up)
    out = pd.concat(parts, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.normalize()
    out["matchup"] = out["team"] + np.where(out["is_home"], " vs ", " @ ") + out["opponent"]
    out["team_game_key"] = out["game_id"] + "-" + out["team"]
    return out.sort_values(["team", "game_date"]).reset_index(drop=True)


def game_shots(season):
    pbp = pd.read_parquet(table_path("league/pbp", "pbp.parquet", season), columns=PBP_COLS)
    box = pd.read_parquet(table_path("league/player_box", "player_box.parquet", season),
                          columns=["game_id", "player_id", "player_name", "team", "fga"])
    s = pbp[pbp["actionType"].isin(["Made Shot", "Missed Shot"])].copy()
    for c in ("teamId", "personId", "xLegacy", "yLegacy", "shotDistance", "shotValue", "actionNumber"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    names = box.drop_duplicates("player_id").set_index("player_id")["player_name"]
    out = pd.DataFrame({
        "game_id": s["game_id"], "GAME_EVENT_ID": s["actionNumber"].astype("Int64"), "season": season,
        "period": s["period"], "clock": s["clock"], "team_id": s["teamId"].astype("Int64"),
        "team": s["teamTricode"], "PLAYER_ID": s["personId"].astype("Int64"),
        "PLAYER_NAME": s["personId"].map(names), "SHOT_MADE_FLAG": (s["actionType"] == "Made Shot").astype(int),
        "SHOT_VALUE": s["shotValue"], "ACTION_TYPE": s["subType"], "SHOT_DISTANCE": s["shotDistance"],
        "x_ft": s["xLegacy"] / 10, "y_ft": s["yLegacy"] / 10})
    fga = box.groupby(["game_id", "team"])["fga"].sum()
    cnt = out.groupby(["game_id", "team"]).size()
    both = pd.concat([fga.rename("box"), cnt.rename("shots")], axis=1).fillna(0)
    return out, (both["box"] == both["shots"]).mean()


def main():
    for season in SEASONS:
        tg = team_games(season)
        tg.to_parquet(table_path("league/team_games", "team_games.parquet", season), index=False)
        shots, match = game_shots(season)
        shots.to_parquet(table_path("league/game_shots", "game_shots.parquet", season), index=False)

        played = tg[tg["status"] == "Final"]
        per_team = played[played["season_type"] == "Regular Season"].groupby("team").size()
        print(f"{season}: {played['game_id'].nunique():,} games played, "
              f"{(tg['status'] != 'Final').sum() // 2} upcoming | "
              f"regular-season games per team {per_team.min()}-{per_team.max()} | "
              f"{len(shots):,} shots, FGA match {match:.1%} of team-games")
        phi = played[(played["team"] == "PHI") & (played["season_type"] == "Regular Season")]
        print(f"   check PHI: {(phi['result'] == 'W').sum()}-{(phi['result'] == 'L').sum()}")


if __name__ == "__main__":
    main()