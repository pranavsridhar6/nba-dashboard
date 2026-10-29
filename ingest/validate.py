"""Checks the tables agree with each other, per season. Rerun after every refresh."""
import pandas as pd

from config import TEAM_ID, SEASONS, CURRENT_SEASON
from utils import table_path

results = []


def check(name, passed, detail=""):
    print(f"  {'PASS' if passed else 'FAIL'}  {name}  {detail}")
    results.append(passed)


for season in SEASONS:
    print(f"\n{season}")
    games = pd.read_parquet(table_path("games", "games.parquet", season))
    box = pd.read_parquet(table_path("player_box", "player_box.parquet", season))
    shots = pd.read_parquet(table_path("shots", "shots.parquet", season))

    # 1. Game counts (only a finished season must have exactly 82)
    counts = games["season_type"].value_counts()
    rs, po = counts.get("Regular Season", 0), counts.get("Playoffs", 0)
    if season == CURRENT_SEASON:
        print(f"  INFO  in progress: {rs} regular season games played so far")
    else:
        check("82 regular season games", rs == 82, f"(found {rs})")
    print(f"  INFO  {po} playoff games")

    # 2. Player points add up to team points
    phi = box[box["team_id"] == TEAM_ID]
    pts = games.set_index("GAME_ID")[["PTS"]].join(
        phi.groupby("game_id")["pts"].sum().rename("box_pts"))
    bad = pts[pts["PTS"] != pts["box_pts"]]
    check("player points = team points", bad.empty, f"({len(bad)} mismatched games)")
    if not bad.empty:
        print(bad.head(10))

    # 3. Shot rows per game = team FGA in box score
    cmp = pd.concat([phi.groupby("game_id")["fga"].sum().rename("box_fga"),
                     shots.groupby("GAME_ID").size().rename("shot_rows")], axis=1)
    bad = cmp[cmp["box_fga"] != cmp["shot_rows"]]
    check("shot rows = box score FGA", bad.empty, f"({len(bad)} mismatched games)")
    if not bad.empty:
        print(bad.head(10))

    # 4. Every game_id in the other tables exists in games
    ids = set(games["GAME_ID"])
    check("box score game_ids exist in games", set(box["game_id"]) <= ids)
    check("shot game_ids exist in games", set(shots["GAME_ID"]) <= ids)

    # 5. Current season only: schedule agrees with games
    if season == CURRENT_SEASON:
        sched = pd.read_parquet(table_path("schedule", "schedule.parquet", season))
        rs_sched = sched[sched["season_type"] == "Regular Season"]
        n_total = len(rs_sched)
        # 80 until December: the last 2 games are added after the NBA Cup group stage
        check("schedule has 80-82 regular season games", 80 <= n_total <= 82, f"(found {n_total})")

        final_ids = set(sched.loc[sched["game_status"] == "Final", "game_id"])
        missing = final_ids - ids
        check("every Final game in schedule is in games", not missing,
              f"(missing {sorted(missing)})" if missing else "")

        scores = sched[sched["game_status"] == "Final"].set_index("game_id")[["team_score"]].join(
            games.set_index("GAME_ID")[["PTS"]], how="inner")
        bad = scores[scores["team_score"] != scores["PTS"]]
        check("schedule scores = games scores", bad.empty, f"({len(bad)} mismatched games)")

        upcoming = (sched["game_status"] == "Scheduled").sum()
        print(f"  INFO  {upcoming} upcoming games")

print(f"\n{sum(results)}/{len(results)} checks passed")