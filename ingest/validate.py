"""Checks the three tables agree with each other, per season. Rerun after every refresh."""
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

print(f"\n{sum(results)}/{len(results)} checks passed")