"""Checks the three tables agree with each other. Rerun after every refresh."""
import pandas as pd

from config import TEAM_ID
from utils import table_path

games = pd.read_parquet(table_path("games", "games.parquet"))
box = pd.read_parquet(table_path("player_box", "player_box.parquet"))
shots = pd.read_parquet(table_path("shots", "shots.parquet"))

results = []


def check(name, passed, detail=""):
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")
    results.append(passed)


# 1. Game counts
counts = games["season_type"].value_counts()
rs, po = counts.get("Regular Season", 0), counts.get("Playoffs", 0)
check("82 regular season games", rs == 82, f"(found {rs})")
check("6 playoff games", po == 6, f"(found {po})")

# 2. Player points add up to team points
phi = box[box["team_id"] == TEAM_ID]
pts = games.set_index("GAME_ID")[["PTS"]].join(phi.groupby("game_id")["pts"].sum().rename("box_pts"))
bad = pts[pts["PTS"] != pts["box_pts"]]
check("player points = team points", bad.empty, f"({len(bad)} mismatched games)")
if not bad.empty:
    print(bad.head(10))

# 3. Shot count per game = team FGA in box score
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