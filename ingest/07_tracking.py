"""Step 7a: closest-defender shot splits for every 76ers player (NBA player tracking data).

For each player-season: how many of his shots came with the nearest defender
0-2 ft (very tight), 2-4 ft (tight), 4-6 ft (open) or 6+ ft (wide open).
Uses jump shots 10+ feet, where defender distance reflects how much a defense respects the shooter.
Finished seasons are fetched once; the current season is refetched each run.

Output: data/tracking/season=<season>/closest_defender.parquet
"""
import os
import time

import pandas as pd
from nba_api.stats.endpoints import playerdashptshots

from config import TEAM_ID, SEASONS, SLEEP_SECONDS
from utils import already_done, call_with_retry, table_path

MIN_MINUTES = 100        # skip players with almost no minutes for the 76ers that season
SET_NAME = "ClosestDefender10ftPlusShooting"


def fetch(pid, season):
    resp = call_with_retry(lambda: playerdashptshots.PlayerDashPtShots(
        team_id=TEAM_ID, player_id=pid, season=season,
        season_type_all_star="Regular Season", per_mode_simple="Totals", timeout=60))
    sets = {rs["name"]: rs for rs in resp.get_dict()["resultSets"]}
    rs = sets.get(SET_NAME)
    if rs is None or not rs["rowSet"]:
        return None
    df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    df["player_id"] = pid
    df["season"] = season
    return df


for season in SEASONS:
    out = table_path("tracking", "closest_defender.parquet", season)
    if already_done(out, season):
        print(f"{season}: already saved, skipping")
        continue
    box = pd.read_parquet(table_path("player_box", "player_box.parquet", season))
    phi = box[(box["team_id"] == TEAM_ID) & box["did_play"] & (box["game_id"].str[:3] == "002")]
    mins = phi.groupby("player_id")["minutes"].sum()
    players = mins[mins >= MIN_MINUTES].index.tolist()
    print(f"{season}: {len(players)} players")

    frames = []
    for i, pid in enumerate(players, 1):
        try:
            df = fetch(int(pid), season)
            if df is not None:
                frames.append(df)
            print(f"  [{i}/{len(players)}] {pid}: {0 if df is None else len(df)} rows")
        except Exception as e:
            print(f"  [{i}/{len(players)}] {pid}: FAILED ({e})")
        time.sleep(SLEEP_SECONDS)

    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        print(f"{season}: saved {out}")