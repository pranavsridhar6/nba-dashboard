"""Step 9a: closest-defender shot splits for every player in the league (for the spacing table on any team).

One call per player per season (players with 100+ regular-season minutes), team_id=0 = all his teams combined.
Checkpointed like 06_league.py: saves every 50 players and on Ctrl+C; rerun to resume.
Output: data/league/tracking/season=<season>/closest_defender.parquet
"""
import os
import time

import pandas as pd
from nba_api.stats.endpoints import playerdashptshots

from config import SEASONS, CURRENT_SEASON
from utils import table_path

MIN_MINUTES = 100
SET_NAME = "ClosestDefender10ftPlusShooting"
SLEEP, TIMEOUT, SAVE_EVERY = 1.5, 30, 50
COOLDOWN, MAX_COOLDOWN = 180, 900


def call(fn, tries=3, wait=10):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == tries:
                raise
            print(f"    attempt {attempt} failed ({type(e).__name__}), waiting {wait * attempt}s")
            time.sleep(wait * attempt)


def fetch(pid, season):
    resp = call(lambda: playerdashptshots.PlayerDashPtShots(
        team_id=0, player_id=pid, season=season, season_type_all_star="Regular Season",
        per_mode_simple="Totals", timeout=TIMEOUT))
    sets = {rs["name"]: rs for rs in resp.get_dict()["resultSets"]}
    rs = sets.get(SET_NAME)
    if rs is None or not rs["rowSet"]:
        return pd.DataFrame({"player_id": [pid], "season": [season], "CLOSE_DEF_DIST_RANGE": [None]})
    df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    df["player_id"], df["season"] = pid, season
    return df


for season in SEASONS:
    out = table_path("league/tracking", "closest_defender.parquet", season)
    existing = pd.read_parquet(out) if os.path.exists(out) else None
    if existing is not None and season == CURRENT_SEASON:
        existing = None                                    # current season: refetch everyone
    done = set(existing["player_id"]) if existing is not None else set()

    box = pd.read_parquet(table_path("league/player_box", "player_box.parquet", season),
                          columns=["game_id", "player_id", "minutes"])
    box = box[box["game_id"].str[:3] == "002"]
    mins = box.groupby("player_id")["minutes"].sum()
    todo = [int(p) for p in mins[mins >= MIN_MINUTES].index if int(p) not in done]
    print(f"{season}: {len(done)} saved, {len(todo)} to fetch (~{len(todo) * 2.3 / 60:.0f} min)")
    if not todo:
        continue

    buffer, streak = [], 0

    def flush():
        global existing, buffer
        if buffer:
            existing = pd.concat(([existing] if existing is not None else []) + buffer, ignore_index=True)
            existing.to_parquet(out, index=False)
            buffer = []

    try:
        for i, pid in enumerate(todo, 1):
            try:
                buffer.append(fetch(pid, season))
                streak = 0
            except Exception as e:
                streak += 1
                pause = min(COOLDOWN * 2 ** (streak - 1), MAX_COOLDOWN)
                print(f"    {pid} FAILED ({type(e).__name__}); cooling down {pause // 60} min")
                flush()
                time.sleep(pause)
            if i % SAVE_EVERY == 0:
                flush()
                print(f"    [{i}/{len(todo)}] saved")
            time.sleep(SLEEP)
    finally:
        flush()
    print(f"{season}: done")