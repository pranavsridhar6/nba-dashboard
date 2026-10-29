"""Step 1b: player box scores for every game in each season's games.parquet. Incremental."""
import os
import time

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3

from config import SEASONS, SLEEP_SECONDS
from utils import call_with_retry, table_path, minutes_to_decimal

RENAME = {
    "teamId": "team_id", "teamTricode": "team", "personId": "player_id",
    "firstName": "first_name", "familyName": "last_name", "position": "position",
    "comment": "comment", "minutes": "minutes_raw",
    "points": "pts", "reboundsOffensive": "oreb", "reboundsDefensive": "dreb",
    "reboundsTotal": "reb", "assists": "ast", "steals": "stl", "blocks": "blk",
    "turnovers": "tov", "foulsPersonal": "pf",
    "fieldGoalsMade": "fgm", "fieldGoalsAttempted": "fga",
    "threePointersMade": "fg3m", "threePointersAttempted": "fg3a",
    "freeThrowsMade": "ftm", "freeThrowsAttempted": "fta",
    "plusMinusPoints": "plus_minus",
}


def clean(df, game_id, season):
    missing = [c for c in RENAME if c not in df.columns]
    if missing:
        print(f"  warning: columns not found: {missing}. print(df.columns) to check names")
    df = df[[c for c in RENAME if c in df.columns]].rename(columns=RENAME)
    df.insert(0, "game_id", game_id)
    df["player_name"] = df["first_name"] + " " + df["last_name"]
    df["minutes"] = df["minutes_raw"].apply(minutes_to_decimal)
    df["did_play"] = df["minutes"] > 0
    df["season"] = season
    return df


for season in SEASONS:
    games = pd.read_parquet(table_path("games", "games.parquet", season))
    out = table_path("player_box", "player_box.parquet", season)

    if os.path.exists(out):
        existing = pd.read_parquet(out)
        done = set(existing["game_id"])
    else:
        existing, done = None, set()

    todo = [g for g in games["GAME_ID"] if g not in done]
    print(f"{season}: {len(done)} games saved, {len(todo)} to fetch")
    if not todo:
        continue

    new_frames, failed = [], []
    for i, gid in enumerate(todo, 1):
        try:
            raw = call_with_retry(lambda gid=gid: boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=gid, timeout=60).get_data_frames()[0])
            new_frames.append(clean(raw, gid, season))
            print(f"  [{i}/{len(todo)}] {gid}: {len(raw)} player rows")
        except Exception as e:
            failed.append(gid)
            print(f"  [{i}/{len(todo)}] {gid}: FAILED ({e})")
        time.sleep(SLEEP_SECONDS)

    if new_frames:
        frames = ([existing] if existing is not None else []) + new_frames
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(out, index=False)
        print(f"{season}: saved {len(combined)} total rows to {out}")
    if failed:
        print(f"{season}: {len(failed)} games failed: {failed}. Rerun to retry only these.")