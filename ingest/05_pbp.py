"""Step 2a: play-by-play for every game in each season's games.parquet. Incremental.
Saves every column the API returns; the stint builder decides what to use."""
import os
import time

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

from config import SEASONS, SLEEP_SECONDS
from utils import call_with_retry, table_path

for season in SEASONS:
    games = pd.read_parquet(table_path("games", "games.parquet", season))
    out = table_path("pbp", "pbp.parquet", season)

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
            df = call_with_retry(lambda gid=gid: playbyplayv3.PlayByPlayV3(
                game_id=gid, timeout=60).get_data_frames()[0])
            df.insert(0, "game_id", gid)
            df["season"] = season
            new_frames.append(df)
            print(f"  [{i}/{len(todo)}] {gid}: {len(df)} events")
        except Exception as e:
            failed.append(gid)
            print(f"  [{i}/{len(todo)}] {gid}: FAILED ({e})")
        time.sleep(SLEEP_SECONDS)

    if new_frames:
        frames = ([existing] if existing is not None else []) + new_frames
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(out, index=False)
        print(f"{season}: saved {len(combined)} total events to {out}")
    if failed:
        print(f"{season}: {len(failed)} games failed: {failed}. Rerun to retry only these.")