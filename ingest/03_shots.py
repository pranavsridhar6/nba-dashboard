"""Step 1c: every 76ers field goal attempt (made AND missed) with court coordinates, per season."""
import time

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail

from config import TEAM_ID, SEASONS, SEASON_TYPES, SLEEP_SECONDS
from utils import call_with_retry, table_path, already_done

for season in SEASONS:
    out = table_path("shots", "shots.parquet", season)
    if already_done(out, season):
        print(f"{season}: already saved, skipping")
        continue

    frames = []
    for st in SEASON_TYPES:
        df = call_with_retry(lambda st=st: shotchartdetail.ShotChartDetail(
            team_id=TEAM_ID,
            player_id=0,                      # 0 = every player on the team
            season_nullable=season,
            season_type_all_star=st,
            context_measure_simple="FGA",     # REQUIRED: default returns made shots only
            timeout=60,
        ).get_data_frames()[0])
        df["season_type"] = st
        frames.append(df)
        print(f"  {season} {st}: {len(df)} shots")
        time.sleep(SLEEP_SECONDS)

    shots = pd.concat(frames, ignore_index=True)
    shots["GAME_ID"] = shots["GAME_ID"].astype(str).str.zfill(10)
    shots["season"] = season
    shots["x_ft"] = shots["LOC_X"] / 10       # tenths of feet -> feet, hoop at (0, 0)
    shots["y_ft"] = shots["LOC_Y"] / 10

    shots.to_parquet(out, index=False)
    print(f"{season}: saved {len(shots)} shots to {out}")