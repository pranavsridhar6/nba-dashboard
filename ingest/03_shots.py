"""Step 1c: every 76ers field goal attempt (made AND missed) with court coordinates."""
import time

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail

from config import TEAM_ID, SEASON, SEASON_TYPES, SLEEP_SECONDS
from utils import call_with_retry, table_path

frames = []
for st in SEASON_TYPES:
    print(f"Fetching {SEASON} {st} shots...")
    df = call_with_retry(lambda st=st: shotchartdetail.ShotChartDetail(
        team_id=TEAM_ID,
        player_id=0,                      # 0 = every player on the team
        season_nullable=SEASON,
        season_type_all_star=st,
        context_measure_simple="FGA",     # REQUIRED: default returns made shots only
        timeout=60,
    ).get_data_frames()[0])
    df["season_type"] = st
    frames.append(df)
    print(f"  {len(df)} shots")
    time.sleep(SLEEP_SECONDS)

shots = pd.concat(frames, ignore_index=True)
shots["GAME_ID"] = shots["GAME_ID"].astype(str).str.zfill(10)
shots["season"] = SEASON

# LOC_X / LOC_Y are tenths of feet with the hoop at (0, 0). Feet are easier to plot.
shots["x_ft"] = shots["LOC_X"] / 10
shots["y_ft"] = shots["LOC_Y"] / 10

out = table_path("shots", "shots.parquet")
shots.to_parquet(out, index=False)
print(f"Saved {len(shots)} shots to {out}")