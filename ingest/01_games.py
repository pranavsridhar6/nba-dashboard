"""Step 1a: list every 76ers game in the season (regular season + playoffs)."""
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

from config import TEAM_ID, SEASON, SEASON_TYPES, SLEEP_SECONDS
from utils import call_with_retry, table_path

frames = []
for st in SEASON_TYPES:
    print(f"Fetching {SEASON} {st} games...")
    df = call_with_retry(lambda st=st: leaguegamefinder.LeagueGameFinder(
        team_id_nullable=TEAM_ID,
        season_nullable=SEASON,
        season_type_nullable=st,
        timeout=60,
    ).get_data_frames()[0])
    df["season_type"] = st
    frames.append(df)
    time.sleep(SLEEP_SECONDS)

games = pd.concat(frames, ignore_index=True)

# GAME_ID must stay a 10-character string with leading zeros, e.g. "0022300061"
games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)
games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

# Derived columns
games["is_home"] = games["MATCHUP"].str.contains("vs.")            # "PHI vs. DAL" = home, "PHI @ DAL" = away
games["opponent"] = games["MATCHUP"].str[-3:]                       # last 3 chars = opponent abbreviation
games["opp_pts"] = (games["PTS"] - games["PLUS_MINUS"]).astype(int)
games["season"] = SEASON

games = (games[["GAME_ID", "GAME_DATE", "MATCHUP", "opponent", "is_home", "WL",
                "PTS", "opp_pts", "season", "season_type"]]
         .drop_duplicates("GAME_ID")
         .sort_values("GAME_DATE")
         .reset_index(drop=True))

out = table_path("games", "games.parquet")
games.to_parquet(out, index=False)
print(f"Saved {len(games)} games to {out}")
print(games["season_type"].value_counts())