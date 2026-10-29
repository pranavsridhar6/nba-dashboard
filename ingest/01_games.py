"""Step 1a: list every 76ers game played, per season (regular season + playoffs)."""
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

from config import TEAM_ID, SEASONS, SEASON_TYPES, SLEEP_SECONDS
from utils import call_with_retry, table_path, already_done


def fetch_season(season):
    frames = []
    for st in SEASON_TYPES:
        df = call_with_retry(lambda st=st: leaguegamefinder.LeagueGameFinder(
            team_id_nullable=TEAM_ID,
            season_nullable=season,
            season_type_nullable=st,
            timeout=60,
        ).get_data_frames()[0])
        df["season_type"] = st
        frames.append(df)
        print(f"  {st}: {len(df)} games")
        time.sleep(SLEEP_SECONDS)

    games = pd.concat(frames, ignore_index=True)
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["is_home"] = games["MATCHUP"].str.contains("vs.")
    games["opponent"] = games["MATCHUP"].str[-3:]
    games["opp_pts"] = (games["PTS"] - games["PLUS_MINUS"]).astype(int)
    games["season"] = season

    return (games[["GAME_ID", "GAME_DATE", "MATCHUP", "opponent", "is_home", "WL",
                   "PTS", "opp_pts", "season", "season_type"]]
            .drop_duplicates("GAME_ID")
            .sort_values("GAME_DATE")
            .reset_index(drop=True))


for season in SEASONS:
    out = table_path("games", "games.parquet", season)
    if already_done(out, season):
        print(f"{season}: already saved, skipping")
        continue
    print(f"{season}: fetching")
    games = fetch_season(season)
    games.to_parquet(out, index=False)
    print(f"{season}: saved {len(games)} games to {out}")