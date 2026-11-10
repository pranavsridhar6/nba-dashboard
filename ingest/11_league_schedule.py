"""Step 9a: full league schedule for the current season (played and upcoming), every team.
One API call. Output: data/league/schedule/season=<current>/schedule.parquet
"""
import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2

from config import CURRENT_SEASON
from utils import call_with_retry, table_path

GAME_TYPES = {"002": "Regular Season", "004": "Playoffs", "005": "PlayIn"}
STATUS = {1: "Scheduled", 2: "Live", 3: "Final"}

raw = call_with_retry(lambda: scheduleleaguev2.ScheduleLeagueV2(
    league_id="00", season=CURRENT_SEASON, timeout=60).get_data_frames()[0])
df = raw.copy()
df["game_id"] = df["gameId"].astype(str).str.zfill(10)
df["season_type"] = df["game_id"].str[:3].map(GAME_TYPES)
df = df[df["season_type"].notna()]
tip = pd.to_datetime(df["gameDateTimeUTC"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
out = pd.DataFrame({
    "game_id": df["game_id"], "season": CURRENT_SEASON, "season_type": df["season_type"],
    "tip_time_et": tip, "game_date": tip.dt.normalize(), "game_status": df["gameStatus"].map(STATUS),
    "home_team_id": df["homeTeam_teamId"], "home_team": df["homeTeam_teamTricode"],
    "away_team_id": df["awayTeam_teamId"], "away_team": df["awayTeam_teamTricode"],
    "home_score": df["homeTeam_score"], "away_score": df["awayTeam_score"],
    "arena": df["arenaName"], "national_tv": df.get("nationalBroadcasters_0_broadcasterDisplay"),
}).sort_values("tip_time_et").reset_index(drop=True)
path = table_path("league/schedule", "schedule.parquet", CURRENT_SEASON)
out.to_parquet(path, index=False)
print(f"{len(out)} games saved to {path}")
print(out.groupby(["season_type", "game_status"]).size().to_string())