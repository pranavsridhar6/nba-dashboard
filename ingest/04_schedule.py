"""Step 1d: full 76ers schedule for the current season, played AND upcoming.
Finished seasons don't need this: every game in them is already in games.parquet."""
import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2

from config import TEAM_ID, CURRENT_SEASON
from utils import call_with_retry, table_path

# First 3 characters of a game ID tell you the game type
GAME_TYPES = {"002": "Regular Season", "004": "Playoffs", "005": "PlayIn"}
STATUS = {1: "Scheduled", 2: "Live", 3: "Final"}

raw = call_with_retry(lambda: scheduleleaguev2.ScheduleLeagueV2(
    league_id="00", season=CURRENT_SEASON, timeout=60).get_data_frames()[0])
print(f"{len(raw)} league-wide games returned")

# Keep only 76ers games, and drop preseason / All-Star / other types
df = raw[(raw["homeTeam_teamId"] == TEAM_ID) | (raw["awayTeam_teamId"] == TEAM_ID)].copy()
df["gameId"] = df["gameId"].astype(str).str.zfill(10)
df["season_type"] = df["gameId"].str[:3].map(GAME_TYPES)
df = df[df["season_type"].notna()]

# Flip home/away columns into "76ers vs opponent" columns
df["is_home"] = df["homeTeam_teamId"] == TEAM_ID
side = lambda home_col, away_col: df[home_col].where(df["is_home"], df[away_col])
df["opponent"] = df["awayTeam_teamTricode"].where(df["is_home"], df["homeTeam_teamTricode"])
df["opponent_id"] = df["awayTeam_teamId"].where(df["is_home"], df["homeTeam_teamId"])
df["team_score"] = side("homeTeam_score", "awayTeam_score")
df["opp_score"] = df["awayTeam_score"].where(df["is_home"], df["homeTeam_score"])

# Tip-off time: convert true UTC to US Eastern, then drop the timezone tag for Power BI
tip = pd.to_datetime(df["gameDateTimeUTC"], utc=True).dt.tz_convert("America/New_York")
df["tip_time_et"] = tip.dt.tz_localize(None)
df["game_date"] = df["tip_time_et"].dt.normalize()

df["game_status"] = df["gameStatus"].map(STATUS)
# Scores are 0 for games not played yet; make them blank so they don't skew averages
not_final = df["game_status"] != "Final"
df.loc[not_final, ["team_score", "opp_score"]] = None

df["season"] = CURRENT_SEASON
schedule = (df.rename(columns={
                "gameId": "game_id", "arenaName": "arena", "ifNecessary": "if_necessary",
                "postponedStatus": "postponed_status",
                "nationalBroadcasters_0_broadcasterDisplay": "national_tv"})
            [["game_id", "game_date", "tip_time_et", "season", "season_type", "game_status",
              "is_home", "opponent", "opponent_id", "team_score", "opp_score",
              "arena", "national_tv", "if_necessary", "postponed_status"]]
            .sort_values("tip_time_et")
            .reset_index(drop=True))

out = table_path("schedule", "schedule.parquet", CURRENT_SEASON)
schedule.to_parquet(out, index=False)
print(f"Saved {len(schedule)} games to {out}")
print(schedule.groupby(["season_type", "game_status"]).size().to_string())