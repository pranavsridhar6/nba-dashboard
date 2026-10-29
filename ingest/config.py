# Settings shared by every ingest script. Change these, not the scripts.

TEAM_ID = 1610612755          # Philadelphia 76ers
TEAM_ABBR = "PHI"
SEASON = "2023-24"
SEASON_TYPES = ["Regular Season", "Playoffs"]

DATA_DIR = "data"             # relative to the project root, so run scripts from the root
SLEEP_SECONDS = 1.0           # pause between API calls so stats.nba.com doesn't block you