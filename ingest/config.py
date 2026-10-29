# Settings shared by every ingest script. Change these, not the scripts.

TEAM_ID = 1610612755          # Philadelphia 76ers
TEAM_ABBR = "PHI"

SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
CURRENT_SEASON = "2026-27"    # in progress: always re-pulled. Finished seasons are pulled once.
SEASON_TYPES = ["Regular Season", "Playoffs"]

DATA_DIR = "data"             # relative to the project root, so run scripts from the root
SLEEP_SECONDS = 1.0           # pause between API calls so stats.nba.com doesn't block you