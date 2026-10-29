"""Scratch file: inspect one game's play-by-play so we can write the stint builder. Delete after."""
import pandas as pd

from utils import table_path

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)

pbp = pd.read_parquet(table_path("pbp", "pbp.parquet", "2023-24"))
gid = pbp["game_id"].iloc[0]
g = pbp[pbp["game_id"] == gid]
print(f"Game {gid}: {len(g)} events\n")

print("--- columns ---")
print(g.columns.tolist())

print("\n--- actionType counts ---")
print(g["actionType"].value_counts().to_string())

cols = [c for c in ["actionNumber", "period", "clock", "teamTricode", "personId",
                    "playerName", "actionType", "subType", "description",
                    "scoreHome", "scoreAway"] if c in g.columns]

print("\n--- first 8 substitution rows ---")
subs = g[g["actionType"].str.contains("ubstitution", case=False, na=False)]
print(subs[cols].head(8).to_string())

print("\n--- first 12 events of period 2 ---")
print(g[g["period"] == 2][cols].head(12).to_string())

print("\n--- last 5 events of the game ---")
print(g[cols].tail(5).to_string())