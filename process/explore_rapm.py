import pandas as pd
for s in ["2025-26", "2026-27"]:
    r = pd.read_parquet(f"data/league/rapm/season={s}/rapm.parquet")
    print(f"\n{s} (window: {r['window'].iloc[0]})")
    print(r[r.player_name.isin(["Jaylen Brown", "LeBron James", "Derrick White", "Luka Dončić"])]
          [["player_name", "team", "rapm", "o_rapm", "d_rapm", "off_poss", "on_court_net"]].to_string(index=False))