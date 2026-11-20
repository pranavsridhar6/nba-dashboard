import duckdb
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

def src(pattern: str) -> str:
    path = (DATA/pattern).as_posix()
    return f"read_parquet('{path}', union_by_name=true, hive_partitioning=false)"

VIEWS = {
    "team_games":   f"SELECT * FROM {src('league/team_games/*/*.parquet')}",
    "team_season":  f"SELECT * FROM {src('league/overview/*/*.parquet')}",
    "rapm":         f"SELECT * FROM {src('league/rapm/*/*.parquet')}",
    "cap_summary":  f"SELECT * FROM {src('cap/season=2026-27/cap_summary.parquet')}",
    "contracts":    f"SELECT * FROM {src('contracts/contracts_all.parquet')}",
    "player_value": f"SELECT * FROM {src('cap/season=2026-27/player_value.parquet')}",
    "lineups":      f"SELECT * FROM {src('league/lineups/*/lineup_combos.parquet')}",
    "rim_protection": f"SELECT * FROM {src('league/lineups/*/on_off_rim.parquet')}",
    "spacing":      f"SELECT * FROM {src('league/lineups/*/spacing.parquet')}",
    "draft_projections": f"SELECT * FROM {src('draft/projections.parquet')}",
    "draft_backtest": f"SELECT * FROM {src('draft/cv_predictions.parquet')}",
}

VIEWS["player_season"] = f"""
SELECT b.season, g.season_type, b.team, b.player_id, b.player_name,
       COUNT(*) AS gp,
       ROUND(AVG(b.pts), 1) AS ppg,
       ROUND(AVG(b.reb), 1) AS rpg,
       ROUND(AVG(b.ast), 1) AS apg,
       ROUND(AVG(b.minutes), 1) AS mpg
FROM {src('league/player_box/*/*.parquet')} b
JOIN {src('league/games/*/*.parquet')} g USING (game_id)
WHERE b.did_play
GROUP BY ALL
"""
def connect():
    con = duckdb.connect()   # in-memory: nothing is ever written to disk
    for name, sql in VIEWS.items():
        con.execute(f"CREATE VIEW {name} AS {sql}")
    return con

if __name__ == "__main__":
    con = connect()
    for name in VIEWS:
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchall()[0][0]
        print(f"{name:18s} {n:>9,}")
    print(con.execute("""
        SELECT COUNT(*) FILTER (WHERE result = 'W') AS w,
               COUNT(*) FILTER (WHERE result = 'L') AS l
        FROM team_games
        WHERE team = 'BOS' AND season = '2025-26' AND season_type = 'Regular Season'
    """).df())