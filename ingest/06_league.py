"""Step 3a: league-wide games, box scores and play-by-play (for RAPM).

Incremental AND checkpointed: progress is saved every 50 games, and on Ctrl+C.
Stop any time, rerun the same command, and it picks up where it left off."""
import os
import time

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3, leaguegamefinder, playbyplayv3

from config import SEASONS, SEASON_TYPES
from utils import already_done, minutes_to_decimal, table_path

SAVE_EVERY = 50
LEAGUE_SLEEP = 1.5          # a bit slower than the 76ers scripts: thousands of calls in a row
TIMEOUT = 30                # a throttled request rarely succeeds by waiting longer
COOLDOWN = 180              # after a failed game, pause 3 min so the API's throttle resets
MAX_COOLDOWN = 900

RENAME = {
    "teamId": "team_id", "teamTricode": "team", "personId": "player_id",
    "firstName": "first_name", "familyName": "last_name", "position": "position",
    "comment": "comment", "minutes": "minutes_raw",
    "points": "pts", "reboundsOffensive": "oreb", "reboundsDefensive": "dreb",
    "reboundsTotal": "reb", "assists": "ast", "steals": "stl", "blocks": "blk",
    "turnovers": "tov", "foulsPersonal": "pf",
    "fieldGoalsMade": "fgm", "fieldGoalsAttempted": "fga",
    "threePointersMade": "fg3m", "threePointersAttempted": "fg3a",
    "freeThrowsMade": "ftm", "freeThrowsAttempted": "fta",
    "plusMinusPoints": "plus_minus",
}


def call(fn, tries=3, wait=10):
    """Retry with growing waits (10, 20s). If all tries fail, incremental() cools down instead."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == tries:
                raise
            w = wait * 2 ** (attempt - 1)
            print(f"    attempt {attempt} failed ({type(e).__name__}), waiting {w}s")
            time.sleep(w)


def fetch_games(season):
    """One row per game: home and away team, final score."""
    frames = []
    for st in SEASON_TYPES:
        df = call(lambda st=st: leaguegamefinder.LeagueGameFinder(
            player_or_team_abbreviation="T", league_id_nullable="00",
            season_nullable=season, season_type_nullable=st, timeout=TIMEOUT,
        ).get_data_frames()[0])
        df["season_type"] = st
        frames.append(df)
        time.sleep(LEAGUE_SLEEP)

    df = pd.concat(frames, ignore_index=True)
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(["002", "004"])]          # regular season + playoffs only

    # Pair the two team rows of each game. Normally one row says "vs." (home) and one "@" (away).
    # Neutral-site games (international games, NBA Cup semifinals) break that pattern, so we
    # never drop a game: if both rows look alike, the first row is treated as the home team.
    cols = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]
    df["is_home_row"] = df["MATCHUP"].str.contains("vs.")
    df = df.sort_values(["GAME_ID", "is_home_row"], ascending=[True, False])
    pairs = df.groupby("GAME_ID")
    home = pairs.nth(0)[cols + ["GAME_DATE", "season_type", "is_home_row"]]
    away = pairs.nth(1)[cols + ["is_home_row"]]
    games = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))
    games["neutral_site"] = games["is_home_row_home"] == games["is_home_row_away"]
    games = games.drop(columns=["is_home_row_home", "is_home_row_away"])
    games = games.rename(columns={
        "GAME_ID": "game_id", "GAME_DATE": "game_date",
        "TEAM_ID_home": "home_team_id", "TEAM_ABBREVIATION_home": "home_team", "PTS_home": "home_pts",
        "TEAM_ID_away": "away_team_id", "TEAM_ABBREVIATION_away": "away_team", "PTS_away": "away_pts",
    })
    games["game_date"] = pd.to_datetime(games["game_date"])
    reg = (games["season_type"] == "Regular Season").sum()
    print(f"  {reg} regular season games ({games['neutral_site'].sum()} neutral site)")
    games["season"] = season
    return games.drop_duplicates("game_id").sort_values("game_date").reset_index(drop=True)


def fetch_box(gid, season):
    raw = call(lambda: boxscoretraditionalv3.BoxScoreTraditionalV3(
        game_id=gid, timeout=TIMEOUT).get_data_frames()[0])
    df = raw[[c for c in RENAME if c in raw.columns]].rename(columns=RENAME)
    df.insert(0, "game_id", gid)
    df["player_name"] = df["first_name"] + " " + df["last_name"]
    df["minutes"] = df["minutes_raw"].apply(minutes_to_decimal)
    df["did_play"] = df["minutes"] > 0
    df["season"] = season
    return df


def fetch_pbp(gid, season):
    df = call(lambda: playbyplayv3.PlayByPlayV3(game_id=gid, timeout=TIMEOUT).get_data_frames()[0])
    df.insert(0, "game_id", gid)
    df["season"] = season
    return df


def incremental(game_ids, out, fetch_one, label):
    """Fetch every game not already in `out`, saving every SAVE_EVERY games and on Ctrl+C."""
    existing = pd.read_parquet(out) if os.path.exists(out) else None
    done = set(existing["game_id"]) if existing is not None else set()
    todo = [g for g in game_ids if g not in done]
    print(f"  {label}: {len(done)} saved, {len(todo)} to fetch "
          f"(~{len(todo) * (LEAGUE_SLEEP + 0.8) / 60:.0f} min)")
    if not todo:
        return

    buffer, failed = [], []
    streak = 0                                    # failed games in a row

    def flush():
        nonlocal existing, buffer
        if buffer:
            existing = pd.concat(([existing] if existing is not None else []) + buffer, ignore_index=True)
            existing.to_parquet(out, index=False)
            buffer = []

    try:
        for i, gid in enumerate(todo, 1):
            try:
                buffer.append(fetch_one(gid))
                streak = 0
            except Exception as e:
                failed.append(gid)
                streak += 1
                pause = min(COOLDOWN * 2 ** (streak - 1), MAX_COOLDOWN)
                print(f"    {gid} FAILED ({type(e).__name__}). API is throttling; "
                      f"cooling down {pause // 60:.0f} min")
                flush()
                time.sleep(pause)
            if i % SAVE_EVERY == 0:
                flush()
                print(f"    [{i}/{len(todo)}] {label} saved")
            time.sleep(LEAGUE_SLEEP)
    finally:
        flush()                                   # runs on Ctrl+C too, so nothing fetched is lost
    if failed:
        print(f"  {label}: {len(failed)} failed. Rerun to retry only these.")


if __name__ == "__main__":
    start = time.time()
    for season in SEASONS:
        print(f"\n=== {season} ===")
        gpath = table_path("league/games", "games.parquet", season)
        if already_done(gpath, season):
            games = pd.read_parquet(gpath)
        else:
            games = fetch_games(season)
            games.to_parquet(gpath, index=False)
        print(f"  games: {len(games)} ({games['season_type'].value_counts().to_dict()})")

        ids = games["game_id"].tolist()
        incremental(ids, table_path("league/player_box", "player_box.parquet", season),
                    lambda g: fetch_box(g, season), "box scores")
        incremental(ids, table_path("league/pbp", "pbp.parquet", season),
                    lambda g: fetch_pbp(g, season), "play-by-play")

    print(f"\nDone in {(time.time() - start) / 60:.0f} min")