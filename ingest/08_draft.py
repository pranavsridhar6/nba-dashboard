"""Step 8a: raw data for the draft translator.

  1. Draft history (every NBA draft pick, with college)        -> data/draft/draft_history.parquet
  2. College player seasons from Barttorvik, 2010 -> now        -> data/draft/college/year=YYYY/college_raw.parquet
     (year = season END year: 2019 = the 2018-19 college season)
  3. Every NBA player's season totals, 2010-11 -> now           -> data/draft/nba/season=YYYY-YY/nba_totals.parquet

Finished seasons are downloaded once; the current college and NBA seasons refresh every run.
If Barttorvik blocks the download, save the CSV from your browser as
data/draft/college_csv/YYYY.csv and rerun: local files are used first.
"""
import io
import os
import time

import pandas as pd
import requests
from nba_api.stats.endpoints import drafthistory, leaguedashplayerstats

from config import DATA_DIR, SLEEP_SECONDS
from utils import call_with_retry

COLLEGE_YEARS = range(2010, 2028)                  # 2027 = the 2026-27 college season
CURRENT_COLLEGE_YEAR = 2027
NBA_SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2010, 2027)]   # 2010-11 ... 2026-27
CURRENT_NBA_SEASON = "2026-27"
BART_URL = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def path(*parts):
    p = os.path.join(DATA_DIR, "draft", *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def draft_history():
    out = path("draft_history.parquet")
    df = call_with_retry(lambda: drafthistory.DraftHistory(league_id="00", timeout=60).get_data_frames()[0])
    df = df[pd.to_numeric(df["SEASON"], errors="coerce") >= 2010]
    df.to_parquet(out, index=False)
    print(f"draft history: {len(df)} picks since 2010")


def college():
    for year in COLLEGE_YEARS:
        out = path("college", f"year={year}", "college_raw.parquet")
        if os.path.exists(out) and year != CURRENT_COLLEGE_YEAR:
            print(f"college {year}: already saved, skipping")
            continue
        local = path("college_csv", f"{year}.csv")
        if os.path.exists(local):
            text = open(local, encoding="utf-8", errors="replace").read()
            src = "local file"
        else:
            r = requests.get(BART_URL.format(year=year), headers=BROWSER, timeout=60)
            text, src = r.text, f"download (HTTP {r.status_code})"
            if r.status_code != 200 or text.lstrip().startswith("<"):
                print(f"college {year}: BLOCKED or not a CSV ({src}). Save it from your browser to {local}")
                time.sleep(2)
                continue
        df = pd.read_csv(io.StringIO(text), header=None, dtype=str)
        df.columns = [f"c{i}" for i in range(df.shape[1])]
        df.to_parquet(out, index=False)
        print(f"college {year}: {len(df):,} player rows, {df.shape[1]} columns ({src})")
        time.sleep(2)


def nba_totals():
    for season in NBA_SEASONS:
        out = path("nba", f"season={season}", "nba_totals.parquet")
        if os.path.exists(out) and season != CURRENT_NBA_SEASON:
            print(f"nba {season}: already saved, skipping")
            continue
        def get(measure):
            return call_with_retry(lambda: leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Regular Season", per_mode_detailed="Totals",
                measure_type_detailed_defense=measure, timeout=60).get_data_frames()[0])
        base = get("Base")
        time.sleep(SLEEP_SECONDS)
        adv = get("Advanced")[["PLAYER_ID", "POSS"]]
        time.sleep(SLEEP_SECONDS)
        df = base.merge(adv, on="PLAYER_ID", how="left")
        df["season"] = season
        df.to_parquet(out, index=False)
        print(f"nba {season}: {len(df)} players")


if __name__ == "__main__":
    draft_history()
    college()
    nba_totals()