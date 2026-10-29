import os
import re
import time

import pandas as pd

from config import DATA_DIR, CURRENT_SEASON


def call_with_retry(fn, tries=3, wait=5):
    """Run an API call; if it fails, wait and try again (stats.nba.com times out often)."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == tries:
                raise
            print(f"  attempt {attempt} failed: {e}. retrying in {wait}s")
            time.sleep(wait)


def table_path(table, filename, season):
    """Build data/<table>/season=<season>/<filename> and create the folder if needed."""
    folder = os.path.join(DATA_DIR, table, f"season={season}")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


def already_done(path, season):
    """True if this is a finished season and its file already exists, so we can skip the API."""
    return season != CURRENT_SEASON and os.path.exists(path)


def minutes_to_decimal(value):
    """Convert minutes like '33:26', '33.000000:26' or 'PT33M26.00S' to 33.43. Blank -> 0."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return 0.0
    s = str(value).strip()
    iso = re.match(r"PT(\d+)M([\d.]+)S", s)
    if iso:
        return round(int(iso.group(1)) + float(iso.group(2)) / 60, 2)
    if ":" in s:
        mins, secs = s.split(":")
        return round(int(float(mins)) + int(float(secs)) / 60, 2)
    return round(float(s), 2)