"""Step 8b: build the draft translator's training table.

1. College seasons (Barttorvik): name the columns, convert types, add height (inches), exact age at the
   draft (from birthdate), 3-point rate, power-conference flag.
2. Match every drafted college player (2010+) to his final college season: by draft year + pick
   number first (Barttorvik stores the pick), then by name as a fallback.
3. Outcome: his rookie NBA season's box-score impact, using the SAME box-score model as the RAPM prior,
   centered so the league-average player that season = 0.

Outputs:
  data/draft/college_all.parquet     every college player-season, cleaned (used for prospects later)
  data/draft/training.parquet        drafted college players + rookie outcome
"""
import glob
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import DATA_DIR        # noqa: E402

D = os.path.join(DATA_DIR, "draft")
COLS = {"c0": "player", "c1": "school", "c2": "conf", "c3": "gp", "c4": "min_pct", "c5": "ortg",
        "c6": "usg", "c7": "efg", "c8": "ts", "c9": "orb_pct", "c10": "drb_pct", "c11": "ast_pct",
        "c12": "tov_pct", "c15": "ft_pct", "c17": "fg2a", "c20": "fg3a", "c21": "fg3_pct",
        "c22": "blk_pct", "c23": "stl_pct", "c24": "ftr", "c25": "cls", "c26": "height", "c31": "year",
        "c32": "bart_id", "c34": "rec_rank", "c45": "pick", "c50": "bpm", "c51": "obpm", "c52": "dbpm",
        "c54": "mpg", "c59": "rpg", "c60": "apg", "c63": "ppg", "c66": "birthdate"}
TEXT = {"player", "school", "conf", "cls", "height", "birthdate"}
POWER = {"ACC", "B10", "B12", "SEC", "BE", "P12"}
SHRINK_POSS, SHRINK_ATT = 1000, 200


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def inches(h):
    m = re.match(r"(\d+)-(\d+)", str(h))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def load_college():
    frames = []
    for f in sorted(glob.glob(os.path.join(D, "college", "year=*", "college_raw.parquet"))):
        df = pd.read_parquet(f)[list(COLS)].rename(columns=COLS)
        df["year"] = int(re.search(r"year=(\d{4})", f.replace("\\", "/")).group(1))   # from the folder name
        frames.append(df)
    c = pd.concat(frames, ignore_index=True)
    for col in COLS.values():
        if col not in TEXT and col != "year":
            c[col] = pd.to_numeric(c[col], errors="coerce")
    c["height_in"] = c["height"].map(inches)
    bd = pd.to_datetime(c["birthdate"], errors="coerce")
    c["age_at_draft"] = (pd.to_datetime(c["year"].astype(str) + "-06-25") - bd).dt.days / 365.25
    c["fg3a_rate"] = c["fg3a"] / (c["fg3a"] + c["fg2a"])
    c["power_conf"] = c["conf"].isin(POWER).astype(int)
    c["name_key"] = c["player"].map(norm)
    return c


def box_coefs():
    meta = sorted(glob.glob(os.path.join(DATA_DIR, "league", "rapm", "season=*", "rapm_meta.json")))[-1]
    bp = json.load(open(meta))["box_prior"]
    return bp["features"], bp["offense_coefs"], bp["defense_coefs"]


def nba_impact(season, feats, o_coef, d_coef):
    """Box-score impact for every NBA player in a season, centered on the possession-weighted average."""
    t = pd.read_parquet(os.path.join(D, "nba", f"season={season}", "nba_totals.parquet"))
    t = t[t["POSS"] > 0].copy()
    poss = t["POSS"]
    for stat, col in [("pts", "PTS"), ("fg3a", "FG3A"), ("fta", "FTA"), ("ast", "AST"), ("oreb", "OREB"),
                      ("dreb", "DREB"), ("stl", "STL"), ("blk", "BLK"), ("tov", "TOV"), ("pf", "PF")]:
        rate = t[col].sum() / poss.sum()
        t[f"{stat}100"] = 100 * (t[col] + rate * SHRINK_POSS) / (poss + SHRINK_POSS)
    att = t["FGA"] + 0.44 * t["FTA"]
    lg_ts = t["PTS"].sum() / (2 * att.sum())
    t["ts_pct"] = (t["PTS"] + lg_ts * 2 * SHRINK_ATT) / (2 * (att + SHRINK_ATT))
    o = sum(t[f] * o_coef[f] for f in feats)
    d = sum(t[f] * d_coef[f] for f in feats)
    w = poss / poss.sum()
    t["rookie_o"] = o - (o * w).sum()
    t["rookie_d"] = d - (d * w).sum()
    t["rookie_impact"] = t["rookie_o"] + t["rookie_d"]
    return t[["PLAYER_ID", "MIN", "POSS", "GP", "rookie_o", "rookie_d", "rookie_impact"]]


def main():
    college = load_college()
    college.to_parquet(os.path.join(D, "college_all.parquet"), index=False)

    dh = pd.read_parquet(os.path.join(D, "draft_history.parquet"))
    dh = dh[dh["ORGANIZATION_TYPE"].astype(str).str.contains("College", na=False)].copy()
    dh["draft_year"] = dh["SEASON"].astype(int)
    dh["OVERALL_PICK"] = pd.to_numeric(dh["OVERALL_PICK"], errors="coerce")
    dh["name_key"] = dh["PLAYER_NAME"].map(norm)

    rows, unmatched = [], []
    for r in dh.itertuples(index=False):
        yr = college[college["year"] == r.draft_year]
        hit = yr[(yr["pick"] == r.OVERALL_PICK) & (yr["name_key"] == r.name_key)]   # pick + name agree
        how = "pick+name"
        if hit.empty:
            hit = yr[yr["pick"] == r.OVERALL_PICK]
            how = "pick"
            if len(hit) != 1:
                hit = yr[yr["name_key"] == r.name_key]
                how = "name"
        if hit.empty:
            # Nicknames (Mo/Mohamed, Cam/Cameron) and sat-out final seasons: last name + first initial
            parts = str(r.PLAYER_NAME).split()
            last, first = norm(parts[-1]) if parts else "", norm(parts[0])[:1] if parts else ""
            near = college[college["year"].isin([r.draft_year, r.draft_year - 1])]
            near = near[near["player"].map(lambda n: norm(str(n).split()[-1]) == last
                                           and norm(str(n).split()[0])[:1] == first)]
            hit, how = near, "last name + initial"
        if hit.empty:
            unmatched.append((r.draft_year, r.OVERALL_PICK, r.PLAYER_NAME, r.ORGANIZATION))
            continue
        h = hit.sort_values(["year", "min_pct"], ascending=False).iloc[0].to_dict()
        h.update(nba_id=int(r.PERSON_ID), nba_name=r.PLAYER_NAME, draft_year=r.draft_year,
                 overall_pick=r.OVERALL_PICK, match=how)
        rows.append(h)
    train = pd.DataFrame(rows)

    feats, o_coef, d_coef = box_coefs()
    outcomes = []
    for y in sorted(train["draft_year"].unique()):
        season = f"{y}-{str(y + 1)[2:]}"
        if os.path.exists(os.path.join(D, "nba", f"season={season}", "nba_totals.parquet")):
            o = nba_impact(season, feats, o_coef, d_coef)
            o["rookie_season"] = season
            outcomes.append(o)
    out = pd.concat(outcomes).rename(columns={"PLAYER_ID": "nba_id", "MIN": "rookie_min",
                                              "POSS": "rookie_poss", "GP": "rookie_gp"})
    train["rookie_season"] = train["draft_year"].map(lambda y: f"{y}-{str(y + 1)[2:]}")
    train = train.merge(out, on=["nba_id", "rookie_season"], how="left")
    train.to_parquet(os.path.join(D, "training.parquet"), index=False)

    # ---------- summary ----------
    total = len(dh)
    print(f"college player-seasons: {len(college):,}")
    print(f"drafted college players since 2010: {total}, matched: {len(train)} ({len(train) / total:.1%})")
    assert train["nba_id"].is_unique, "a player appears more than once"
    print(f"  match method: {train['match'].value_counts().to_dict()}")
    print(f"  with a rookie NBA season: {train['rookie_impact'].notna().sum()}")
    if unmatched:
        print(f"  unmatched ({len(unmatched)}), first 10:")
        for u in unmatched[:10]:
            print(f"    {u}")
    print("\nbest rookie seasons (min 1,000 possessions):")
    best = train[train["rookie_poss"] >= 1000].sort_values("rookie_impact", ascending=False)
    print(best[["draft_year", "overall_pick", "nba_name", "school", "age_at_draft", "bpm",
                "rookie_min", "rookie_impact"]].head(10).round(2).to_string(index=False))
    print("\ncheck, a few known players:")
    known = train[train["nba_name"].isin(["Zion Williamson", "Ja Morant", "Paolo Banchero", "Cooper Flagg"])]
    print(known[["draft_year", "overall_pick", "nba_name", "match", "age_at_draft", "height_in", "bpm",
                 "rookie_impact"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()