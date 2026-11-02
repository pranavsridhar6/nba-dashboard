"""Step 3c: Regularized Adjusted Plus-Minus (RAPM) from league-wide stints.

For each season S, fits on S + the season before it (2-season window), regular season only.
Each stint becomes two rows (home on offense, away on offense). For each row:
    y      = points scored per 100 possessions by the offense
    weight = possessions
    X      = +1 for each offensive player, +1 for each defensive player (separate columns),
             plus a home-court column
Ridge regression shrinks players with little playing time toward 0 (league average).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import SEASONS          # noqa: E402
from utils import table_path        # noqa: E402

ALPHAS = [100, 300, 1000, 3000, 10000, 30000, 100000]   # ridge strengths to test
FOLDS = 5
MIN_POSS = 2000          # offense + defense possessions before we call a player's RAPM reliable
HOME = [f"home_p{i}" for i in range(1, 6)]
AWAY = [f"away_p{i}" for i in range(1, 6)]


def load_stints(season):
    """Regular season, full 5-on-5 lineups, from games whose stint points add up."""
    st = pd.read_parquet(table_path("league/stints", "stints.parquet", season))
    chk = pd.read_parquet(table_path("league/stints", "game_checks.parquet", season))
    good = set(chk.loc[chk["pts_ok"], "game_id"])
    keep = st["game_id"].str[:3].eq("002") & st["lineup_complete"] & st["game_id"].isin(good)
    return st[keep]


def to_rows(st):
    """Every stint -> 2 rows: home team on offense, then away team on offense."""
    off = np.vstack([st[HOME].to_numpy(), st[AWAY].to_numpy()]).astype(np.int64)
    dfn = np.vstack([st[AWAY].to_numpy(), st[HOME].to_numpy()]).astype(np.int64)
    pts = np.concatenate([st["home_pts"].to_numpy(), st["away_pts"].to_numpy()]).astype(float)
    poss = np.concatenate([st["home_poss"].to_numpy(), st["away_poss"].to_numpy()]).astype(float)
    is_home = np.concatenate([np.ones(len(st)), np.zeros(len(st))])
    games = np.concatenate([st["game_id"].to_numpy(), st["game_id"].to_numpy()])
    keep = poss > 0
    return off[keep], dfn[keep], pts[keep], poss[keep], is_home[keep], games[keep]


def design_matrix(off, dfn, is_home):
    """Sparse X: [offense columns | defense columns | home column]."""
    players = np.unique(np.concatenate([off.ravel(), dfn.ravel()]))
    n, p = len(off), len(players)
    rows = np.repeat(np.arange(n), 5)
    x_off = sparse.csr_matrix((np.ones(n * 5), (rows, np.searchsorted(players, off.ravel()))), shape=(n, p))
    x_def = sparse.csr_matrix((np.ones(n * 5), (rows, np.searchsorted(players, dfn.ravel()))), shape=(n, p))
    X = sparse.hstack([x_off, x_def, sparse.csr_matrix(is_home.reshape(-1, 1))]).tocsr()
    return X, x_off, x_def, players


def pick_alpha(X, y, w, groups):
    """Cross-validate by game (a game's stints are never split between train and test)."""
    results = {}
    for a in ALPHAS:
        errs = []
        for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, groups):
            m = Ridge(alpha=a).fit(X[tr], y[tr], sample_weight=w[tr])
            errs.append(np.average((y[te] - m.predict(X[te])) ** 2, weights=w[te]))
        results[a] = float(np.mean(errs))
        print(f"    alpha {a:>7}: CV error {results[a]:.1f}")
    return min(results, key=results.get), results


def player_info(window, season):
    """Name, and the team a player logged the most minutes for in the selected season."""
    boxes = []
    for s in window:
        b = pd.read_parquet(table_path("league/player_box", "player_box.parquet", s),
                            columns=["player_id", "player_name", "team", "minutes"])
        b["season"] = s
        boxes.append(b)
    box = pd.concat(boxes)
    names = box.sort_values("season").groupby("player_id")["player_name"].last()
    cur = box[box["season"] == season]
    if cur.empty:
        cur = box
    teams = (cur.groupby(["player_id", "team"])["minutes"].sum()
                .reset_index().sort_values("minutes")
                .groupby("player_id")["team"].last())
    return names, teams


def run(season):
    i = SEASONS.index(season)
    window = SEASONS[max(0, i - 1): i + 1]
    print(f"\n{season}  (window: {', '.join(window)})")

    st = pd.concat([load_stints(s) for s in window], ignore_index=True)
    off, dfn, pts, poss, is_home, games = to_rows(st)
    y = 100 * pts / poss
    X, x_off, x_def, players = design_matrix(off, dfn, is_home)
    print(f"  {len(st):,} stints -> {X.shape[0]:,} rows, {len(players):,} players")

    alpha, cv = pick_alpha(X, y, poss, games)
    model = Ridge(alpha=alpha).fit(X, y, sample_weight=poss)
    p = len(players)
    coef = model.coef_

    # Raw on-court net rating, for comparison with RAPM
    off_pts, off_poss = x_off.T @ pts, x_off.T @ poss
    def_pts, def_poss = x_def.T @ pts, x_def.T @ poss
    with np.errstate(divide="ignore", invalid="ignore"):
        on_net = 100 * (off_pts / off_poss - def_pts / def_poss)

    names, teams = player_info(window, season)
    out = pd.DataFrame({
        "player_id": players,
        "o_rapm": coef[:p],
        "d_rapm": -coef[p:2 * p],          # flip sign: positive = holds opponents below average
        "off_poss": off_poss,
        "def_poss": def_poss,
        "on_court_net": on_net,
    })
    out["rapm"] = out["o_rapm"] + out["d_rapm"]
    out["player_name"] = out["player_id"].map(names)
    out["team"] = out["player_id"].map(teams)
    out["reliable"] = (out["off_poss"] + out["def_poss"]) >= MIN_POSS
    out["season"] = season
    out["window"] = " + ".join(window)
    out = out[["season", "window", "player_id", "player_name", "team", "rapm", "o_rapm", "d_rapm",
               "off_poss", "def_poss", "on_court_net", "reliable"]]
    out = out.round({"rapm": 2, "o_rapm": 2, "d_rapm": 2, "off_poss": 0, "def_poss": 0, "on_court_net": 2})
    out = out.sort_values("rapm", ascending=False).reset_index(drop=True)

    path = table_path("league/rapm", "rapm.parquet", season)
    out.to_parquet(path, index=False)
    meta = dict(season=season, window=window, alpha=alpha, cv_error=cv,
                rows=int(X.shape[0]), players=int(p),
                intercept=round(float(model.intercept_), 2),         # league points per 100 poss
                home_court=round(float(coef[-1]), 2))                # home offense bonus per 100 poss
    with open(table_path("league/rapm", "rapm_meta.json", season), "w") as f:
        json.dump(meta, f, indent=2)

    rel = out[out["reliable"]]
    print(f"  chosen alpha: {alpha}")
    print(f"  intercept (league pts/100): {meta['intercept']}   home court: {meta['home_court']:+}")
    print(f"  reliable players (>= {MIN_POSS} poss): {len(rel)}")
    print(f"  corr(RAPM, raw on-court net): {rel['rapm'].corr(rel['on_court_net']):.2f}")
    cols = ["player_name", "team", "rapm", "o_rapm", "d_rapm", "off_poss", "on_court_net"]
    print("  top 15:")
    print(rel[cols].head(15).to_string(index=False))
    print("  bottom 5:")
    print(rel[cols].tail(5).to_string(index=False))
    print(f"  saved {path}")


if __name__ == "__main__":
    for s in SEASONS:
        run(s)