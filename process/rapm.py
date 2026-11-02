"""Step 3c (v2): RAPM with garbage time removed, recency weights, a box-score prior and error bars.

For each season S the window is S + the season before it (regular season only).
  1. Basic RAPM: ridge regression shrinking every player toward 0 (average).  Kept as *_basic.
  2. Box-score prior: a model learns how box stats per 100 possessions relate to basic RAPM,
     separately for offense and defense. Its prediction is each player's starting point.
  3. Final RAPM: ridge regression shrinking each player toward his prior instead of toward 0.
     Lots of possessions -> the data wins. Few possessions -> the prior carries more weight.
  4. Error bars: refit on games resampled with replacement; the spread of results is the uncertainty.
Rows are weighted by possessions, and the previous season counts PREV_SEASON_WEIGHT as much.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import SEASONS          # noqa: E402
from utils import table_path        # noqa: E402
from lineups import tag_garbage     # noqa: E402  same garbage-time rule as the lineup tool

ALPHAS = [100, 300, 1000, 3000, 10000, 30000, 100000]
FOLDS = 5
PREV_SEASON_WEIGHT = 0.5
N_BOOT = 40
MIN_POSS = 4000                 # offense + defense possessions before a RAPM is "reliable" (~1,000 minutes)
BOX_SHRINK_POSS = 1000          # per-100 box stats are pulled toward league average by this many possessions
TS_SHRINK_ATTEMPTS = 200        # true shooting % pulled toward league average by this many shot attempts
BOX_MIN_OFF_POSS = 1500         # players used to fit the box-score prior
STATS = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
         "ast", "stl", "blk", "tov", "pf", "minutes"]
BOX_FEATURES = ["pts100", "fg3a100", "fta100", "ast100", "oreb100", "dreb100",
                "stl100", "blk100", "tov100", "pf100", "ts_pct"]
HOME = [f"home_p{i}" for i in range(1, 6)]
AWAY = [f"away_p{i}" for i in range(1, 6)]


# ---------- data ----------

def load_stints(season):
    """Regular season, full lineups, games whose points add up, garbage time removed."""
    st = pd.read_parquet(table_path("league/stints", "stints.parquet", season))
    chk = pd.read_parquet(table_path("league/stints", "game_checks.parquet", season))
    good = set(chk.loc[chk["pts_ok"], "game_id"])
    st = st[st["game_id"].str[:3].eq("002") & st["lineup_complete"] & st["game_id"].isin(good)].copy()
    box = pd.read_parquet(table_path("league/player_box", "player_box.parquet", season),
                          columns=["game_id", "player_id", "position"])
    st["garbage"] = tag_garbage(st, box)
    return st[~st["garbage"]], st["garbage"].mean()


def to_rows(st, season_weight):
    """Every stint -> 2 rows: home team on offense, then away team on offense."""
    off = np.vstack([st[HOME].to_numpy(), st[AWAY].to_numpy()]).astype(np.int64)
    dfn = np.vstack([st[AWAY].to_numpy(), st[HOME].to_numpy()]).astype(np.int64)
    pts = np.concatenate([st["home_pts"].to_numpy(), st["away_pts"].to_numpy()]).astype(float)
    poss = np.concatenate([st["home_poss"].to_numpy(), st["away_poss"].to_numpy()]).astype(float)
    is_home = np.concatenate([np.ones(len(st)), np.zeros(len(st))])
    games = np.concatenate([st["game_id"].to_numpy(), st["game_id"].to_numpy()])
    sw = np.full(len(off), season_weight)
    keep = poss > 0
    return off[keep], dfn[keep], pts[keep], poss[keep], is_home[keep], games[keep], sw[keep]


def design_matrix(off, dfn, is_home):
    """Sparse X: [offense columns | defense columns | home column]."""
    players = np.unique(np.concatenate([off.ravel(), dfn.ravel()]))
    n, p = len(off), len(players)
    rows = np.repeat(np.arange(n), 5)
    x_off = sparse.csr_matrix((np.ones(n * 5), (rows, np.searchsorted(players, off.ravel()))), shape=(n, p))
    x_def = sparse.csr_matrix((np.ones(n * 5), (rows, np.searchsorted(players, dfn.ravel()))), shape=(n, p))
    X = sparse.hstack([x_off, x_def, sparse.csr_matrix(is_home.reshape(-1, 1))]).tocsr()
    return X, x_off, x_def, players


def box_totals(window):
    frames = []
    for s in window:
        b = pd.read_parquet(table_path("league/player_box", "player_box.parquet", s),
                            columns=["game_id", "player_id"] + STATS)
        frames.append(b[b["game_id"].str[:3] == "002"])
    return pd.concat(frames).groupby("player_id")[STATS].sum()


def season_context(season):
    """Games, minutes and points per game in the selected season (what fans see)."""
    b = pd.read_parquet(table_path("league/player_box", "player_box.parquet", season),
                        columns=["game_id", "player_id", "player_name", "team", "minutes", "pts"])
    b = b[(b["game_id"].str[:3] == "002") & (b["minutes"] > 0)]
    g = b.groupby("player_id").agg(games=("game_id", "nunique"), minutes=("minutes", "sum"), pts=("pts", "sum"))
    g["mpg"] = g["minutes"] / g["games"]
    g["ppg"] = g["pts"] / g["games"]
    return g[["games", "mpg", "ppg"]]


def top_teammates(st, names):
    """For each player: the teammate he shared the most minutes with, and what share of his minutes."""
    recs = []
    secs = st["seconds"].to_numpy()
    for cols in (HOME, AWAY):
        P = st[cols].to_numpy().astype(np.int64)
        for i in range(5):
            for j in range(5):
                if i != j:
                    recs.append(pd.DataFrame({"player_id": P[:, i], "mate": P[:, j], "s": secs}))
    pairs = pd.concat(recs).groupby(["player_id", "mate"])["s"].sum().reset_index()
    total = pd.concat([pd.DataFrame({"player_id": st[c].astype(np.int64), "s": st["seconds"]})
                       for c in HOME + AWAY]).groupby("player_id")["s"].sum()
    top = pairs.sort_values("s").groupby("player_id").last()
    top["top_teammate"] = top["mate"].map(names)
    top["top_teammate_share"] = (100 * top["s"] / total.reindex(top.index)).round(1)
    return top[["top_teammate", "top_teammate_share"]]


def player_info(window, season):
    boxes = []
    for s in window:
        b = pd.read_parquet(table_path("league/player_box", "player_box.parquet", s),
                            columns=["player_id", "player_name", "team", "minutes"])
        b["season"] = s
        boxes.append(b)
    box = pd.concat(boxes)
    names = box.sort_values("season").groupby("player_id")["player_name"].last()
    def main_team(b):
        return (b.groupby(["player_id", "team"])["minutes"].sum().reset_index()
                 .sort_values("minutes").groupby("player_id")["team"].last())

    latest = box[box["season"] == box.groupby("player_id")["season"].transform("max")]
    teams = main_team(box[box["season"] == season]).combine_first(main_team(latest))
    return names, teams


# ---------- model ----------

def pick_alpha(X, y, w, groups, alphas):
    results = {}
    for a in alphas:
        errs = []
        for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, groups):
            m = Ridge(alpha=a).fit(X[tr], y[tr], sample_weight=w[tr])
            errs.append(np.average((y[te] - m.predict(X[te])) ** 2, weights=w[te]))
        results[a] = float(np.mean(errs))
    return min(results, key=results.get), results


def prepare(season):
    i = SEASONS.index(season)
    window = SEASONS[max(0, i - 1): i + 1]
    parts, garbage_share = [], {}
    for s in window:
        st, g = load_stints(s)
        garbage_share[s] = round(float(g), 4)
        parts.append((st, 1.0 if s == season else PREV_SEASON_WEIGHT))
    stints = pd.concat([p[0] for p in parts], ignore_index=True)
    rows = [to_rows(st, sw) for st, sw in parts]
    off, dfn, pts, poss, is_home, games, sw = (np.concatenate(z) for z in zip(*rows))
    X, x_off, x_def, players = design_matrix(off, dfn, is_home)

    box = box_totals(window).reindex(players)
    off_poss = np.asarray(x_off.T @ poss).ravel()
    def_poss = np.asarray(x_def.T @ poss).ravel()
    # Per-100 box stats, pulled toward the league average for players with few possessions.
    # Without this, a player with 300 possessions and a hot week gets an extreme prior.
    per100 = pd.DataFrame(index=players)
    has_box = box["pts"].notna().to_numpy()
    for stat in ["pts", "fg3a", "fta", "ast", "oreb", "dreb", "stl", "blk", "tov", "pf"]:
        total = box[stat].fillna(0).to_numpy()
        league_rate = total[has_box].sum() / off_poss[has_box].sum()
        per100[f"{stat}100"] = 100 * (total + league_rate * BOX_SHRINK_POSS) / (off_poss + BOX_SHRINK_POSS)
    att = (box["fga"].fillna(0) + 0.44 * box["fta"].fillna(0)).to_numpy()
    pts_all = box["pts"].fillna(0).to_numpy()
    league_ts = pts_all[has_box].sum() / (2 * att[has_box].sum())
    per100["ts_pct"] = (pts_all + league_ts * 2 * TS_SHRINK_ATTEMPTS) / (2 * (att + TS_SHRINK_ATTEMPTS))
    per100.loc[~has_box] = np.nan

    names, teams = player_info(window, season)
    return dict(season=season, window=window, garbage_share=garbage_share, stints=stints,
                X=X, x_off=x_off, x_def=x_def, players=players,
                y=100 * pts / poss, w=poss * sw, poss=poss, pts=pts, groups=games,
                off_poss=off_poss, def_poss=def_poss, per100=per100, names=names, teams=teams)


def bootstrap_sd(X, y_adj, w, groups, alpha, p):
    codes, uniq = pd.factorize(groups)
    rng = np.random.default_rng(0)
    totals = []
    for b in range(N_BOOT):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        wb = w * counts[codes]
        keep = wb > 0
        coef = Ridge(alpha=alpha).fit(X[keep], y_adj[keep], sample_weight=wb[keep]).coef_
        totals.append(coef[:p] - coef[p:2 * p])            # offense - defense coefficient = total RAPM
        if (b + 1) % 10 == 0:
            print(f"    bootstrap {b + 1}/{N_BOOT}")
    return np.array(totals).std(axis=0)


def main():
    data = {}
    print("Stage 1: basic RAPM (shrink toward 0)")
    for s in SEASONS:
        d = prepare(s)
        alpha, cv = pick_alpha(d["X"], d["y"], d["w"], d["groups"], ALPHAS)
        coef = Ridge(alpha=alpha).fit(d["X"], d["y"], sample_weight=d["w"]).coef_
        p = len(d["players"])
        d.update(alpha_basic=alpha, cv_basic=cv, o_basic=coef[:p], d_basic=-coef[p:2 * p])
        data[s] = d
        print(f"  {s}: alpha {alpha}, garbage time removed {d['garbage_share']}")

    # ---------- box-score prior ----------
    train = []
    for s, d in data.items():
        f = d["per100"].copy()
        f["o"], f["d"], f["poss"] = d["o_basic"], d["d_basic"], d["off_poss"]
        train.append(f)
    train = pd.concat(train).dropna()
    train = train[train["poss"] >= BOX_MIN_OFF_POSS]
    o_model = LinearRegression().fit(train[BOX_FEATURES], train["o"], sample_weight=train["poss"])
    d_model = LinearRegression().fit(train[BOX_FEATURES], train["d"], sample_weight=train["poss"])
    r2_o = o_model.score(train[BOX_FEATURES], train["o"], sample_weight=train["poss"])
    r2_d = d_model.score(train[BOX_FEATURES], train["d"], sample_weight=train["poss"])
    print(f"\nBox-score prior fit on {len(train)} player-windows: R2 offense {r2_o:.2f}, defense {r2_d:.2f}")

    print("\nStage 2: RAPM with box-score prior + error bars")
    for s, d in data.items():
        p = len(d["players"])
        feats = d["per100"][BOX_FEATURES]
        ok = feats.notna().all(axis=1).to_numpy()
        o_prior, d_prior = np.zeros(p), np.zeros(p)
        o_prior[ok] = o_model.predict(feats[ok])
        d_prior[ok] = d_model.predict(feats[ok])
        b0 = np.concatenate([o_prior, -d_prior, [0.0]])              # defense columns are sign-flipped
        y_adj = d["y"] - d["X"] @ b0                                 # fit the deviation from the prior

        alpha, cv = pick_alpha(d["X"], y_adj, d["w"], d["groups"], ALPHAS)
        model = Ridge(alpha=alpha).fit(d["X"], y_adj, sample_weight=d["w"])
        coef = model.coef_ + b0
        sd = bootstrap_sd(d["X"], y_adj, d["w"], d["groups"], alpha, p)

        x_off, x_def = d["x_off"], d["x_def"]
        with np.errstate(divide="ignore", invalid="ignore"):
            on_net = 100 * (np.asarray(x_off.T @ d["pts"]).ravel() / d["off_poss"]
                            - np.asarray(x_def.T @ d["pts"]).ravel() / d["def_poss"])

        out = pd.DataFrame({"player_id": d["players"], "o_rapm": coef[:p], "d_rapm": -coef[p:2 * p],
                            "rapm_sd": sd, "prior_rapm": o_prior + d_prior,
                            "o_rapm_basic": d["o_basic"], "d_rapm_basic": d["d_basic"],
                            "off_poss": d["off_poss"], "def_poss": d["def_poss"], "on_court_net": on_net})
        out["rapm"] = out["o_rapm"] + out["d_rapm"]
        out["rapm_basic"] = out["o_rapm_basic"] + out["d_rapm_basic"]
        out["rapm_low"] = out["rapm"] - 1.645 * out["rapm_sd"]         # 90% range
        out["rapm_high"] = out["rapm"] + 1.645 * out["rapm_sd"]
        out["player_name"] = out["player_id"].map(d["names"])
        out["team"] = out["player_id"].map(d["teams"])
        out = out.join(season_context(s), on="player_id")
        out = out.join(d["per100"][["pts100", "ts_pct"]], on="player_id")
        out = out.join(top_teammates(d["stints"], d["names"].to_dict()), on="player_id")
        out["reliable"] = (out["off_poss"] + out["def_poss"]) >= MIN_POSS
        out["played_this_season"] = out["games"].notna()
        out["season"], out["window"] = s, " + ".join(d["window"])
        cols = ["season", "window", "player_id", "player_name", "team",
                "rapm", "o_rapm", "d_rapm", "rapm_sd", "rapm_low", "rapm_high",
                "prior_rapm", "rapm_basic", "o_rapm_basic", "d_rapm_basic",
                "off_poss", "def_poss", "on_court_net", "games", "mpg", "ppg", "pts100", "ts_pct",
                "top_teammate", "top_teammate_share", "reliable", "played_this_season"]
        out = out[cols].round(2).sort_values("rapm", ascending=False).reset_index(drop=True)
        path = table_path("league/rapm", "rapm.parquet", s)
        out.to_parquet(path, index=False)

        meta = dict(season=s, window=d["window"], prev_season_weight=PREV_SEASON_WEIGHT,
                    garbage_share_removed=d["garbage_share"],
                    alpha_basic=d["alpha_basic"], cv_basic=d["cv_basic"], alpha=alpha, cv=cv,
                    n_boot=N_BOOT, rows=int(d["X"].shape[0]), players=int(p),
                    intercept=round(float(model.intercept_), 2), home_court=round(float(coef[-1]), 2),
                    box_prior=dict(features=BOX_FEATURES, r2_offense=round(r2_o, 3), r2_defense=round(r2_d, 3),
                                   offense_coefs=dict(zip(BOX_FEATURES, np.round(o_model.coef_, 4).tolist())),
                                   defense_coefs=dict(zip(BOX_FEATURES, np.round(d_model.coef_, 4).tolist()))))
        with open(table_path("league/rapm", "rapm_meta.json", s), "w") as f:
            json.dump(meta, f, indent=2)

        rel = out[out["reliable"]].copy()
        rel["change"] = rel["rapm"] - rel["rapm_basic"]
        show = ["player_name", "team", "rapm", "rapm_low", "rapm_high", "rapm_basic", "prior_rapm", "on_court_net"]
        print(f"\n{s} (window {' + '.join(d['window'])}): alpha {alpha}, "
              f"intercept {meta['intercept']}, home {meta['home_court']:+}")
        print("  top 10:")
        print(rel[show].head(10).to_string(index=False))
        print("  biggest risers vs basic:")
        print(rel.sort_values("change", ascending=False)[show].head(5).to_string(index=False))
        print("  biggest fallers vs basic:")
        print(rel.sort_values("change")[show].head(5).to_string(index=False))
        watch = rel[rel["player_name"].isin(["Jaylen Brown", "LeBron James"])]
        if len(watch):
            print("  watch list:")
            print(watch[show + ["top_teammate", "top_teammate_share"]].to_string(index=False))
        print(f"  saved {path}")


if __name__ == "__main__":
    main()