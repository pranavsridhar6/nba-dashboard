"""Step 8c: draft translator model, validation, projections and comparisons.

Target: years 1-3 box-score impact (same scale as RAPM: 0 = average NBA player), pulled toward
        replacement level (-2) for players who barely played, so busts count instead of being dropped.
Model:  ridge regression on college stats, weighted by NBA possessions played.
Tested: one draft class at a time (train on every other class, predict the held-out one),
        and compared against the NBA's own draft order (pick number) as a baseline.
Comps:  3 nearest past draftees by college profile (Mahalanobis distance).

Outputs (data/draft/):
  projections.parquet     2026 rookies (with their real season so far) + 2027 college prospects
  cv_predictions.parquet  every historical draftee: projected vs actual rookie impact
  model_meta.json         settings, validation results, coefficients
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import percentileofscore
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import DATA_DIR        # noqa: E402

D = os.path.join(DATA_DIR, "draft")
FEATURES = ["bpm", "obpm", "dbpm", "usg", "ts", "orb_pct", "drb_pct", "ast_pct", "tov_pct", "blk_pct",
            "stl_pct", "ftr", "fg3a_rate", "fg3_pct", "ft_pct", "height_in", "age_at_draft", "power_conf",
            "min_pct", "rec_score", "unranked"]
LAST_TRAIN_DRAFT = 2025
ROOKIE_CLASS, PROSPECT_YEAR = 2026, 2027
PROSPECT_MIN_GP, PROSPECT_MIN_MIN_PCT = 8, 40       # fewer games than this -> use last season's stats
FALLBACK_MIN_GP = 10
ALPHAS = np.logspace(-3, 5, 60)
TARGET, WEIGHT = "target_3yr", "target_weight"
POOL_REC, POOL_BPM_POWER, POOL_BPM_ANY = 90, 6, 9   # who counts as a draft-relevant prospect


def add_features(df):
    """High school recruiting rank: the scouts' talent rating before college. Unranked -> 0 + a flag."""
    df = df.copy()
    df["unranked"] = df["rec_rank"].isna().astype(int)
    df["rec_score"] = df["rec_rank"].fillna(0)
    return df
SHOW = ["ppg", "rpg", "apg", "bpm", "ts", "usg", "height_in", "age_at_draft"]


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS))


def boosting():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)


CANDIDATES = {"ridge": ridge, "boosting": boosting}


def fit(name, df):
    w = df[WEIGHT] / df[WEIGHT].mean()                 # normalized: keeps the ridge strength on a sane scale
    m = CANDIDATES[name]()
    if name == "ridge":
        m.fit(df[FEATURES], df[TARGET], ridgecv__sample_weight=w)
    else:
        m.fit(df[FEATURES], df[TARGET], sample_weight=w)
    return m


def spearman_by_class(df, col):
    vals = [g[TARGET].corr(g[col], method="spearman") for _, g in df.groupby("draft_year") if len(g) >= 5]
    return float(np.nanmean(vals))


def main():
    t = pd.read_parquet(os.path.join(D, "training.parquet"))
    college = pd.read_parquet(os.path.join(D, "college_all.parquet"))
    t, college = add_features(t), add_features(college)
    train = t[t["draft_year"] <= LAST_TRAIN_DRAFT].copy()
    train["log_pick"] = np.log(train["overall_pick"])

    print(f"recruiting rank available for {train['rec_rank'].notna().mean():.0%} of training draftees")

    # ---------- leave-one-draft-class-out validation, both candidates ----------
    preds = []
    for year in sorted(train["draft_year"].unique()):
        tr, te = train[train["draft_year"] != year], train[train["draft_year"] == year].copy()
        for name in CANDIDATES:
            te[f"proj_{name}"] = fit(name, tr).predict(te[FEATURES])
        base = LinearRegression().fit(tr[["log_pick"]], tr[TARGET], sample_weight=tr[WEIGHT])
        te["pick_baseline"] = base.predict(te[["log_pick"]])
        preds.append(te)
    cv = pd.concat(preds)
    w = cv[WEIGHT]

    def score(col):
        return {"r": round(float(cv[TARGET].corr(cv[col])), 3),
                "mae": round(float(np.average((cv[TARGET] - cv[col]).abs(), weights=w)), 3),
                "rank_within_class": round(spearman_by_class(cv, col), 3)}

    results = {name: score(f"proj_{name}") for name in CANDIDATES}
    results["draft_order"] = score("pick_baseline")
    best = max(CANDIDATES, key=lambda n: results[n]["rank_within_class"])   # ranking is what a draft board needs
    cv["projected"] = cv[f"proj_{best}"]
    metrics = {"players": int(len(cv)), "draft_classes": int(cv["draft_year"].nunique()),
               "chosen_model": best, "results": results}
    cv[["draft_year", "overall_pick", "nba_name", "school", "projected", "pick_baseline", TARGET,
        "poss_3yr", "seasons_played"]].to_parquet(os.path.join(D, "cv_predictions.parquet"), index=False)
    history = cv["projected"].to_numpy()

    # ---------- final models on every class ----------
    final = fit(best, train)
    lin = fit("ridge", train)                                # always kept: readable coefficients + comps space
    imp, sc, rcv = lin.named_steps["simpleimputer"], lin.named_steps["standardscaler"], lin.named_steps["ridgecv"]
    coefs = dict(sorted(zip(FEATURES, np.round(rcv.coef_, 3)), key=lambda kv: -abs(kv[1])))

    # ---------- comparisons: Mahalanobis distance in the model's standardized feature space ----------
    Z_train = sc.transform(imp.transform(train[FEATURES]))
    VI = np.linalg.pinv(np.cov(Z_train, rowvar=False))

    def comps(df):
        Z = sc.transform(imp.transform(df[FEATURES]))
        out = []
        for z in Z:
            diff = Z_train - z
            dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, VI, diff))
            idx = np.argsort(dist)[:3]
            out.append({f"comp{i + 1}": f"{train.iloc[j]['nba_name']} ({int(train.iloc[j]['draft_year'])}, "
                                        f"#{int(train.iloc[j]['overall_pick'])}): {train.iloc[j][TARGET]:+.1f}"
                        for i, j in enumerate(idx)})
        return pd.DataFrame(out, index=df.index)

    # ---------- 2026 rookies ----------
    rookies = t[t["draft_year"] == ROOKIE_CLASS].copy()
    rookies["group"] = f"{ROOKIE_CLASS} rookies"
    rookies["player_name"] = rookies["nba_name"]
    rookies["actual_so_far"] = rookies["rookie_impact"]
    rookies["poss_so_far"] = rookies["rookie_poss"]

    # ---------- 2027 prospects ----------
    # Players on a current college roster. Until they reach PROSPECT_MIN_GP games, use last season's
    # stats (returners only). Age is always measured at the 2027 draft.
    cur = college[college["year"] == PROSPECT_YEAR].copy()
    prev = college[(college["year"] == PROSPECT_YEAR - 1) & (college["gp"] >= FALLBACK_MIN_GP)
                   & (college["min_pct"] >= PROSPECT_MIN_MIN_PCT)].set_index("bart_id")
    enough = (cur["gp"] >= PROSPECT_MIN_GP) & (cur["min_pct"] >= PROSPECT_MIN_MIN_PCT)
    now = cur[enough].assign(stats_from=f"{PROSPECT_YEAR - 1}-{str(PROSPECT_YEAR)[2:]} (this season)")
    waiting = cur[~enough & cur["bart_id"].isin(prev.index)]
    last = prev.loc[waiting["bart_id"]].reset_index()
    last["school"], last["conf"] = waiting["school"].to_numpy(), waiting["conf"].to_numpy()   # current team
    bd = pd.to_datetime(last["birthdate"], errors="coerce")
    last["age_at_draft"] = (pd.Timestamp(f"{PROSPECT_YEAR}-06-25") - bd).dt.days / 365.25
    last["stats_from"] = f"{PROSPECT_YEAR - 2}-{str(PROSPECT_YEAR - 1)[2:]} (last season)"
    pros = add_features(pd.concat([now, last], ignore_index=True))
    in_pool = ((pros["rec_score"] >= POOL_REC) | ((pros["bpm"] >= POOL_BPM_POWER) & (pros["power_conf"] == 1))
               | (pros["bpm"] >= POOL_BPM_ANY))
    pros = pros[in_pool]
    pros["group"] = f"{PROSPECT_YEAR} prospects"
    pros["player_name"] = pros["player"]
    pros["overall_pick"] = np.nan
    rookies["stats_from"] = "final college season"

    proj = pd.concat([rookies, pros], ignore_index=True)
    proj["projected"] = final.predict(proj[FEATURES])
    proj["percentile"] = [round(percentileofscore(history, v), 0) for v in proj["projected"]]   # vs past draftees
    proj = pd.concat([proj, comps(proj)], axis=1)
    keep = ["group", "player_name", "school", "conf", "stats_from", "overall_pick", "gp", "rec_rank", "projected", "percentile", "actual_so_far",
            "poss_so_far", "comp1", "comp2", "comp3"] + SHOW
    proj = proj[[c for c in keep if c in proj.columns]].round(2)
    proj.to_parquet(os.path.join(D, "projections.parquet"), index=False)

    meta = dict(target="years 1-3 box-score impact, shrunk toward replacement (-2) with 500 pseudo-possessions",
                train_drafts=f"2010-{LAST_TRAIN_DRAFT}", ridge_alpha=float(rcv.alpha_), features=FEATURES,
                validation=metrics, standardized_coefficients=coefs)
    json.dump(meta, open(os.path.join(D, "model_meta.json"), "w"), indent=2, default=float)

    # ---------- printout ----------
    print(f"trained on {len(train)} draftees ({metrics['draft_classes']} draft classes), ridge alpha {rcv.alpha_:.2f}"
          + ("  (WARNING: at the edge of the search range)" if rcv.alpha_ in (ALPHAS[0], ALPHAS[-1]) else ""))
    print("\nvalidation (each class predicted by models that never saw it):")
    print(f"  {'':12} {'correlation':>12} {'avg error':>10} {'rank within class':>18}")
    for name, r in results.items():
        print(f"  {name:12} {r['r']:>12} {r['mae']:>10} {r['rank_within_class']:>18}")
    print(f"  chosen: {best}")
    print("\nmost important inputs (ridge, standardized coefficients):")
    for k, v in list(coefs.items())[:8]:
        print(f"  {k:>14}: {v:+.3f}")
    r = proj[proj["group"].str.contains("rookies")].sort_values("projected", ascending=False)
    print(f"\n{ROOKIE_CLASS} rookies, top 10 projected (actual so far is a small sample):")
    print(r[["player_name", "overall_pick", "projected", "percentile", "actual_so_far", "comp1"]].head(10).to_string(index=False))
    p = proj[proj["group"].str.contains("prospects")].sort_values("projected", ascending=False)
    print(f"\n{PROSPECT_YEAR} prospects, top 10 projected ({len(p)} players qualify so far):")
    print(p[["player_name", "school", "stats_from", "rec_rank", "projected", "percentile", "comp1"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()