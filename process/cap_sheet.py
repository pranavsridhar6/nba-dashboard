"""Step 9b: cap sheets for all 30 teams, from the scraped contracts (ingest/10_contracts.py).

Inputs:
  data/contracts/contracts_all.parquet     every contract (ShamSports, with reference/contract_overrides.csv applied)
  data/contracts/team_exceptions.parquet   exception money each team has left
  reference/cap_thresholds.csv             league cap, tax, aprons, exception sizes
  reference/team_cap_settings.csv          optional hand overrides per team (e.g. a hard cap from a sign-and-trade)

Hard cap: not listed on the source pages, so it is inferred from exception use (CBA rule: using the bi-annual,
or more of the mid-level than the taxpayer amount, hard-caps a team at the first apron; using only the
taxpayer mid-level hard-caps at the second apron). A team_cap_settings.csv row always wins.

Outputs (data/cap/season=<current>/):
  cap_summary.parquet    one row per team: totals, room under each line, tier, hard cap, exceptions, rules
  contracts.parquet      every contract, with NBA player ids
  commitments.parquet    total and guaranteed salary per team per future season
  player_value.parquet   current salary vs RAPM, every player (production-vs-cost chart)
"""
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import CURRENT_SEASON, DATA_DIR        # noqa: E402
from utils import table_path                        # noqa: E402

NOT_ON_CAP = {"two_way", "exhibit_10"}
CON = os.path.join(DATA_DIR, "contracts")


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def m(v):
    return f"${v / 1e6:,.3f}M"


def attach_ids(df):
    """NBA player ids: exact name first, then same team + last name + first initial
    (spelling differences like Dominic / Dominick Barlow)."""
    frames = []
    for s in ("2025-26", CURRENT_SEASON):
        p = table_path("league/player_box", "player_box.parquet", s)
        if os.path.exists(p):
            frames.append(pd.read_parquet(p, columns=["player_id", "player_name", "team"]))
    box = pd.concat(frames)
    lookup = box.drop_duplicates("player_id", keep="last")
    by_name = lookup.assign(k=lookup["player_name"].map(norm)).groupby("k")["player_id"].agg(set)

    def last_initial(n):
        parts = re.sub(r"\b(Jr|Sr|II|III|IV)\b\.?", "", str(n)).split()
        return (norm(parts[-1]), norm(parts[0])[:1]) if parts else ("", "")

    by_team = {}
    for r in box.drop_duplicates(["team", "player_id"]).itertuples(index=False):
        by_team.setdefault((r.team,) + last_initial(r.player_name), set()).add(r.player_id)

    def find(name, team):
        k = norm(name)
        if k in by_name and len(by_name[k]) == 1:
            return next(iter(by_name[k]))
        ids = by_team.get((team,) + last_initial(name), set())
        return next(iter(ids)) if len(ids) == 1 else None

    df["player_id"] = [find(n, t) for n, t in zip(df["player_name"], df["team"])]
    return df


def settings_for(team, exc, manual, th, total):
    """Exception money left and hard cap for one team (manual row > inferred from exceptions).

    A team below the first apron has the non-taxpayer MLE; using more than the taxpayer amount of it
    (or any of the bi-annual) hard-caps it at the first apron. A team over the first apron only has the
    taxpayer MLE; using any of it hard-caps it at the second apron.
    """
    tol = 10_000                                    # ignore rounding-sized differences
    e = exc.get(team, {})
    mle, room, bae = e.get("mle_remaining"), e.get("room_mle_remaining"), e.get("bae_remaining")
    hard, source = "", ""
    if bae is not None and bae < th["bae"] - tol:
        hard, source = "first_apron", "inferred: bi-annual exception used"
    elif mle is not None and total < th["first_apron"] and mle < th["ntmle"] - th["tmle"] - tol:
        hard, source = "first_apron", "inferred: mid-level used beyond the taxpayer amount"
    elif mle is not None and total >= th["first_apron"] and mle < th["tmle"] - tol:
        hard, source = "second_apron", "inferred: taxpayer mid-level used"
    mrow = manual.get(team)
    if mrow is not None:
        if str(mrow.get("hard_cap", "")).strip():
            hard, source = mrow["hard_cap"].strip(), "set by hand (team_cap_settings.csv)"
        if str(mrow.get("mle_remaining", "")).strip():
            mle = float(mrow["mle_remaining"])
        if str(mrow.get("bae_remaining", "")).strip():
            bae = float(mrow["bae_remaining"])
    return mle, room, bae, hard, source


def tier_and_rules(total, th, mle, room, bae, hard):
    cap, tax, a1, a2 = th["salary_cap"], th["tax_level"], th["first_apron"], th["second_apron"]
    if total < cap:
        tier = "Under cap"
    elif total < tax:
        tier = "Over cap"
    elif total < a1:
        tier = "Over tax"
    elif total < a2:
        tier = "Over first apron"
    else:
        tier = "Over second apron"

    parts = []
    if total < cap:
        parts.append(f"Cap space {m(cap - total)}")
    if mle is not None:
        parts.append(f"MLE left {m(mle)}")
    if room is not None:
        parts.append(f"Room MLE left {m(room)}")
    if bae is not None:
        parts.append(f"BAE left {m(bae)}")
    exceptions = " | ".join(parts) or "No exceptions available"

    rules = {"Under cap": ["Can sign free agents into cap space"],
             "Over cap": ["Standard salary matching in trades"],
             "Over tax": ["Pays luxury tax on salary above the tax line"],
             "Over first apron": ["Can't take back more salary than it sends out in trades",
                                  "Can't use the non-taxpayer MLE or BAE",
                                  "Can't acquire players in sign-and-trades"],
             "Over second apron": ["Can't aggregate salaries in trades", "Can't send cash in trades",
                                   "Can't use any mid-level exception",
                                   "Future first-round pick restrictions if it stays above"]}[tier]
    if hard:
        limit = a1 if hard == "first_apron" else a2
        rules = [f"Hard-capped at the {hard.replace('_', ' ')} ({m(limit)})"] + rules
        if total > limit:
            rules.append(f"WARNING: listed salary is {m(total - limit)} above the hard cap; "
                         f"non-guaranteed deals must be cut or salary moved")
    return tier, exceptions, " | ".join(rules)


def main():
    th = pd.read_csv(os.path.join("reference", "cap_thresholds.csv")).set_index("season").loc[CURRENT_SEASON]
    c = pd.read_parquet(os.path.join(CON, "contracts_all.parquet"))
    exc = pd.read_parquet(os.path.join(CON, "team_exceptions.parquet"))
    exc = {r["team"]: {k: (None if pd.isna(v) else v) for k, v in r.items()} for r in exc.to_dict("records")}
    mp = os.path.join("reference", "team_cap_settings.csv")
    manual = {}
    if os.path.exists(mp):
        ms = pd.read_csv(mp, dtype=str).fillna("")
        manual = {r["team"]: r for r in ms[ms["season"] == CURRENT_SEASON].to_dict("records")}

    c = attach_ids(c)
    c["counts_on_cap"] = ~c["contract_type"].isin(NOT_ON_CAP)
    c.to_parquet(table_path("cap", "contracts.parquet", CURRENT_SEASON), index=False)

    on_cap = c[c["counts_on_cap"]]
    commit = (on_cap.groupby(["team", "season"])
                    .agg(total_salary=("salary", "sum"), guaranteed=("guaranteed", "sum"),
                         contracts=("player_name", "nunique")).reset_index())
    commit["non_guaranteed"] = commit["total_salary"] - commit["guaranteed"]
    commit.to_parquet(table_path("cap", "commitments.parquet", CURRENT_SEASON), index=False)

    cur = on_cap[on_cap["season"] == CURRENT_SEASON]
    rows = []
    for team, g in cur.groupby("team"):
        total = float(g["salary"].sum())
        mle, room, bae, hard, source = settings_for(team, exc, manual, th, total)
        tier, exceptions, rules = tier_and_rules(total, th, mle, room, bae, hard)
        rows.append(dict(team=team, season=CURRENT_SEASON, total_salary=total,
                         guaranteed=float(g["guaranteed"].sum()), roster_count=int(g["player_name"].nunique()),
                         room_to_cap=th["salary_cap"] - total, room_to_tax=th["tax_level"] - total,
                         room_to_first_apron=th["first_apron"] - total,
                         room_to_second_apron=th["second_apron"] - total,
                         below_min_salary_by=max(th["min_team_salary"] - total, 0),
                         tier=tier, hard_cap=hard or "None", hard_cap_source=source or "",
                         mle_remaining=mle, bae_remaining=bae, exceptions=exceptions, key_rules=rules,
                         source_url=g["source_url"].iloc[0], scraped_at=g["scraped_at"].iloc[0]))
    summary = pd.DataFrame(rows).sort_values("total_salary", ascending=False)
    summary.to_parquet(table_path("cap", "cap_summary.parquet", CURRENT_SEASON), index=False)

    rapm = pd.read_parquet(table_path("league/rapm", "rapm.parquet", CURRENT_SEASON),
                           columns=["player_id", "rapm", "o_rapm", "d_rapm", "reliable"])
    value = cur[["team", "player_name", "player_id", "salary", "contract_type", "option"]].merge(
        rapm, on="player_id", how="left")
    value["cap_pct"] = (100 * value["salary"] / th["salary_cap"]).round(1)
    value.to_parquet(table_path("cap", "player_value.parquet", CURRENT_SEASON), index=False)

    # ---------- printout ----------
    print(f"{len(summary)} teams, {cur['player_name'].nunique()} players under contract for {CURRENT_SEASON}")
    print(f"player ids matched: {c['player_id'].notna().mean():.1%} of contract rows "
          f"(unmatched are mostly rookies who haven't played yet)")
    print(f"tiers: {summary['tier'].value_counts().to_dict()}")
    print(f"hard-capped: {(summary['hard_cap'] != 'None').sum()} teams\n")
    show = summary[["team", "total_salary", "tier", "hard_cap", "exceptions"]].copy()
    show["total_salary"] = show["total_salary"].map(lambda v: f"${v / 1e6:,.1f}M")
    print(show.to_string(index=False))
    print("\ncheck PHI (should match the old hand-made sheet: $209.5M, over first apron, hard-capped first apron):")
    p = summary[summary["team"] == "PHI"].iloc[0]
    print(f"  {m(p['total_salary'])} | {p['tier']} | hard cap: {p['hard_cap']} ({p['hard_cap_source']}) | {p['exceptions']}")


if __name__ == "__main__":
    main()