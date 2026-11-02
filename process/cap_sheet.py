"""Step 5: 76ers cap sheet from hand-curated contracts.

Inputs (reference/):
  contracts_phi.csv      one row per player per season (you fill this in)
  cap_thresholds.csv     league cap, tax and apron levels and exception values
  team_cap_settings.csv  team-specific facts: BAE used last season, hard cap triggered

Outputs (data/cap/season=<current>/):
  contracts.parquet          cleaned contracts, every season
  commitments.parquet        total salary per future season
  cap_summary.parquet        current season: totals, room under each line, tier, exceptions
  phi_value.parquet          current salary vs RAPM, for the production-vs-cost chart
"""
import os
import re
import sys
import unicodedata

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import CURRENT_SEASON, TEAM_ID, TEAM_ABBR        # noqa: E402
from utils import table_path                                 # noqa: E402

REF = "reference"
CONTRACT_TYPES = {"standard", "rookie_scale", "minimum", "two_way", "exhibit_10", "dead_money"}
OPTIONS = {"", "PO", "TO", "ETO"}                 # player option, team option, early termination
FA_TYPES = {"", "UFA", "RFA"}
BIRD = {"", "Full", "Early", "Non", "None"}
NOT_ON_CAP = {"two_way", "exhibit_10"}            # don't count against the cap


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def money(v):
    """'$43,200,000' / '43.2M' / 43200000 -> 43200000."""
    if pd.isna(v) or str(v).strip() == "":
        return None
    s = str(v).strip().replace("$", "").replace(",", "").upper()
    if s.endswith("M"):
        return round(float(s[:-1]) * 1_000_000)
    return round(float(s))


def load_contracts():
    df = pd.read_csv(os.path.join(REF, "contracts_phi.csv"), dtype=str).fillna("")
    if df.empty:
        raise SystemExit("contracts_phi.csv is empty. Fill it in first.")
    errors = []
    for i, r in df.iterrows():
        line = i + 2                                              # +1 header, +1 for 1-based rows
        if not re.fullmatch(r"\d{4}-\d{2}", r["season"]):
            errors.append(f"row {line}: season '{r['season']}' should look like 2026-27")
        if r["contract_type"] not in CONTRACT_TYPES:
            errors.append(f"row {line}: contract_type '{r['contract_type']}' not in {sorted(CONTRACT_TYPES)}")
        if r["option"] not in OPTIONS:
            errors.append(f"row {line}: option '{r['option']}' not in PO / TO / ETO / blank")
        if r["fa_type"] not in FA_TYPES:
            errors.append(f"row {line}: fa_type '{r['fa_type']}' not in UFA / RFA / blank")
        if r["bird_rights"] not in BIRD:
            errors.append(f"row {line}: bird_rights '{r['bird_rights']}' not in Full / Early / Non / None / blank")
        try:
            if money(r["salary"]) is None:
                errors.append(f"row {line}: salary is blank")
        except ValueError:
            errors.append(f"row {line}: salary '{r['salary']}' is not a number")
    if errors:
        raise SystemExit("Fix these in contracts_phi.csv:\n  " + "\n  ".join(errors))

    df["salary"] = df["salary"].map(money)
    df["guaranteed"] = df["guaranteed"].map(money)
    df["guaranteed"] = df["guaranteed"].fillna(df["salary"])     # blank = fully guaranteed
    df["counts_on_cap"] = ~df["contract_type"].isin(NOT_ON_CAP)
    df["team"] = TEAM_ABBR
    return df


def attach_player_ids(df):
    """Match names to NBA player ids (needed to join RAPM)."""
    box = pd.read_parquet(table_path("league/player_box", "player_box.parquet", CURRENT_SEASON),
                          columns=["player_id", "player_name"])
    prev = pd.read_parquet(table_path("league/player_box", "player_box.parquet", "2025-26"),
                           columns=["player_id", "player_name"])
    lookup = pd.concat([prev, box]).drop_duplicates("player_id", keep="last")
    by_name = {}
    for pid, name in zip(lookup["player_id"], lookup["player_name"]):
        by_name.setdefault(norm(name), set()).add(int(pid))
    ids = df["player_name"].map(lambda n: next(iter(by_name.get(norm(n), {None}))) if len(by_name.get(norm(n), ())) == 1 else None)
    df["player_id"] = ids
    missing = df.loc[ids.isna() & (df["contract_type"] != "dead_money"), "player_name"].unique()
    if len(missing):
        print(f"  note: no NBA id matched for {list(missing)} (spelling, or hasn't played yet)")
    return df


def tier_and_exceptions(total, th, settings):
    cap, tax, a1, a2 = th["salary_cap"], th["tax_level"], th["first_apron"], th["second_apron"]
    bae_used = str(settings.get("bae_used_last_season", "")).strip().lower() in {"yes", "true", "1", "y"}
    hard_cap = str(settings.get("hard_cap", "")).strip()

    # If the team already used exceptions this season, report what's left instead of full amounts
    def remaining(col):
        v = money(settings.get(col, ""))
        return v

    if hard_cap in {"first_apron", "second_apron"}:
        limit = a1 if hard_cap == "first_apron" else a2
        tier = ("Over first apron" if total >= a1 else "Over tax" if total >= tax else
                "Over cap" if total >= cap else "Under cap")
        exceptions = []
        for col, label in (("mle_remaining", "MLE left"), ("bae_remaining", "BAE left")):
            v = remaining(col)
            if v is not None:
                exceptions.append(f"{label} ${v / 1e6:.3f}M")
        rules = [f"Hard-capped at the {hard_cap.replace('_', ' ')} (${limit / 1e6:.3f}M): "
                 f"total salary can't go above it at any point this season"]
        if total > limit:
            rules.append(f"WARNING: listed salary is ${(total - limit) / 1e6:.3f}M above the hard cap. "
                         f"Non-guaranteed contracts must be cut, or the listing is off. Check it.")
        else:
            rules.append(f"${(limit - total) / 1e6:.3f}M of room under the hard cap")
        return tier, exceptions or ["See mle_remaining / bae_remaining"], rules
    if total < cap:
        tier = "Under cap"
        exceptions = [f"Cap space ${(cap - total) / 1e6:.1f}M",
                      f"Room exception ${th['room_mle'] / 1e6:.3f}M (after using the space)"]
        rules = ["Using cap space removes the non-taxpayer MLE and BAE for the season"]
    elif total < a1:
        tier = "Over cap" if total < tax else "Over tax"
        exceptions = [f"Non-taxpayer MLE ${th['ntmle'] / 1e6:.3f}M (hard-caps at first apron if more than "
                      f"${th['tmle'] / 1e6:.3f}M is used)"]
        exceptions.append("BAE not available (used last season)" if bae_used
                          else f"BAE ${th['bae'] / 1e6:.3f}M (hard-caps at first apron)")
        rules = ["Standard salary matching in trades"]
        if total >= tax:
            rules.append("Pays luxury tax on salary above the tax line")
    elif total < a2:
        tier = "Over first apron"
        exceptions = [f"Taxpayer MLE ${th['tmle'] / 1e6:.3f}M (hard-caps at second apron)"]
        rules = ["Can't take back more salary than it sends out in trades",
                 "Can't use the non-taxpayer MLE or BAE",
                 "Can't acquire players in sign-and-trades"]
    else:
        tier = "Over second apron"
        exceptions = ["None"]
        rules = ["Can't aggregate salaries in trades", "Can't send cash in trades",
                 "Can't use any mid-level exception",
                 "Future first-round pick restrictions if it stays above"]
    return tier, exceptions, rules


def main():
    th = pd.read_csv(os.path.join(REF, "cap_thresholds.csv")).set_index("season").loc[CURRENT_SEASON]
    settings = pd.read_csv(os.path.join(REF, "team_cap_settings.csv"), dtype=str).fillna("")
    settings = settings[(settings["team"] == TEAM_ABBR) & (settings["season"] == CURRENT_SEASON)]
    settings = settings.iloc[0].to_dict() if len(settings) else {}

    c = attach_player_ids(load_contracts())
    c.to_parquet(table_path("cap", "contracts.parquet", CURRENT_SEASON), index=False)

    on_cap = c[c["counts_on_cap"]]
    commit = (on_cap.groupby("season")
                    .agg(total_salary=("salary", "sum"), guaranteed=("guaranteed", "sum"))
                    .join(on_cap[on_cap["contract_type"] != "dead_money"]
                          .groupby("season")["player_name"].nunique().rename("contracts"))
                    .fillna({"contracts": 0})
                    .reset_index())
    commit["non_guaranteed"] = commit["total_salary"] - commit["guaranteed"]
    commit["team"] = TEAM_ABBR
    commit.to_parquet(table_path("cap", "commitments.parquet", CURRENT_SEASON), index=False)

    cur = on_cap[on_cap["season"] == CURRENT_SEASON]
    total = float(cur["salary"].sum())
    tier, exceptions, rules = tier_and_exceptions(total, th, settings)
    summary = pd.DataFrame([{
        "team": TEAM_ABBR, "season": CURRENT_SEASON, "total_salary": total,
        "dead_money": float(cur.loc[cur["contract_type"] == "dead_money", "salary"].sum()),
        "roster_count": int(cur.loc[cur["contract_type"] != "dead_money", "player_name"].nunique()),
        "room_to_cap": th["salary_cap"] - total, "room_to_tax": th["tax_level"] - total,
        "room_to_first_apron": th["first_apron"] - total, "room_to_second_apron": th["second_apron"] - total,
        "below_min_salary_by": max(th["min_team_salary"] - total, 0),
        "tier": tier, "hard_cap": settings.get("hard_cap", "") or "None",
        "exceptions": " | ".join(exceptions), "key_rules": " | ".join(rules),
    }])
    summary.to_parquet(table_path("cap", "cap_summary.parquet", CURRENT_SEASON), index=False)

    # Production vs cost: current salary vs RAPM
    rapm = pd.read_parquet(table_path("league/rapm", "rapm.parquet", CURRENT_SEASON),
                           columns=["player_id", "rapm", "o_rapm", "d_rapm", "reliable"])
    value = cur[cur["contract_type"] != "dead_money"][["player_name", "player_id", "salary", "contract_type", "option"]]
    value = value.merge(rapm, on="player_id", how="left")
    value["cap_pct"] = (100 * value["salary"] / th["salary_cap"]).round(1)
    value.to_parquet(table_path("cap", "phi_value.parquet", CURRENT_SEASON), index=False)

    # ---------- printout ----------
    m = lambda v: f"${v / 1e6:,.2f}M"
    s = summary.iloc[0]
    print(f"\n{TEAM_ABBR} {CURRENT_SEASON} cap sheet")
    print(f"  total salary:     {m(s['total_salary'])}  ({s['roster_count']} players, dead money {m(s['dead_money'])})")
    for line in ("cap", "tax", "first_apron", "second_apron"):
        v = s[f"room_to_{line}"]
        print(f"  {'room to' if v >= 0 else 'over'} {line.replace('_', ' '):<13} {m(abs(v))}")
    print(f"  tier:             {s['tier']}   hard cap: {s['hard_cap']}")
    print(f"  exceptions:       {s['exceptions']}")
    print(f"  key rules:        {s['key_rules']}")
    print("\n  commitments by season:")
    print(commit[["season", "total_salary", "guaranteed", "contracts"]]
          .assign(total_salary=lambda d: d["total_salary"].map(m), guaranteed=lambda d: d["guaranteed"].map(m))
          .to_string(index=False))
    print("\n  salary vs RAPM:")
    print(value.sort_values("salary", ascending=False)[["player_name", "salary", "cap_pct", "rapm", "reliable"]]
          .assign(salary=lambda d: d["salary"].map(m)).to_string(index=False))


if __name__ == "__main__":
    main()