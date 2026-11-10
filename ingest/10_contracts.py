"""Step 9b: every NBA contract, from ShamSports team salary pages (all 30 teams).

Polite by design: checks robots.txt, waits between pages, and re-downloads a team page at most once a day
(saved copies in data/contracts/raw/). Source mistakes can be corrected in reference/contract_overrides.csv.

Outputs (data/contracts/):
  contracts_all.parquet   one row per player per season: salary, guaranteed, contract_type, option, notes
  cap_holds_all.parquet   free agents each team holds rights to (cap hold, qualifying offer, Bird years)
  page_totals.parquet     each page's own "Total salary", used to check the parse
"""
import io
import os
import re
import time
import urllib.robotparser
from datetime import datetime, timedelta

import pandas as pd
import requests

from config import DATA_DIR

OUT = os.path.join(DATA_DIR, "contracts")
RAW = os.path.join(OUT, "raw")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
BASE = "https://www.shamsports.com/salaries/{slug}/"
DELAY, MAX_AGE = 6, timedelta(hours=20)
TEAMS = {"ATL": "atlanta-hawks", "BOS": "boston-celtics", "BKN": "brooklyn-nets", "CHA": "charlotte-hornets",
         "CHI": "chicago-bulls", "CLE": "cleveland-cavaliers", "DAL": "dallas-mavericks", "DEN": "denver-nuggets",
         "DET": "detroit-pistons", "GSW": "golden-state-warriors", "HOU": "houston-rockets", "IND": "indiana-pacers",
         "LAC": "los-angeles-clippers", "LAL": "los-angeles-lakers", "MEM": "memphis-grizzlies", "MIA": "miami-heat",
         "MIL": "milwaukee-bucks", "MIN": "minnesota-timberwolves", "NOP": "new-orleans-pelicans",
         "NYK": "new-york-knicks", "OKC": "oklahoma-city-thunder", "ORL": "orlando-magic",
         "PHI": "philadelphia-76ers", "PHX": "phoenix-suns", "POR": "portland-trail-blazers",
         "SAC": "sacramento-kings", "SAS": "san-antonio-spurs", "TOR": "toronto-raptors", "UTA": "utah-jazz",
         "WAS": "washington-wizards"}
OPTIONS = {"player option": "PO", "team option": "TO", "early termination option": "ETO"}
KNOWN = ["stretch provision", "rookie-scale", "trade bonus", "option", "unguaranteed", "guaranteed",
         "non-guaranteed", "two-way", "exhibit 10", "minimum", "waived"]


def money(v):
    s = str(v).replace("$", "").replace(",", "").strip()
    return int(s) if s.isdigit() else None


def season(label):                      # "2026/27" -> "2026-27"
    return label.replace("/", "-")


def get_page(abbr, slug, robots):
    path = os.path.join(RAW, f"{abbr}.html")
    if os.path.exists(path) and datetime.now() - datetime.fromtimestamp(os.path.getmtime(path)) < MAX_AGE:
        return open(path, encoding="utf-8").read(), "saved copy"
    url = BASE.format(slug=slug)
    if not robots.can_fetch(UA, url):
        raise SystemExit(f"robots.txt no longer allows {url}. Stopping.")
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    open(path, "w", encoding="utf-8").write(r.text)
    time.sleep(DELAY)
    return r.text, "downloaded"


def parse_notes(notes, seasons):
    """Turn ShamSports' notes sentence into options, guarantees and contract type."""
    low = notes.lower()
    opt, guar = {}, {}
    for s in seasons:
        lab = s.replace("-", "/")
        m = re.search(re.escape(lab) + r":\s*([^.]*)", notes, re.IGNORECASE)
        if not m:
            continue
        clause = m.group(1).lower()
        for phrase, code in OPTIONS.items():
            if phrase in clause:
                opt[s] = code
        if "unguaranteed" in clause or "non-guaranteed" in clause:
            guar[s] = 0
        pg = (re.search(r"partially guaranteed[^$]*\$([\d,]+)", clause)
              or re.search(r"\$([\d,]+)\s+guaranteed", clause))        # "2026/27: $250,000 guaranteed"
        if pg:
            guar[s] = money(pg.group(1))
    if "two-way" in low:
        ctype = "two_way"
    elif "exhibit 10" in low:
        ctype = "exhibit_10"
    elif re.search(r"\bwaived\b", low):
        ctype = "dead_money"
    elif "rookie-scale" in low:
        ctype = "rookie_scale"
    elif "minimum" in low:
        ctype = "minimum"
    else:
        ctype = "standard"
    unknown = [x.strip() for x in re.split(r"\.\s+", notes) if x.strip()
               and not any(k in x.lower() for k in KNOWN)]
    return opt, guar, ctype, unknown


def parse_team(abbr, html):
    tables = pd.read_html(io.StringIO(html))
    con = tables[0]
    seas_cols = [c for c in con.columns if re.fullmatch(r"\d{4}/\d{2}", str(c))]
    rows, unknown = [], []
    i = 0
    while i < len(con):
        r = con.iloc[i]
        name = str(r["Player"]).strip()
        is_note = name.split(":")[0].strip() and all(str(r[c]) == name for c in seas_cols)
        if is_note or name.lower().startswith("total") or name == "nan":
            i += 1
            continue
        notes = ""
        if i + 1 < len(con):
            nxt = str(con.iloc[i + 1]["Player"])
            if nxt.startswith(name + ":"):
                notes = nxt[len(name) + 1:].strip()
        seasons = [season(c) for c in seas_cols]
        opt, guar, ctype, unk = parse_notes(notes, seasons)
        unknown += unk
        for c in seas_cols:
            sal = money(r[c])
            s = season(c)
            if sal is None and ctype != "two_way":
                continue
            rows.append(dict(team=abbr, player_name=name, season=s, salary=sal or 0,
                             guaranteed=guar.get(s, sal or 0), contract_type=ctype, option=opt.get(s, ""),
                             notes=notes))
        i += 1
    holds = tables[1].copy() if len(tables) > 1 else pd.DataFrame()
    if not holds.empty:
        holds.insert(0, "team", abbr)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " | ", html))
    m = re.search(r"total salary[\s|]*\$\s*([\d,]+)", text, re.IGNORECASE)
    return pd.DataFrame(rows), holds, (money(m.group(1)) if m else None), unknown, parse_exceptions(abbr, text)


def parse_exceptions(abbr, text):
    """The team's 'Available exceptions' block: amount left, or None if 'Unavailable'."""
    start = text.find("Available exceptions")
    block = text[start:text.find("cap reference", start)] if start >= 0 else ""
    out = {"team": abbr}
    for col, label in [("mle_remaining", r"\|\s*Mid-level exception"),           # capital M = the team's MLE
                       ("room_mle_remaining", r"Room mid-level exception"),
                       ("bae_remaining", r"Bi-annual exception")]:              # lowercase = team block, not league table
        m = re.search(label + r"[\s|]*(\$[\d,]+|Unavailable)", block)
        out[col] = money(m.group(1)) if m and m.group(1).startswith("$") else None
    out["exceptions_found"] = bool(block)
    return out


def apply_overrides(df):
    path = os.path.join("reference", "contract_overrides.csv")
    if not os.path.exists(path):
        return df
    ov = pd.read_csv(path, dtype=str).fillna("")
    for o in ov.itertuples(index=False):
        mask = (df["team"] == o.team) & (df["player_name"] == o.player_name)
        if o.season:
            mask &= df["season"] == o.season
        df.loc[mask, o.field] = pd.to_numeric(o.value) if o.field in ("salary", "guaranteed") else o.value
    return df


def main():
    os.makedirs(RAW, exist_ok=True)
    robots = urllib.robotparser.RobotFileParser("https://www.shamsports.com/robots.txt")
    robots.read()
    cons, holds, totals, unknown, excs = [], [], [], [], []
    for abbr, slug in TEAMS.items():
        html, how = get_page(abbr, slug, robots)
        c, h, page_total, unk, exc = parse_team(abbr, html)
        excs.append(exc)
        cons.append(c)
        holds.append(h)
        unknown += unk
        totals.append(dict(team=abbr, page_total=page_total))
        print(f"  {abbr}: {c['player_name'].nunique():>2} players ({how})")

    df = apply_overrides(pd.concat(cons, ignore_index=True))
    df["source_url"] = df["team"].map(lambda t: BASE.format(slug=TEAMS[t]))
    df["scraped_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.to_parquet(os.path.join(OUT, "contracts_all.parquet"), index=False)
    pd.concat([h for h in holds if not h.empty], ignore_index=True).astype(str).to_parquet(
        os.path.join(OUT, "cap_holds_all.parquet"), index=False)
    ex = pd.DataFrame(excs)
    ex.to_parquet(os.path.join(OUT, "team_exceptions.parquet"), index=False)
    tot = pd.DataFrame(totals)
    tot.to_parquet(os.path.join(OUT, "page_totals.parquet"), index=False)

    # ---------- check: parsed 2026-27 total vs the page's own total ----------
    cur = df[(df["season"] == "2026-27") & ~df["contract_type"].isin(["two_way", "exhibit_10"])]
    chk = tot.merge(cur.groupby("team")["salary"].sum().rename("parsed_total").reset_index(), on="team")
    chk["diff"] = chk["parsed_total"] - chk["page_total"]
    ok = (chk["diff"].abs() <= 1).sum()
    print(f"\n{len(df):,} contract rows, {df['player_name'].nunique()} players across 30 teams")
    print(f"parsed 2026-27 total matches the page's total for {ok}/30 teams")
    if ok < 30:
        print(chk[chk["diff"].abs() > 1].to_string(index=False))
    print(f"\ncontract types: {df.drop_duplicates(['team', 'player_name'])['contract_type'].value_counts().to_dict()}")
    print(f"options found: {df['option'].replace('', None).dropna().value_counts().to_dict()}")
    print(f"exceptions block found for {ex['exceptions_found'].sum()}/30 teams; "
          f"MLE money left: {ex['mle_remaining'].notna().sum()} teams, BAE left: {ex['bae_remaining'].notna().sum()} teams")
    part = df[df["guaranteed"] < df["salary"]]
    print(f"not fully guaranteed: {len(part)} contract-seasons "
          f"(${(part['salary'] - part['guaranteed']).sum() / 1e6:,.1f}M could still come off the books)")
    if unknown:
        top = pd.Series(unknown).value_counts().head(15)
        print("\nnote phrases the parser doesn't use yet (check none matter):")
        print(top.to_string())


if __name__ == "__main__":
    main()