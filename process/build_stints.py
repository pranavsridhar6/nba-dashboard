"""Step 2b: turn play-by-play into stints (stretches with the same 10 players on court).

One row per stint: who was on the floor, how long, points and possession estimates per side.
Built in home/away form so the same code works for league-wide games later."""
import os
import re
import sys
import unicodedata
from difflib import get_close_matches
from collections import defaultdict

import pandas as pd

# config.py and utils.py live in ingest/, so make them importable from here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import SEASONS          # noqa: E402
from utils import table_path        # noqa: E402

SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)$")     # "SUB: Crowder FOR G. Antetokounmpo"
CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")            # "PT11M49.00S"
FT_RE = re.compile(r"(\d+) of (\d+)")                  # "Free Throw 1 of 2"


# ---------- small helpers ----------

def norm(name):
    """'G. Antetokounmpo' -> 'gantetokounmpo'. Strips accents, dots, spaces, case."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\.?$", re.IGNORECASE)


def base_last(last):
    """'Porter Jr.' -> 'Porter', 'Butler III' -> 'Butler'."""
    return SUFFIX_RE.sub("", str(last or "").strip())


def clock_seconds(clock):
    """Seconds left in the period."""
    m = CLOCK_RE.match(str(clock))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else 0.0


def period_length(p):
    return 720 if p <= 4 else 300                       # 12 min quarters, 5 min overtimes


def period_offset(p):
    """Game seconds elapsed before period p starts."""
    return 720 * min(p - 1, 4) + 300 * max(p - 5, 0)


def build_name_lookup(roster, g):
    """(team_id, normalized name) -> set of player_ids. Used to identify the player coming IN,
    who only appears as a name in the substitution description."""
    keys = defaultdict(set)
    for r in roster.itertuples(index=False):
        first, last = str(r.first_name or ""), str(r.last_name or "")
        for ln in {last, base_last(last)}:
            keys[(r.team_id, norm(ln))].add(r.player_id)
            keys[(r.team_id, norm(first + ln))].add(r.player_id)
            if first:
                keys[(r.team_id, norm(first[0] + ln))].add(r.player_id)
    team_of = dict(zip(roster["player_id"], roster["team_id"]))
    for r in g[["personId", "playerName", "playerNameI"]].drop_duplicates().itertuples(index=False):
        t = team_of.get(r.personId)
        if t is None:
            continue
        for name in (r.playerName, r.playerNameI):
            if isinstance(name, str) and name.strip():
                keys[(t, norm(name))].add(r.personId)
    return keys


def infer_period_starters(rows, team_of, resolve):
    """A player started the period if his first appearance in it is anything other than
    being subbed IN (a shot, foul, rebound, or being subbed OUT)."""
    status = {}
    for r in rows:
        if r.actionType == "Substitution":
            out_id = r.personId
            team = team_of.get(out_id)
            m = SUB_RE.match(str(r.description or ""))
            in_id = resolve(team, m.group(1)) if (m and team) else None
            if out_id in team_of and out_id not in status:
                status[out_id] = "starter"
            if in_id is not None and in_id not in status:
                status[in_id] = "bench"
        elif r.personId in team_of and r.personId not in status:
            status[r.personId] = "starter"
    starters = defaultdict(set)
    for p, s in status.items():
        if s == "starter":
            starters[team_of[p]].add(p)
    return starters, status


def find_home_team(g, team_of, teams):
    """The team whose made shot moves scoreHome is the home team."""
    prev = 0
    for r in g.itertuples(index=False):
        if r.sh != prev and r.actionType in ("Made Shot", "Free Throw"):
            t = team_of.get(r.personId)
            if t is not None:
                return t
        prev = r.sh
    raise ValueError("could not determine home team")


# ---------- one game ----------

def build_game(gid, g, roster, season):
    roster = roster.copy()
    roster["player_id"] = roster["player_id"].astype(int)
    roster["team_id"] = roster["team_id"].astype(int)
    team_of = dict(zip(roster["player_id"], roster["team_id"]))
    teams = sorted(set(team_of.values()))
    keys = build_name_lookup(roster, g)

    by_team = defaultdict(list)                     # team -> [(player_id, first, last)] normalized
    for r in roster.itertuples(index=False):
        by_team[r.team_id].append((r.player_id, norm(r.first_name or ""), norm(base_last(r.last_name))))

    def unique(ids):
        ids = set(ids)
        return next(iter(ids)) if len(ids) == 1 else None

    def resolve(team, name):
        """Match a substitution name to a player id, strictest rule first."""
        # a) exact: last name, initial + last, full name (with and without Jr./III)
        found = unique(keys.get((team, norm(name)), set()))
        if found is not None:
            return found
        players = by_team.get(team, [])
        # b) "Ja. Green" / "Jay. Williams": first-name prefix + last name
        m = re.match(r"^\s*([A-Za-z]+)\.\s+(.+)$", str(name))
        if m:
            prefix, last = norm(m.group(1)), norm(base_last(m.group(2)))
            found = unique(p for p, f, l in players if l == last and f.startswith(prefix))
            if found is not None:
                return found
        target = norm(base_last(name))
        # c) name order differs (e.g. listed by first name): match on first name
        found = unique(p for p, f, l in players if f == target)
        if found is not None:
            return found
        # d) spelling differs ("Pöltl" vs "Poeltl"): closest last name, if clearly close
        lasts = [l for _, _, l in players]
        close = get_close_matches(target, lasts, n=2, cutoff=0.8)
        if len(close) == 1:
            return unique(p for p, f, l in players if l == close[0])
        return None

    home = find_home_team(g, team_of, teams)
    away = [t for t in teams if t != home][0]
    side = {home: "home", away: "away"}

    # Period 1 starters straight from the box score (position is filled in only for starters)
    box_starters = defaultdict(set)
    for r in roster.itertuples(index=False):
        if isinstance(r.position, str) and r.position.strip():
            box_starters[r.team_id].add(r.player_id)

    stints, period_log, issues = [], [], []
    prev_end = None

    for period, prows in g.groupby("period", sort=True):
        rows = list(prows.itertuples(index=False))

        # --- who starts this period ---
        if period == 1 and all(len(box_starters[t]) == 5 for t in (home, away)):
            on = {t: set(box_starters[t]) for t in (home, away)}
            for t in (home, away):
                period_log.append(dict(game_id=gid, period=period, team_id=t, n=5, method="box"))
        else:
            inferred, status = infer_period_starters(rows, team_of, resolve)
            on = {}
            for t in (home, away):
                s, method = set(inferred.get(t, set())), "inferred"
                if len(s) < 5 and prev_end is not None:
                    # Players on court at the end of last period who did nothing all period
                    quiet = [p for p in prev_end[t] if p not in status]
                    if len(s) + len(quiet) == 5:
                        s |= set(quiet)
                        method = "filled"
                on[t] = s
                period_log.append(dict(game_id=gid, period=period, team_id=t, n=len(s), method=method))

        # --- walk the period ---
        def new_stint(start_t, sh, sa):
            return dict(lineup={t: frozenset(on[t]) for t in (home, away)},
                        start_t=start_t, start_sh=sh, start_sa=sa,
                        **{f"{s}_{k}": 0 for s in ("home", "away") for k in ("fga", "fta", "tov", "oreb")})

        def close(st, end_t, end_sh, end_sa):
            seconds = end_t - st["start_t"]
            pts_h, pts_a = end_sh - st["start_sh"], end_sa - st["start_sa"]
            counts = sum(st[f"{s}_{k}"] for s in ("home", "away") for k in ("fga", "fta", "tov"))
            if seconds == 0 and pts_h == 0 and pts_a == 0 and counts == 0:
                return                                   # empty stint from stacked subs, skip
            rec = dict(game_id=gid, season=season, period=period,
                       home_team_id=home, away_team_id=away,
                       start_t=st["start_t"], end_t=end_t, seconds=seconds,
                       home_margin_start=st["start_sh"] - st["start_sa"],
                       home_pts=pts_h, away_pts=pts_a)
            for s, t in (("home", home), ("away", away)):
                ids = sorted(st["lineup"][t])
                for i in range(5):
                    rec[f"{s}_p{i + 1}"] = ids[i] if i < len(ids) else None
                rec[f"{s}_n_players"] = len(ids)
                for k in ("fga", "fta", "tov", "oreb"):
                    rec[f"{s}_{k}"] = st[f"{s}_{k}"]
                # Possessions estimate (offense): FGA - OREB + TOV + 0.44 * FTA
                rec[f"{s}_poss"] = round(st[f"{s}_fga"] - st[f"{s}_oreb"]
                                         + st[f"{s}_tov"] + 0.44 * st[f"{s}_fta"], 2)
            rec["lineup_complete"] = rec["home_n_players"] == 5 and rec["away_n_players"] == 5
            stints.append(rec)

        prev_sh, prev_sa = rows[0].sh, rows[0].sa
        cur = new_stint(period_offset(period), prev_sh, prev_sa)
        last_sub_t = None
        last_miss = None                                  # (team, live_ball) of the last miss

        for r in rows:
            if r.actionType == "Substitution":
                out_id = r.personId
                team = team_of.get(out_id)
                m = SUB_RE.match(str(r.description or ""))
                if team is None or not m:
                    issues.append(dict(game_id=gid, period=period, problem="unreadable sub",
                                       detail=r.description))
                else:
                    on[team].discard(out_id)
                    in_id = resolve(team, m.group(1))
                    if in_id is None:
                        issues.append(dict(game_id=gid, period=period, problem="unmatched name",
                                           detail=m.group(1)))
                    else:
                        on[team].add(in_id)
                last_sub_t = r.t
                prev_sh, prev_sa = r.sh, r.sa
                continue

            # A non-sub event: if subs changed the lineup, close the old stint first
            if any(frozenset(on[t]) != cur["lineup"][t] for t in (home, away)):
                close(cur, last_sub_t, prev_sh, prev_sa)
                cur = new_stint(last_sub_t, prev_sh, prev_sa)

            # Which team does this event belong to?
            team = team_of.get(r.personId)
            if team is None and r.personId in side:
                team = r.personId                         # team-level rows use the team id
            if team is None and r.teamId in side:
                team = r.teamId

            at = r.actionType
            if team is not None:
                s = side[team]
                if at in ("Made Shot", "Missed Shot"):
                    cur[f"{s}_fga"] += 1
                    last_miss = (team, True) if at == "Missed Shot" else None
                elif at == "Free Throw":
                    cur[f"{s}_fta"] += 1
                    if "MISS" in str(r.description).upper():
                        m = FT_RE.search(str(r.subType))
                        last_miss = (team, bool(m) and m.group(1) == m.group(2))
                    else:
                        last_miss = None
                elif at == "Rebound":
                    if last_miss and last_miss[1] and team == last_miss[0]:
                        cur[f"{s}_oreb"] += 1
                    last_miss = None
                elif at == "Turnover":
                    cur[f"{s}_tov"] += 1

            prev_sh, prev_sa = r.sh, r.sa

        close(cur, period_offset(period) + period_length(period), prev_sh, prev_sa)
        prev_end = {t: set(on[t]) for t in (home, away)}

    return stints, period_log, issues


# ---------- all games ----------

def prepare_pbp(pbp):
    pbp = pbp.copy()
    for c in ("personId", "teamId"):
        pbp[c] = pd.to_numeric(pbp[c], errors="coerce").fillna(0).astype(int)
    pbp["actionType"] = pbp["actionType"].fillna("").astype(str)
    pbp["clock_s"] = pbp["clock"].map(clock_seconds)
    pbp["t"] = [period_offset(p) + period_length(p) - c for p, c in zip(pbp["period"], pbp["clock_s"])]
    return pbp


def main():
    for season in SEASONS:
        pbp = prepare_pbp(pd.read_parquet(table_path("pbp", "pbp.parquet", season)))
        box = pd.read_parquet(table_path("player_box", "player_box.parquet", season))

        all_stints, all_periods, all_issues, checks = [], [], [], []
        for gid, g in pbp.groupby("game_id", sort=False):
            g = g.reset_index(drop=True)                      # keep the API's event order
            g["sh"] = pd.to_numeric(g["scoreHome"], errors="coerce").ffill().fillna(0).astype(int)
            g["sa"] = pd.to_numeric(g["scoreAway"], errors="coerce").ffill().fillna(0).astype(int)
            roster = box[box["game_id"] == gid]
            try:
                s, p, i = build_game(gid, g, roster, season)
            except Exception as e:
                all_issues.append(dict(game_id=gid, period=None, problem="game failed", detail=str(e)))
                continue
            all_stints += s
            all_periods += p
            all_issues += i

            # Per-game check: stint points and seconds add up to the final score and game length
            sdf = pd.DataFrame(s)
            n_periods = int(g["period"].max())
            checks.append(dict(
                game_id=gid,
                pts_ok=sdf["home_pts"].sum() == g["sh"].iloc[-1] and sdf["away_pts"].sum() == g["sa"].iloc[-1],
                secs_ok=abs(sdf["seconds"].sum() - (period_offset(n_periods) + period_length(n_periods))) < 1,
            ))

        stints = pd.DataFrame(all_stints)
        out = table_path("stints", "stints.parquet", season)
        stints.to_parquet(out, index=False)
        pd.DataFrame(all_periods).to_parquet(table_path("stints", "period_starters.parquet", season), index=False)
        pd.DataFrame(all_issues, columns=["game_id", "period", "problem", "detail"]).to_parquet(
            table_path("stints", "issues.parquet", season), index=False)

        # ---------- summary ----------
        chk = pd.DataFrame(checks)
        per = pd.DataFrame(all_periods)
        iss = pd.DataFrame(all_issues, columns=["game_id", "period", "problem", "detail"])
        print(f"\n{season}: {chk.shape[0]} games -> {len(stints)} stints, saved to {out}")
        print(f"  games where stint points = final score:   {chk['pts_ok'].sum()}/{len(chk)}")
        print(f"  games where stint seconds = game length:  {chk['secs_ok'].sum()}/{len(chk)}")
        print(f"  stints with a full 5-on-5 lineup:          {stints['lineup_complete'].mean():.1%}")
        print(f"  period starters found by method:\n{per.groupby('method')['n'].agg(['count', lambda x: (x == 5).mean()]).rename(columns={'<lambda_0>': 'share_with_5'}).to_string()}")
        if not iss.empty:
            print(f"  issues: {iss['problem'].value_counts().to_dict()}")
            print(iss.head(10).to_string())


if __name__ == "__main__":
    main()