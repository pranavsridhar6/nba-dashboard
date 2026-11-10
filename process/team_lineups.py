"""Step 9a: lineups, rim on/off and spacing for all 30 teams (generalizes lineups.py, on_off.py, spacing.py).

Every league stint is viewed from both teams' sides. Outputs per season (data/league/lineups/season=<s>/):
  team_stints.parquet           every stint from each team's point of view (opponent tier, garbage time)
  lineup_combos.parquet         every 2-5 man combination per team, summed by season type x tier x garbage
  lineup_combo_players.parquet  one row per player per combination (for "lineups that include all picks")
  on_off_rim.parquet            opponents' rim frequency / FG% / rating with each player on vs off
  spacing.parquet               teammates' rim rate / eFG% on vs off, plus tracking "tightly guarded" share
On/off rules (same as before): games the player played, regular season, no garbage time, full lineups.
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import SEASONS          # noqa: E402
from utils import table_path        # noqa: E402
from lineups import tag_garbage     # noqa: E402

RIM_FEET, MIN_POSS = 4, 1000
T = [f"t_p{i}" for i in range(1, 6)]
O = [f"o_p{i}" for i in range(1, 6)]
SEASON_TYPE = {"002": "Regular Season", "004": "Playoffs"}


def team_view(st, ratings):
    """Each stint twice: once from the home team's side, once from the away team's."""
    def side(us, them):
        v = pd.DataFrame({"stint_id": st.index, "game_id": st["game_id"].values, "period": st["period"].values,
                          "seconds": st["seconds"].values, "garbage": st["garbage"].values,
                          "team_id": st[f"{us}_team_id"].values, "opp_team_id": st[f"{them}_team_id"].values,
                          "pts_for": st[f"{us}_pts"].values, "pts_against": st[f"{them}_pts"].values,
                          "poss_off": st[f"{us}_poss"].values, "poss_def": st[f"{them}_poss"].values})
        for i in range(5):
            v[T[i]] = st[f"{us}_p{i + 1}"].values
            v[O[i]] = st[f"{them}_p{i + 1}"].values
        return v
    tv = pd.concat([side("home", "away"), side("away", "home")], ignore_index=True)
    tv["season_type"] = tv["game_id"].str[:3].map(SEASON_TYPE)
    tv["complete"] = tv[T].notna().all(axis=1)
    r = ratings.set_index("team_id")
    tv["team"], tv["opp_team"] = tv["team_id"].map(r["team"]), tv["opp_team_id"].map(r["team"])
    tv["opp_def_tier"] = tv["opp_team_id"].map(r["def_tier"])
    return tv


def lineup_combos(tv, names, season):
    c = tv[tv["complete"]].copy()
    ids = np.sort(c[T].to_numpy().astype(np.int64), axis=1)
    strs = [ids[:, i].astype(str) for i in range(5)]                 # numpy string arrays for np.char
    keys = ["team", "season_type", "opp_def_tier", "garbage"]
    base = c[keys + ["seconds", "pts_for", "pts_against", "poss_off", "poss_def"]].reset_index(drop=True)
    parts = []
    for size in range(2, 6):
        for combo in combinations(range(5), size):
            key = strs[combo[0]]
            for j in combo[1:]:
                key = np.char.add(np.char.add(key, "-"), strs[j])
            parts.append(base.assign(size=size, combo_key=key))
    df = pd.concat(parts, ignore_index=True)
    agg = df.groupby(keys + ["size", "combo_key"], as_index=False).sum()
    agg["minutes"] = (agg["seconds"] / 60).round(1)
    agg["season"] = season
    uniq = agg[["team", "combo_key", "size"]].drop_duplicates()
    label = {k: " / ".join(sorted(names.get(int(p), p) for p in k.split("-"))) for k in uniq["combo_key"]}
    agg["players"] = agg["combo_key"].map(label)
    cp = uniq.assign(player_id=uniq["combo_key"].str.split("-")).explode("player_id")
    cp["player_id"] = cp["player_id"].astype(int)
    cp["player_name"], cp["season"] = cp["player_id"].map(names), season
    return agg.drop(columns="seconds"), cp


def tag_shots(season, st):
    """Every FGA in the league, with the league stint it happened in (row-number matching)."""
    pbp = pd.read_parquet(table_path("league/pbp", "pbp.parquet", season),
                          columns=["game_id", "actionType", "teamId", "personId", "shotDistance", "shotValue"])
    pbp["row"] = pbp.groupby("game_id", sort=False).cumcount()
    s = pbp[pbp["actionType"].isin(["Made Shot", "Missed Shot"])].copy()
    for col in ("teamId", "personId", "shotDistance", "shotValue"):
        s[col] = pd.to_numeric(s[col], errors="coerce")
    s = s.dropna(subset=["teamId", "personId"])
    s["teamId"], s["personId"] = s["teamId"].astype(np.int64), s["personId"].astype(np.int64)   # match stint ids
    s["made"] = (s["actionType"] == "Made Shot").astype(int)
    s["rim"] = (s["shotDistance"] <= RIM_FEET).astype(int)
    s["three"] = (s["shotValue"] == 3).astype(int)
    s["stint_id"] = -1
    stg = st.sort_values(["game_id", "start_row"]).groupby("game_id")
    for gid, idx in s.groupby("game_id").groups.items():
        if gid not in stg.groups:
            continue
        g = st.loc[stg.groups[gid]].sort_values("start_row")
        rows = s.loc[idx, "row"].to_numpy()
        pos = np.searchsorted(g["start_row"].to_numpy(), rows, side="right") - 1
        ok = (pos >= 0) & (rows <= g["end_row"].to_numpy()[np.clip(pos, 0, None)])
        s.loc[idx, "stint_id"] = np.where(ok, g.index.to_numpy()[np.clip(pos, 0, None)], -1)
    s = s[s["stint_id"] >= 0]
    s["rim_made"] = s["made"] * s["rim"]
    s["three_made"] = s["made"] * s["three"]
    return s


def shot_sums(s, by):
    return s.groupby(by).agg(fga=("made", "size"), fgm=("made", "sum"), rim_fga=("rim", "sum"),
                             rim_fgm=("rim_made", "sum"), fg3a=("three", "sum"), fg3m=("three_made", "sum"))


def on_off_and_spacing(tv, shots, box, tracking, season):
    use = tv[(tv["season_type"] == "Regular Season") & ~tv["garbage"] & tv["complete"]].copy()
    by_team = shot_sums(shots, ["stint_id", "teamId"])
    by_player = shot_sums(shots, ["stint_id", "personId"])
    stats = ["fga", "fgm", "rim_fga", "rim_fgm", "fg3a", "fg3m"]
    opp = by_team.rename_axis(["stint_id", "opp_team_id"]).add_prefix("opp_")
    own = by_team.rename_axis(["stint_id", "team_id"]).add_prefix("own_")
    use = use.join(opp, on=["stint_id", "opp_team_id"]).join(own, on=["stint_id", "team_id"]).fillna(
        {f"{p}_{s}": 0 for p in ("opp", "own") for s in stats})

    played = box[box["did_play"] & (box["game_id"].str[:3] == "002")]
    names = played.drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"]
    rows = []
    for (team, pid), games in played.groupby(["team", "player_id"])["game_id"]:
        g = use[(use["team"] == team) & use["game_id"].isin(set(games))]
        if g.empty:
            continue
        on = g[T].eq(pid).any(axis=1).to_numpy()
        mine = by_player.loc[by_player.index.get_level_values(1) == pid]
        mine = mine.droplevel(1).reindex(g.loc[on, "stint_id"]).fillna(0).sum()
        rec = {"season": season, "team": team, "player_id": int(pid), "player_name": names.get(pid),
               "opp_poss_on": round(g.loc[on, "poss_def"].sum()), "poss_on": round(g.loc[on, "poss_off"].sum())}
        for side, part, minus in (("on", g[on], mine), ("off", g[~on], None)):
            o = part[[f"opp_{s}" for s in stats]].sum()
            w = part[[f"own_{s}" for s in stats]].sum()
            if minus is not None:                                   # teammates only: remove his own shots
                w = w - minus[stats].to_numpy()
            dposs = part["poss_def"].sum()
            rec[f"rim_freq_{side}"] = o["opp_rim_fga"] / o["opp_fga"] if o["opp_fga"] else np.nan
            rec[f"rim_fg_{side}"] = o["opp_rim_fgm"] / o["opp_rim_fga"] if o["opp_rim_fga"] else np.nan
            rec[f"drtg_{side}"] = 100 * part["pts_against"].sum() / dposs if dposs else np.nan
            rec[f"tm_rim_freq_{side}"] = w["own_rim_fga"] / w["own_fga"] if w["own_fga"] else np.nan
            rec[f"tm_efg_{side}"] = (w["own_fgm"] + 0.5 * w["own_fg3m"]) / w["own_fga"] if w["own_fga"] else np.nan
            rec[f"tm_3pa_rate_{side}"] = w["own_fg3a"] / w["own_fga"] if w["own_fga"] else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    for mtr in ("rim_freq", "rim_fg", "drtg", "tm_rim_freq", "tm_efg", "tm_3pa_rate"):
        out[f"{mtr}_diff"] = out[f"{mtr}_on"] - out[f"{mtr}_off"]
    out = out.merge(tracking, on="player_id", how="left")
    for col in ("jumpers_10ft", "tight_share", "open_share", "wide_open_share"):   # present even without tracking
        if col not in out.columns:
            out[col] = np.nan
    out["reliable"] = out["poss_on"] >= MIN_POSS
    pct = [c for c in out.columns if c.startswith(("rim_freq", "rim_fg", "tm_", "tight_", "open_", "wide_"))]
    out[pct] = out[pct] * 100
    return out.round(1)


def tracking_shares(season):
    p = table_path("league/tracking", "closest_defender.parquet", season)
    if not os.path.exists(p):
        return pd.DataFrame(columns=["player_id"])
    t = pd.read_parquet(p).dropna(subset=["CLOSE_DEF_DIST_RANGE"])
    t["bucket"] = t["CLOSE_DEF_DIST_RANGE"].str.extract(r"(Very Tight|Tight|Wide Open|Open)")[0]
    pv = t.pivot_table(index="player_id", columns="bucket", values="FGA", aggfunc="sum").fillna(0)
    tot = pv.sum(axis=1)
    out = pd.DataFrame({"jumpers_10ft": tot,
                        "tight_share": (pv.get("Very Tight", 0) + pv.get("Tight", 0)) / tot,
                        "open_share": (pv.get("Open", 0) + pv.get("Wide Open", 0)) / tot,
                        "wide_open_share": pv.get("Wide Open", 0) / tot}).reset_index()
    return out


def main():
    for season in SEASONS:
        st = pd.read_parquet(table_path("league/stints", "stints.parquet", season))
        box = pd.read_parquet(table_path("league/player_box", "player_box.parquet", season))
        ratings = pd.read_parquet(table_path("league/team_ratings", "team_ratings.parquet", season))
        names = box.drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"].to_dict()
        st["garbage"] = tag_garbage(st, box)

        tv = team_view(st, ratings)
        tv.to_parquet(table_path("league/lineups", "team_stints.parquet", season), index=False)
        combos, cp = lineup_combos(tv, names, season)
        combos.to_parquet(table_path("league/lineups", "lineup_combos.parquet", season), index=False)
        cp.to_parquet(table_path("league/lineups", "lineup_combo_players.parquet", season), index=False)

        shots = tag_shots(season, st)
        oo = on_off_and_spacing(tv, shots, box, tracking_shares(season), season)
        onoff_cols = ["season", "team", "player_id", "player_name", "opp_poss_on"] + \
                     [c for c in oo.columns if c.startswith(("rim_freq", "rim_fg", "drtg"))] + ["reliable"]
        space_cols = ["season", "team", "player_id", "player_name", "poss_on"] + \
                     [c for c in oo.columns if c.startswith(("tm_", "tight_", "open_", "wide_", "jumpers"))] + ["reliable"]
        oo[onoff_cols].to_parquet(table_path("league/lineups", "on_off_rim.parquet", season), index=False)
        oo[space_cols].to_parquet(table_path("league/lineups", "spacing.parquet", season), index=False)

        five = combos[combos["size"] == 5]
        print(f"{season}: {len(tv):,} team-stints | {combos['combo_key'].nunique():,} lineup combos "
              f"({five['combo_key'].nunique():,} five-man) | {len(shots):,} shots tagged | "
              f"{oo['reliable'].sum()} reliable on/off players across {oo['team'].nunique()} teams")
        p = oo[(oo["team"] == "PHI") & oo["reliable"]].sort_values("tm_efg_diff", ascending=False)
        if len(p):
            print("   check PHI (compare with the 76ers-only spacing table):")
            print(p[["player_name", "poss_on", "tm_rim_freq_diff", "tm_efg_diff", "rim_freq_diff", "tight_share"]]
                  .head(4).to_string(index=False))


if __name__ == "__main__":
    main()