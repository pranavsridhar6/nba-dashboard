"""Step 7b: spacing / gravity proxy for 76ers players.

Two public proxies for "spacing distortion" (true defender positioning is proprietary):
  1. How tightly he is guarded on jump shots 10+ ft (tracking data): a shooter defenses
     respect gets fewer open looks. Lower open share = more gravity.
  2. What happens to his TEAMMATES' shots when he is on the floor vs off (play-by-play + stints):
     more rim attempts and better eFG% with him on = he opens up the floor.
Only games he played, regular season, no garbage time, full lineups.

Output: data/lineups/spacing.parquet (all seasons in one file per season folder)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import SEASONS, TEAM_ID        # noqa: E402
from utils import table_path               # noqa: E402

RIM_FEET = 4
MIN_ON_POSS = 1000
PHI = [f"phi_p{i}" for i in range(1, 6)]


def team_shots(season, stints):
    """Every 76ers FGA, tagged with its stint (same row-matching as on_off.py)."""
    pbp = pd.read_parquet(table_path("pbp", "pbp.parquet", season),
                          columns=["game_id", "actionType", "teamId", "personId", "shotDistance", "shotValue"])
    pbp["row"] = pbp.groupby("game_id", sort=False).cumcount()
    for c in ("teamId", "personId", "shotDistance", "shotValue"):
        pbp[c] = pd.to_numeric(pbp[c], errors="coerce")
    s = pbp[pbp["actionType"].isin(["Made Shot", "Missed Shot"]) & (pbp["teamId"] == TEAM_ID)].copy()
    s["made"] = (s["actionType"] == "Made Shot").astype(int)
    s["rim"] = (s["shotDistance"] <= RIM_FEET).astype(int)
    s["three"] = (s["shotValue"] == 3).astype(int)
    s["stint_idx"] = -1
    for gid, idx in s.groupby("game_id").groups.items():
        st = stints[stints["game_id"] == gid].sort_values("start_row")
        if st.empty:
            continue
        rows = s.loc[idx, "row"].to_numpy()
        pos = np.searchsorted(st["start_row"].to_numpy(), rows, side="right") - 1
        ok = (pos >= 0) & (rows <= st["end_row"].to_numpy()[np.clip(pos, 0, None)])
        s.loc[idx, "stint_idx"] = np.where(ok, st.index.to_numpy()[np.clip(pos, 0, None)], -1)
    return s[s["stint_idx"] >= 0]


def shooting(shots):
    fga = len(shots)
    if fga == 0:
        return np.nan, np.nan, np.nan
    fgm, fg3m = shots["made"].sum(), (shots["made"] & shots["three"]).sum()
    return shots["rim"].mean(), (fgm + 0.5 * fg3m) / fga, shots["three"].mean()


def tracking(season):
    path = table_path("tracking", "closest_defender.parquet", season)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["player_id"])
    t = pd.read_parquet(path)
    t["bucket"] = t["CLOSE_DEF_DIST_RANGE"].str.extract(r"(Very Tight|Tight|Wide Open|Open)")[0]
    pv = t.pivot_table(index="player_id", columns="bucket", values="FGA", aggfunc="sum").fillna(0)
    total = pv.sum(axis=1)
    out = pd.DataFrame(index=pv.index)
    out["jumpers_10ft"] = total
    out["tight_share"] = (pv.get("Very Tight", 0) + pv.get("Tight", 0)) / total
    out["open_share"] = (pv.get("Open", 0) + pv.get("Wide Open", 0)) / total
    out["wide_open_share"] = pv.get("Wide Open", 0) / total
    return out.reset_index()


def main():
    for season in SEASONS:
        phi = pd.read_parquet(table_path("lineups", "phi_stints.parquet", season))
        box = pd.read_parquet(table_path("player_box", "player_box.parquet", season))
        shots = team_shots(season, phi)
        st = phi[(phi["season_type"] == "Regular Season") & ~phi["garbage"] & phi["phi_complete"]]
        shots = shots[shots["stint_idx"].isin(st.index)]

        players = box[(box["team_id"] == TEAM_ID) & box["did_play"]]
        names = players.drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"]
        rows = []
        for pid, games in players.groupby("player_id")["game_id"]:
            in_games = st[st["game_id"].isin(set(games))]
            on_mask = in_games[PHI].eq(pid).any(axis=1)
            on_ids, off_ids = set(in_games.index[on_mask]), set(in_games.index[~on_mask])
            on_shots = shots[shots["stint_idx"].isin(on_ids) & (shots["personId"] != pid)]   # teammates only
            off_shots = shots[shots["stint_idx"].isin(off_ids)]
            rec = {"season": season, "player_id": int(pid), "player_name": names[pid],
                   "poss_on": round(in_games.loc[on_mask, "phi_poss"].sum())}
            for side, part in (("on", on_shots), ("off", off_shots)):
                rim, efg, three = shooting(part)
                rec[f"tm_rim_freq_{side}"], rec[f"tm_efg_{side}"], rec[f"tm_3pa_rate_{side}"] = rim, efg, three
            rows.append(rec)

        out = pd.DataFrame(rows)
        for m in ("tm_rim_freq", "tm_efg", "tm_3pa_rate"):
            out[f"{m}_diff"] = out[f"{m}_on"] - out[f"{m}_off"]      # positive = teammates better with him on
        out = out.merge(tracking(season), on="player_id", how="left")
        out["reliable"] = out["poss_on"] >= MIN_ON_POSS
        pct = [c for c in out.columns if c.startswith(("tm_", "tight_", "open_", "wide_"))]
        out[pct] = out[pct] * 100
        out = out.round(1).sort_values("tm_efg_diff", ascending=False)
        out.to_parquet(table_path("lineups", "spacing.parquet", season), index=False)

        rel = out[out["reliable"]]
        print(f"\n{season}: {len(shots):,} 76ers shots tagged, {len(rel)} reliable players")
        cols = ["player_name", "poss_on", "tm_rim_freq_diff", "tm_efg_diff", "open_share", "tight_share"]
        print(rel[cols].to_string(index=False))


if __name__ == "__main__":
    main()