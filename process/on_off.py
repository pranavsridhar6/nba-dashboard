"""Step 4b: on/off rim defense for 76ers players (public proxy for 'rim deterrence').

For every opponent field goal attempt, find the stint it happened in (by event row),
so we know which 76ers were on the floor. Then, per player, compare opponents with him
ON the floor vs OFF the floor (only in games he played; regular season; no garbage time):
  rim_freq   share of opponent shots taken within 4 feet       (lower = more deterrence)
  rim_fg     opponent FG% on those shots                       (lower = better rim protection)
  rim_per100 opponent rim attempts per 100 opponent possessions
  drtg       opponent points per 100 possessions
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import SEASONS, TEAM_ID        # noqa: E402
from utils import table_path               # noqa: E402

RIM_FEET = 4
MIN_ON_POSS = 1000        # opponent possessions with the player on court before the split is meaningful
PHI = [f"phi_p{i}" for i in range(1, 6)]


def opponent_shots(season, stints):
    """Every opponent FGA in 76ers games, tagged with the stint it belongs to."""
    pbp = pd.read_parquet(table_path("pbp", "pbp.parquet", season),
                          columns=["game_id", "actionType", "teamId", "shotDistance"])
    pbp["row"] = pbp.groupby("game_id", sort=False).cumcount()   # same row numbers build_stints used
    pbp["teamId"] = pd.to_numeric(pbp["teamId"], errors="coerce")
    shots = pbp[pbp["actionType"].isin(["Made Shot", "Missed Shot"])
                & pbp["teamId"].notna() & (pbp["teamId"] != TEAM_ID)].copy()
    shots["made"] = shots["actionType"] == "Made Shot"
    shots["dist"] = pd.to_numeric(shots["shotDistance"], errors="coerce")
    shots["rim"] = shots["dist"] <= RIM_FEET

    # Match each shot to the stint whose row range contains it
    shots["stint_idx"] = -1
    for gid, idx in shots.groupby("game_id").groups.items():
        s = stints[stints["game_id"] == gid].sort_values("start_row")
        if s.empty:
            continue
        rows = shots.loc[idx, "row"].to_numpy()
        pos = np.searchsorted(s["start_row"].to_numpy(), rows, side="right") - 1
        ok = (pos >= 0) & (rows <= s["end_row"].to_numpy()[np.clip(pos, 0, None)])
        shots.loc[idx, "stint_idx"] = np.where(ok, s.index.to_numpy()[np.clip(pos, 0, None)], -1)
    return shots


def main():
    for season in SEASONS:
        phi = pd.read_parquet(table_path("lineups", "phi_stints.parquet", season))
        box = pd.read_parquet(table_path("player_box", "player_box.parquet", season))
        shots = opponent_shots(season, phi)
        matched = (shots["stint_idx"] >= 0).mean()

        # Attach shot counts to each stint
        sh = shots[shots["stint_idx"] >= 0]
        per_stint = sh.groupby("stint_idx").agg(fga=("made", "size"),
                                                rim_fga=("rim", "sum"),
                                                rim_fgm=("made", lambda m: (m & sh.loc[m.index, "rim"]).sum()))
        st = phi.join(per_stint).fillna({"fga": 0, "rim_fga": 0, "rim_fgm": 0})
        st = st[(st["season_type"] == "Regular Season") & ~st["garbage"] & st["phi_complete"]]

        phi_players = box[(box["team_id"] == TEAM_ID) & box["did_play"]]
        names = phi_players.drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"]
        rows = []
        for pid, games in phi_players.groupby("player_id")["game_id"]:
            in_games = st[st["game_id"].isin(set(games))]
            on_mask = in_games[PHI].eq(pid).any(axis=1)
            rec = {"season": season, "player_id": int(pid), "player_name": names[pid]}
            for side, part in (("on", in_games[on_mask]), ("off", in_games[~on_mask])):
                fga, rim_fga, rim_fgm = part["fga"].sum(), part["rim_fga"].sum(), part["rim_fgm"].sum()
                poss = part["opp_poss"].sum()
                rec[f"opp_poss_{side}"] = round(poss)
                rec[f"rim_freq_{side}"] = rim_fga / fga if fga else np.nan
                rec[f"rim_fg_{side}"] = rim_fgm / rim_fga if rim_fga else np.nan
                rec[f"rim_per100_{side}"] = 100 * rim_fga / poss if poss else np.nan
                rec[f"drtg_{side}"] = 100 * part["opp_pts"].sum() / poss if poss else np.nan
            rows.append(rec)

        out = pd.DataFrame(rows)
        for m in ("rim_freq", "rim_fg", "rim_per100", "drtg"):
            out[f"{m}_diff"] = out[f"{m}_on"] - out[f"{m}_off"]      # negative = better with him on
        out["reliable"] = out["opp_poss_on"] >= MIN_ON_POSS
        pct = [c for c in out.columns if c.startswith(("rim_freq", "rim_fg"))]
        out[pct] = out[pct] * 100                                    # store as percentages
        out = out.round(1).sort_values("rim_freq_diff")
        out.to_parquet(table_path("lineups", "on_off_rim.parquet", season), index=False)

        print(f"\n{season}: {len(shots):,} opponent shots, {matched:.1%} matched to a stint")
        cols = ["player_name", "opp_poss_on", "rim_freq_on", "rim_freq_off", "rim_freq_diff",
                "rim_fg_on", "rim_fg_off", "rim_fg_diff", "drtg_diff"]
        print(out[out["reliable"]][cols].to_string(index=False))


if __name__ == "__main__":
    main()