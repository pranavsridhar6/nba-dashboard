"""Step 4a: team ratings, garbage time, and the 76ers lineup simulator tables.

Outputs per season:
  data/league/team_ratings/   every team's offensive/defensive rating and defense rank
  data/lineups/phi_stints     76ers stints from the 76ers' point of view, tagged with
                              opponent defense tier and garbage time
  data/lineups/lineup_combos  every 2-, 3-, 4- and 5-man 76ers combination, summed by
                              season type x opponent tier x garbage time (Power BI filters these)
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import SEASONS, TEAM_ID        # noqa: E402
from utils import table_path               # noqa: E402

HOME = [f"home_p{i}" for i in range(1, 6)]
AWAY = [f"away_p{i}" for i in range(1, 6)]
PHI = [f"phi_p{i}" for i in range(1, 6)]
OPP = [f"opp_p{i}" for i in range(1, 6)]
SEASON_TYPE = {"002": "Regular Season", "004": "Playoffs"}


# ---------- team ratings (league-wide, regular season) ----------

def team_ratings(season):
    st = pd.read_parquet(table_path("league/stints", "stints.parquet", season))
    st = st[st["game_id"].str[:3] == "002"]
    home = pd.DataFrame({"team_id": st["home_team_id"], "pts": st["home_pts"], "poss": st["home_poss"],
                         "opp_pts": st["away_pts"], "opp_poss": st["away_poss"]})
    away = pd.DataFrame({"team_id": st["away_team_id"], "pts": st["away_pts"], "poss": st["away_poss"],
                         "opp_pts": st["home_pts"], "opp_poss": st["home_poss"]})
    t = pd.concat([home, away]).groupby("team_id").sum()
    t["ortg"] = 100 * t["pts"] / t["poss"]
    t["drtg"] = 100 * t["opp_pts"] / t["opp_poss"]
    t["net"] = t["ortg"] - t["drtg"]
    t["def_rank"] = t["drtg"].rank(method="min").astype(int)          # 1 = best defense
    t["def_tier"] = np.select([t["def_rank"] <= 10, t["def_rank"] >= 21],
                              ["Top 10 D", "Bottom 10 D"], "Middle 10 D")

    games = pd.read_parquet(table_path("league/games", "games.parquet", season))
    abbr = pd.concat([games[["home_team_id", "home_team"]].set_axis(["team_id", "team"], axis=1),
                      games[["away_team_id", "away_team"]].set_axis(["team_id", "team"], axis=1)]
                     ).drop_duplicates("team_id").set_index("team_id")["team"]
    t["team"] = t.index.map(abbr)
    t["season"] = season
    t = t.reset_index()[["season", "team_id", "team", "ortg", "drtg", "net", "def_rank", "def_tier"]]
    return t.round({"ortg": 1, "drtg": 1, "net": 1}).sort_values("def_rank")


# ---------- garbage time (Cleaning the Glass rule) ----------

def garbage_threshold(seconds_left):
    """Lead needed in the 4th quarter for garbage time, by time remaining."""
    return np.select([seconds_left > 540, seconds_left > 360], [25, 20], 10)


def tag_garbage(st, box):
    """4th quarter, lead past the threshold, and neither team has more than 2 starters on court."""
    starters = set(zip(box.loc[box["position"].fillna("").str.strip() != "", "game_id"],
                       box.loc[box["position"].fillna("").str.strip() != "", "player_id"]))

    def n_starters(row, cols):
        return sum((row["game_id"], row[c]) in starters for c in cols if pd.notna(row[c]))

    in_4th = st["period"] == 4
    seconds_left = 2880 - st["start_t"]
    big_lead = st["home_margin_start"].abs() >= garbage_threshold(seconds_left)
    candidates = in_4th & big_lead
    garbage = pd.Series(False, index=st.index)
    for i, row in st[candidates].iterrows():
        garbage.at[i] = n_starters(row, HOME) <= 2 and n_starters(row, AWAY) <= 2
    return garbage


# ---------- 76ers point of view ----------

def phi_view(st, ratings, names):
    is_home = st["home_team_id"] == TEAM_ID
    out = pd.DataFrame({
        "game_id": st["game_id"], "season": st["season"],
        "season_type": st["game_id"].str[:3].map(SEASON_TYPE),
        "period": st["period"], "start_row": st["start_row"], "end_row": st["end_row"],
        "seconds": st["seconds"], "garbage": st["garbage"],
        "opp_team_id": np.where(is_home, st["away_team_id"], st["home_team_id"]),
        "phi_pts": np.where(is_home, st["home_pts"], st["away_pts"]),
        "opp_pts": np.where(is_home, st["away_pts"], st["home_pts"]),
        "phi_poss": np.where(is_home, st["home_poss"], st["away_poss"]),
        "opp_poss": np.where(is_home, st["away_poss"], st["home_poss"]),
    })
    for i in range(5):
        out[PHI[i]] = np.where(is_home, st[HOME[i]], st[AWAY[i]])
        out[OPP[i]] = np.where(is_home, st[AWAY[i]], st[HOME[i]])
    out["phi_complete"] = out[PHI].notna().all(axis=1)

    r = ratings.set_index("team_id")
    out["opp_team"] = out["opp_team_id"].map(r["team"])
    out["opp_def_rank"] = out["opp_team_id"].map(r["def_rank"])
    out["opp_def_tier"] = out["opp_team_id"].map(r["def_tier"])
    out["phi_lineup"] = out[PHI].apply(
        lambda row: " / ".join(sorted(names.get(int(p), str(p)) for p in row if pd.notna(p))), axis=1)
    return out


def combos(phi, names):
    """Explode each complete stint into every 2-5 player subset, then sum."""
    phi = phi[phi["phi_complete"]]
    recs = []
    keys = ["season", "season_type", "opp_def_tier", "garbage"]
    for row in phi.itertuples(index=False):
        ids = sorted(int(getattr(row, c)) for c in PHI)
        base = {k: getattr(row, k) for k in keys}
        for size in range(2, 6):
            for combo in combinations(ids, size):
                recs.append({**base, "size": size, "combo_key": "-".join(map(str, combo)),
                             "seconds": row.seconds, "pts_for": row.phi_pts, "pts_against": row.opp_pts,
                             "poss_off": row.phi_poss, "poss_def": row.opp_poss})
    df = pd.DataFrame(recs)
    agg = df.groupby(keys + ["size", "combo_key"], as_index=False).sum()
    agg["players"] = agg["combo_key"].map(
        lambda k: " / ".join(sorted(names.get(int(p), p) for p in k.split("-"))))
    agg["minutes"] = (agg["seconds"] / 60).round(1)
    return agg.drop(columns="seconds")


def main():
    for season in SEASONS:
        ratings = team_ratings(season)
        ratings.to_parquet(table_path("league/team_ratings", "team_ratings.parquet", season), index=False)

        st = pd.read_parquet(table_path("stints", "stints.parquet", season))
        box = pd.read_parquet(table_path("player_box", "player_box.parquet", season))
        names = box.drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"].to_dict()

        st["garbage"] = tag_garbage(st, box)
        phi = phi_view(st, ratings, names)
        phi.to_parquet(table_path("lineups", "phi_stints.parquet", season), index=False)

        combo = combos(phi, names)
        combo.to_parquet(table_path("lineups", "lineup_combos.parquet", season), index=False)

        # Helper for Power BI: one row per player per lineup, so a measure can check
        # that EVERY picked player is in a lineup (slicers alone give "any of")
        cp = combo[["combo_key", "size"]].drop_duplicates()
        cp = cp.assign(player_id=cp["combo_key"].str.split("-")).explode("player_id")
        cp["player_id"] = cp["player_id"].astype(int)
        cp["player_name"] = cp["player_id"].map(names)
        cp["season"] = season
        cp.to_parquet(table_path("lineups", "lineup_combo_players.parquet", season), index=False)

        # ---------- summary ----------
        mins = phi["seconds"].sum() / 60
        g_mins = phi.loc[phi["garbage"], "seconds"].sum() / 60
        five = combo[combo["size"] == 5]
        print(f"\n{season}")
        print(f"  team ratings: {len(ratings)} teams. Best D: {ratings.iloc[0]['team']} "
              f"({ratings.iloc[0]['drtg']}), worst D: {ratings.iloc[-1]['team']} ({ratings.iloc[-1]['drtg']})")
        print(f"  76ers stints: {len(phi)}, {mins:,.0f} min, garbage time {g_mins:,.0f} min ({g_mins / mins:.1%})")
        print(f"  minutes by opponent tier: "
              f"{(phi.groupby('opp_def_tier')['seconds'].sum() / 60).round(0).to_dict()}")
        print(f"  combos saved: {combo['combo_key'].nunique():,} unique "
              f"({five['combo_key'].nunique()} five-man lineups)")
        # Check: 5-man minutes add up to all complete-lineup minutes
        complete_mins = phi.loc[phi["phi_complete"], "seconds"].sum() / 60
        tolerance = 0.05 * len(five) + 0.1               # minutes are rounded to 0.1 per row
        print(f"  check 5-man minutes = complete-lineup minutes: "
              f"{abs(five['minutes'].sum() - complete_mins) <= tolerance}")
        top = (five[~five["garbage"] & (five["season_type"] == "Regular Season")]
               .groupby("players")[["minutes", "pts_for", "pts_against", "poss_off", "poss_def"]].sum())
        top["net"] = 100 * (top["pts_for"] / top["poss_off"] - top["pts_against"] / top["poss_def"])
        print("  most-used 5-man lineups (regular season, no garbage time):")
        print(top.sort_values("minutes", ascending=False).head(5)[["minutes", "net"]].round(1).to_string())


if __name__ == "__main__":
    main()