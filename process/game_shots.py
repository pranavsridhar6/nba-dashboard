"""Both teams' shots in every 76ers game, built from play-by-play (no extra API calls).

Play-by-play has every field goal attempt with court coordinates (xLegacy / yLegacy, tenths
of feet, hoop at 0,0), for both teams. Column names match the Shots table, so the Deneb
court works unchanged. Validated against the official 76ers shot chart data.

Output: data/game_shots/season=<season>/game_shots.parquet
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from config import SEASONS, TEAM_ID        # noqa: E402
from utils import table_path               # noqa: E402

COLS = ["game_id", "actionNumber", "period", "clock", "teamId", "teamTricode", "personId",
        "xLegacy", "yLegacy", "shotDistance", "actionType", "subType", "shotValue"]


def main():
    for season in SEASONS:
        pbp = pd.read_parquet(table_path("pbp", "pbp.parquet", season), columns=COLS)
        box = pd.read_parquet(table_path("player_box", "player_box.parquet", season),
                              columns=["game_id", "player_id", "player_name", "team", "fga"])

        s = pbp[pbp["actionType"].isin(["Made Shot", "Missed Shot"])].copy()
        for c in ("teamId", "personId", "xLegacy", "yLegacy", "shotDistance", "shotValue", "actionNumber"):
            s[c] = pd.to_numeric(s[c], errors="coerce")
        names = box.drop_duplicates("player_id").set_index("player_id")["player_name"]

        out = pd.DataFrame({
            "game_id": s["game_id"],
            "GAME_EVENT_ID": s["actionNumber"].astype("Int64"),
            "season": season,
            "period": s["period"],
            "clock": s["clock"],
            "team_id": s["teamId"].astype("Int64"),
            "team": s["teamTricode"],
            "side": np.where(s["teamId"] == TEAM_ID, "76ers", "Opponent"),
            "PLAYER_ID": s["personId"].astype("Int64"),
            "PLAYER_NAME": s["personId"].map(names),
            "SHOT_MADE_FLAG": (s["actionType"] == "Made Shot").astype(int),
            "SHOT_VALUE": s["shotValue"],
            "ACTION_TYPE": s["subType"],
            "SHOT_DISTANCE": s["shotDistance"],
            "x_ft": s["xLegacy"] / 10,
            "y_ft": s["yLegacy"] / 10,
        })
        out.to_parquet(table_path("game_shots", "game_shots.parquet", season), index=False)

        # ---------- checks ----------
        # 1. Shot count per team per game = that team's FGA in the box score
        fga = box.groupby(["game_id", "team"])["fga"].sum()
        cnt = out.groupby(["game_id", "team"]).size()
        both = pd.concat([fga.rename("box_fga"), cnt.rename("shots")], axis=1).fillna(0)
        fga_ok = (both["box_fga"] == both["shots"]).mean()

        # 2. 76ers coordinates agree with the official shot chart data
        official = pd.read_parquet(table_path("shots", "shots.parquet", season),
                                   columns=["GAME_ID", "GAME_EVENT_ID", "LOC_X", "LOC_Y"])
        official["GAME_EVENT_ID"] = official["GAME_EVENT_ID"].astype("Int64")
        phi = out[out["side"] == "76ers"].merge(
            official, left_on=["game_id", "GAME_EVENT_ID"], right_on=["GAME_ID", "GAME_EVENT_ID"], how="inner")
        matched = len(phi) / max(1, (out["side"] == "76ers").sum())
        same_spot = ((phi["x_ft"] * 10 - phi["LOC_X"]).abs().le(1)
                     & (phi["y_ft"] * 10 - phi["LOC_Y"]).abs().le(1)).mean()

        print(f"{season}: {len(out):,} shots ({(out['side'] == 'Opponent').sum():,} opponent), "
              f"FGA match {fga_ok:.1%} of team-games, "
              f"76ers shots matched to official {matched:.1%}, same coordinates {same_spot:.1%}, "
              f"missing names {out['PLAYER_NAME'].isna().sum()}")


if __name__ == "__main__":
    main()