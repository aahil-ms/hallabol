"""
data_processor.py
-----------------
Loads raw CSVs from the Data Pack and engineers per-player features
for BOTH batting and bowling roles. The output is saved as:
  - processed/batting_features.csv
  - processed/bowling_features.csv
  - processed/player_lookup.csv
"""

import os
import pandas as pd
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load ─────────────────────────────────────────────────────────────────────
def load_data():
    bbb       = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
    matches   = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
    playing11 = pd.read_csv(os.path.join(DATA_DIR, "playing_eleven.csv"))
    comps     = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
    players   = pd.read_csv(os.path.join(DATA_DIR, "competition_team_player_mapping.csv"))
    teams     = pd.read_csv(os.path.join(DATA_DIR, "teams.csv"))
    return bbb, matches, playing11, comps, players, teams


# ── helpers ───────────────────────────────────────────────────────────────────
def _legal_ball(bbb):
    """Return only legal deliveries (no wides/no-balls that add extras)."""
    return bbb[~bbb["extras_type"].isin(["wide", "Overstep"])]


# ── BATTING features ──────────────────────────────────────────────────────────
def build_batting_features(bbb, matches, playing11):
    """
    Per-player batting aggregation.
    Returns DataFrame with index = player_id.
    """
    legal = _legal_ball(bbb)

    # merge match metadata for phase / over info
    bbb2 = bbb.merge(matches[["match_id", "comp_id", "match_date"]], on="match_id", how="left")
    legal2 = legal.merge(matches[["match_id", "comp_id", "match_date"]], on="match_id", how="left")

    # Phase buckets: powerplay 1-6, middle 7-15, death 16-20
    def phase(ov):
        if ov <= 6:   return "powerplay"
        if ov <= 15:  return "middle"
        return "death"

    bbb2["phase"]  = bbb2["overid"].apply(phase)
    legal2["phase"] = legal2["overid"].apply(phase)

    grp = legal2.groupby("striker_id")

    bat = pd.DataFrame()
    bat["player_name"]      = bbb2.groupby("striker_id")["striker_name"].first()

    bat["matches_batted"]   = playing11[playing11["is_played"] == True].groupby("player_id").size()
    bat["innings"]          = grp["match_id"].nunique()
    bat["total_runs"]       = grp["runs"].sum()
    bat["balls_faced"]      = grp.size()
    bat["fours"]            = legal2.groupby("striker_id")["is_four"].sum()
    bat["sixes"]            = legal2.groupby("striker_id")["is_six"].sum()
    bat["dismissals"]       = bbb2.groupby("striker_id")["is_wicket"].sum()

    bat["strike_rate"]      = (bat["total_runs"] / bat["balls_faced"] * 100).round(2)
    bat["batting_avg"]      = (bat["total_runs"] / bat["dismissals"].replace(0, np.nan)).round(2)
    bat["boundary_pct"]     = ((bat["fours"] * 4 + bat["sixes"] * 6) / bat["total_runs"].replace(0, np.nan) * 100).round(2)
    bat["dot_ball_pct"]     = (legal2.groupby("striker_id").apply(lambda x: (x["runs"] == 0).sum()) / bat["balls_faced"] * 100).round(2)

    # Phase-specific SR
    for ph in ["powerplay", "middle", "death"]:
        sub = legal2[legal2["phase"] == ph].groupby("striker_id")
        runs_ph  = sub["runs"].sum()
        balls_ph = sub.size()
        bat[f"sr_{ph}"]   = (runs_ph / balls_ph.replace(0, np.nan) * 100).round(2)

    # 30+ and 50+ scores (innings with ≥30 or ≥50 runs)
    inn_runs = bbb2.groupby(["striker_id", "match_id"])["runs"].sum().reset_index()
    bat["30_plus"] = inn_runs[inn_runs["runs"] >= 30].groupby("striker_id").size()
    bat["50_plus"] = inn_runs[inn_runs["runs"] >= 50].groupby("striker_id").size()
    bat["30_plus"] = bat["30_plus"].fillna(0)
    bat["50_plus"] = bat["50_plus"].fillna(0)

    # Consistency = % innings with ≥15 runs
    inn_15 = inn_runs[inn_runs["runs"] >= 15].groupby("striker_id").size()
    bat["consistency_pct"] = (inn_15 / bat["innings"].replace(0, np.nan) * 100).round(2)

    bat = bat.fillna(0)
    bat.index.name = "player_id"
    return bat.reset_index()


# ── BOWLING features ──────────────────────────────────────────────────────────
def build_bowling_features(bbb, matches):
    legal = _legal_ball(bbb)
    bbb2  = bbb.merge(matches[["match_id", "comp_id"]], on="match_id", how="left")
    legal2 = legal.merge(matches[["match_id", "comp_id"]], on="match_id", how="left")

    def phase(ov):
        if ov <= 6:   return "powerplay"
        if ov <= 15:  return "middle"
        return "death"

    legal2["phase"] = legal2["overid"].apply(phase)

    # wicket-taking balls (excludes run-outs for bowler credit)
    bowler_wickets = bbb2[
        bbb2["is_wicket"] &
        ~bbb2["dismissal_type"].isin(["RunOut", "Obstracting the field", "Obstructing the field"])
    ]

    grp   = legal2.groupby("bowler_id")
    grp_b = bbb2.groupby("bowler_id")

    bow = pd.DataFrame()
    bow["player_name"]   = bbb2.groupby("bowler_id")["bowler_name"].first()
    bow["innings_bowled"] = grp["match_id"].nunique()
    bow["balls_bowled"]  = grp.size()
    bow["runs_conceded"] = grp["runs"].sum()
    bow["wickets"]       = bowler_wickets.groupby("bowler_id").size()
    bow["wickets"]       = bow["wickets"].fillna(0)
    bow["wides"]         = bbb2[bbb2["extras_type"] == "wide"].groupby("bowler_id").size().fillna(0)
    bow["no_balls"]      = bbb2[bbb2["extras_type"] == "Overstep"].groupby("bowler_id").size().fillna(0)

    bow["economy"]       = (bow["runs_conceded"] / (bow["balls_bowled"] / 6).replace(0, np.nan)).round(2)
    bow["bowling_avg"]   = (bow["runs_conceded"] / bow["wickets"].replace(0, np.nan)).round(2)
    bow["strike_rate_b"] = (bow["balls_bowled"] / bow["wickets"].replace(0, np.nan)).round(2)
    bow["dot_ball_pct"]  = (legal2.groupby("bowler_id").apply(lambda x: (x["runs"] == 0).sum()) / bow["balls_bowled"].replace(0, np.nan) * 100).round(2)

    # Phase economy
    for ph in ["powerplay", "middle", "death"]:
        sub = legal2[legal2["phase"] == ph].groupby("bowler_id")
        runs_ph  = sub["runs"].sum()
        balls_ph = sub.size()
        bow[f"econ_{ph}"] = (runs_ph / (balls_ph / 6).replace(0, np.nan)).round(2)

    # 3-wkt & 5-wkt hauls
    inn_wkts = bowler_wickets.groupby(["bowler_id", "match_id"]).size().reset_index(name="wkts")
    bow["3wkt_hauls"] = inn_wkts[inn_wkts["wkts"] >= 3].groupby("bowler_id").size().fillna(0)
    bow["5wkt_hauls"] = inn_wkts[inn_wkts["wkts"] >= 5].groupby("bowler_id").size().fillna(0)
    bow["3wkt_hauls"] = bow["3wkt_hauls"].fillna(0)
    bow["5wkt_hauls"] = bow["5wkt_hauls"].fillna(0)

    bow = bow.fillna(0)
    bow.index.name = "player_id"
    return bow.reset_index()


# ── PLAYER lookup ─────────────────────────────────────────────────────────────
def build_player_lookup(players, playing11, matches, comps):
    # most recent team per player
    p11_m = playing11.merge(matches[["match_id", "comp_id", "match_date"]], on="match_id")
    p11_m = p11_m.merge(comps[["comp_id", "comp_name"]], on="comp_id", how="left")
    p11_m["match_date"] = pd.to_datetime(p11_m["match_date"], errors="coerce")
    latest = p11_m.sort_values("match_date").groupby("player_id").last().reset_index()
    lookup = latest[["player_id", "team_id"]].copy()
    # add name from players mapping
    pname = players[["player_id", "player_name", "team_name"]].drop_duplicates("player_id")
    lookup = lookup.merge(pname, on="player_id", how="left")
    return lookup


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    bbb, matches, playing11, comps, players, teams = load_data()

    print("Building batting features...")
    bat_df = build_batting_features(bbb, matches, playing11)
    bat_df.to_csv(os.path.join(OUT_DIR, "batting_features.csv"), index=False)
    print(f"  → {len(bat_df)} batters saved")

    print("Building bowling features...")
    bow_df = build_bowling_features(bbb, matches)
    bow_df.to_csv(os.path.join(OUT_DIR, "bowling_features.csv"), index=False)
    print(f"  → {len(bow_df)} bowlers saved")

    print("Building player lookup...")
    lookup = build_player_lookup(players, playing11, matches, comps)
    lookup.to_csv(os.path.join(OUT_DIR, "player_lookup.csv"), index=False)
    print(f"  → {len(lookup)} players in lookup")

    print("Done! Check the 'processed/' folder.")
