"""
train_model.py
--------------
Trains a Random Forest model that scores every player on a 0-100 scale
("SupeRR Score") for BATTING and BOWLING separately.

The model is unsupervised (no labeled "good/bad" column in the data).
Instead we:
  1. Engineer a composite "expert score" from known cricket KPIs
  2. Use that score as the training label (supervised regression)
  3. This lets the ML model learn the *feature interactions* and generalise
     to unseen players, even without manual labels.

Outputs (saved to models/):
  - batting_model.pkl
  - bowling_model.pkl
  - batting_scaler.pkl
  - bowling_scaler.pkl
  - feature_importance_bat.csv
  - feature_importance_bow.csv
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

PROC_DIR   = os.path.join(os.path.dirname(__file__), "processed")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Expert scoring functions (domain knowledge) ───────────────────────────────

def batting_expert_score(df: pd.DataFrame) -> pd.Series:
    """
    Creates a composite batting score (0–100) using weights drawn from
    fantasy-cricket / auction analytics frameworks.
    """
    s = pd.Series(0.0, index=df.index)

    # core quality (40 pts)
    sr_norm  = df["strike_rate"].clip(0, 250) / 250
    avg_norm = df["batting_avg"].clip(0, 80) / 80
    s += 20 * sr_norm + 20 * avg_norm

    # boundary hitting (15 pts)
    bnd_norm = df["boundary_pct"].clip(0, 60) / 60
    s += 15 * bnd_norm

    # phase coverage (15 pts)
    sr_pp_n  = df["sr_powerplay"].clip(0, 200) / 200
    sr_mid_n = df["sr_middle"].clip(0, 180) / 180
    sr_dth_n = df["sr_death"].clip(0, 300) / 300
    s += 5 * sr_pp_n + 5 * sr_mid_n + 5 * sr_dth_n

    # match impact / big scores (20 pts)
    half_norm = df["50_plus"].clip(0, 20) / 20
    thirty_n  = df["30_plus"].clip(0, 40) / 40
    s += 12 * half_norm + 8 * thirty_n

    # consistency (10 pts)
    cons_n = df["consistency_pct"].clip(0, 100) / 100
    s += 10 * cons_n

    # penalise excessive dots (-5 to 0)
    dot_pen = df["dot_ball_pct"].clip(0, 60) / 60
    s -= 5 * dot_pen

    return s.clip(0, 100)


def bowling_expert_score(df: pd.DataFrame) -> pd.Series:
    """
    Creates a composite bowling score (0–100).
    """
    s = pd.Series(0.0, index=df.index)

    # economy (lower is better) – inverted, capped at 6-16 era (40 pts)
    econ_inv = 1 - (df["economy"].clip(4, 16) - 4) / 12
    s += 20 * econ_inv

    # bowling average (lower is better) (20 pts)
    avg_inv  = 1 - (df["bowling_avg"].clip(10, 80) - 10) / 70
    avg_inv  = avg_inv.where(df["wickets"] > 0, 0)  # no wickets → 0
    s += 20 * avg_inv

    # strike rate (lower is better) (20 pts)
    sr_inv = 1 - (df["strike_rate_b"].clip(5, 60) - 5) / 55
    sr_inv = sr_inv.where(df["wickets"] > 0, 0)
    s += 20 * sr_inv

    # dot ball % (higher is better) (10 pts)
    dot_n = df["dot_ball_pct"].clip(20, 70) / 70
    s += 10 * dot_n

    # wicket hauls (10 pts)
    haul_n = (df["3wkt_hauls"] * 1 + df["5wkt_hauls"] * 2).clip(0, 15) / 15
    s += 10 * haul_n

    # death phase ability (bonus up to +5)
    dth_n = 1 - (df["econ_death"].clip(5, 20) - 5) / 15
    s += 5 * dth_n

    # penalise wides/no-balls
    extra_pen = (df["wides"] + df["no_balls"]).clip(0, 50) / 50
    s -= 5 * extra_pen

    return s.clip(0, 100)


# ── Feature sets ──────────────────────────────────────────────────────────────

BAT_FEATURES = [
    "strike_rate", "batting_avg", "boundary_pct", "dot_ball_pct",
    "sr_powerplay", "sr_middle", "sr_death",
    "30_plus", "50_plus", "consistency_pct",
    "fours", "sixes", "innings", "total_runs", "balls_faced"
]

BOW_FEATURES = [
    "economy", "bowling_avg", "strike_rate_b", "dot_ball_pct",
    "econ_powerplay", "econ_middle", "econ_death",
    "wickets", "3wkt_hauls", "5wkt_hauls",
    "wides", "no_balls", "innings_bowled", "balls_bowled", "runs_conceded"
]


# ── Training ──────────────────────────────────────────────────────────────────

def train(features_csv: str, feature_cols: list, score_fn, role: str):
    df = pd.read_csv(features_csv)

    # Only keep players with meaningful sample sizes
    if role == "batting":
        df = df[df["balls_faced"] >= 30].copy()
    else:
        df = df[df["balls_bowled"] >= 24].copy()  # at least 4 overs bowled

    df = df.fillna(0)

    # generate labels via expert heuristic
    df["expert_score"] = score_fn(df)

    X = df[feature_cols].fillna(0)
    y = df["expert_score"]

    # scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # train Random Forest
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    print(f"  [{role}] R² CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # predict suprrr score (0-100) and save back to df
    df["superr_score"] = model.predict(X_scaled).clip(0, 100).round(2)

    # feature importances
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, scaler, df, fi


def main():
    print("=== Training Batting Model ===")
    bat_model, bat_scaler, bat_df, bat_fi = train(
        os.path.join(PROC_DIR, "batting_features.csv"),
        BAT_FEATURES,
        batting_expert_score,
        "batting"
    )

    print("=== Training Bowling Model ===")
    bow_model, bow_scaler, bow_df, bow_fi = train(
        os.path.join(PROC_DIR, "bowling_features.csv"),
        BOW_FEATURES,
        bowling_expert_score,
        "bowling"
    )

    # save models
    for name, obj in [
        ("batting_model.pkl",  bat_model),
        ("batting_scaler.pkl", bat_scaler),
        ("bowling_model.pkl",  bow_model),
        ("bowling_scaler.pkl", bow_scaler),
    ]:
        with open(os.path.join(MODEL_DIR, name), "wb") as f:
            pickle.dump(obj, f)

    bat_fi.to_csv(os.path.join(MODEL_DIR, "feature_importance_bat.csv"), index=False)
    bow_fi.to_csv(os.path.join(MODEL_DIR, "feature_importance_bow.csv"), index=False)

    # save scored data
    bat_df.to_csv(os.path.join(PROC_DIR, "batting_scored.csv"), index=False)
    bow_df.to_csv(os.path.join(PROC_DIR, "bowling_scored.csv"), index=False)

    print("\nTop 10 Batters by SupeRR Score:")
    print(bat_df.nlargest(10, "superr_score")[["player_name", "total_runs", "strike_rate", "batting_avg", "superr_score"]])

    print("\nTop 10 Bowlers by SupeRR Score:")
    print(bow_df.nlargest(10, "superr_score")[["player_name", "wickets", "economy", "bowling_avg", "superr_score"]])

    print("\nModels saved to models/")
    print("Scored data saved to processed/")


if __name__ == "__main__":
    main()
