from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
PREFIX = "trained_hr_model"


def build_backtest_payload() -> dict:
    summary_path = ROOT / f"{PREFIX}_backtest_summary.csv"
    scored_path = ROOT / f"{PREFIX}_scored_test_rows.csv"
    if not summary_path.exists() or not scored_path.exists():
        return {"summary": [], "daily": [], "drivers": []}

    summary = pd.read_csv(summary_path)
    summary_records = [
        {key: clean(value) for key, value in row.items()}
        for row in summary.to_dict("records")
    ]

    scored = pd.read_csv(scored_path)
    scored["game_date"] = pd.to_datetime(scored["game_date"], errors="coerce")
    scored = scored.dropna(subset=["game_date"]).copy()
    sort_col = "pred_hr_prob" if "pred_hr_prob" in scored else "raw_hr_prob"
    scored = scored.sort_values(sort_col, ascending=False).drop_duplicates(["game_date", "batter"])
    daily_records = []
    driver_records = []
    driver_columns = {
        "batter_power_score_prior": "Batter power", "batter_recent_hr_rate_10": "Recent HR rate",
        "batter_barrel_rate_prior": "Barrel rate", "batter_hard_hit_rate_prior": "Hard-hit rate",
        "pitcher_damage_score_prior": "Pitcher vulnerability", "pitcher_hr_rate_allowed_prior": "Pitcher HR rate allowed",
        "pitcher_recent_hr_allowed_rate_10": "Recent pitcher HR rate allowed", "pitcher_k_rate_prior": "Pitcher strikeout rate",
        "park_factor": "Park factor", "temp_f": "Temperature", "pull_wind_mph": "Pull-side wind",
        "batter_recent_pa_10": "Recent plate appearances",
    }
    for top_n in [10, 20, 30, 40]:
        ranked = scored.groupby("game_date", as_index=False, group_keys=False).head(top_n)
        daily = ranked.groupby("game_date", as_index=False).agg(
            players=("batter", "count"), homers=("home_run_game", "sum"), avg_model_prob=(sort_col, "mean")
        ).sort_values("game_date")
        daily["hit_rate"] = daily["homers"] / daily["players"]
        daily["cumulative_players"] = daily["players"].cumsum()
        daily["cumulative_homers"] = daily["homers"].cumsum()
        daily["cumulative_hit_rate"] = daily["cumulative_homers"] / daily["cumulative_players"]
        daily["top_n"] = top_n
        daily["game_date"] = daily["game_date"].dt.strftime("%Y-%m-%d")
        daily_records.extend(
            {key: clean(value) for key, value in row.items()} for row in daily.to_dict("records")
        )
        available = [c for c in driver_columns if c in ranked.columns]
        if available and len(daily) >= 8:
            outcomes = daily.copy()
            outcomes["game_date"] = pd.to_datetime(outcomes["game_date"], errors="coerce")
            analysis = ranked.groupby("game_date")[available].mean(numeric_only=True).join(
                outcomes.set_index("game_date")["hit_rate"], how="inner"
            ).dropna(subset=["hit_rate"])
            low_cut, high_cut = analysis.hit_rate.quantile(.25), analysis.hit_rate.quantile(.75)
            for col in available:
                sample = analysis[[col, "hit_rate"]].dropna()
                if len(sample) < 8 or sample[col].nunique() < 2: continue
                median = sample[col].median(); lower = sample[sample[col] <= median].hit_rate; upper = sample[sample[col] > median].hit_rate
                driver_records.append({"top_n": top_n, "metric": col, "label": driver_columns[col],
                    "correlation": clean(sample[col].corr(sample.hit_rate)),
                    "low_day_avg": clean(sample.loc[sample.hit_rate <= low_cut, col].mean()),
                    "high_day_avg": clean(sample.loc[sample.hit_rate >= high_cut, col].mean()),
                    "median": clean(median), "hit_rate_below_median": clean(lower.mean()),
                    "hit_rate_above_median": clean(upper.mean()), "days_below": int(len(lower)), "days_above": int(len(upper))})
    return {"summary": summary_records, "daily": daily_records, "drivers": driver_records}


def latest_board() -> Path:
    boards = sorted(ROOT.glob(f"{PREFIX}_board_????-??-??.csv"))
    boards = [p for p in boards if "graded" not in p.name]
    if not boards:
        raise FileNotFoundError("No daily board CSV was produced.")
    return boards[-1]


def clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def main() -> None:
    try:
        board_path = latest_board()
        frame = pd.read_csv(board_path).sort_values("ranking")
    except FileNotFoundError:
        board_path = None
        frame = pd.DataFrame()
    columns = [
        "ranking", "game_pk", "commence_time", "batter_name_hand", "batting_team",
        "fielding_team", "is_home_batter", "game_matchup", "pitcher_name_hand",
        "final_hr_probability", "calibrated_hr_probability", "bet_quality_score",
        "batter_power", "recent_form", "pitcher_vulnerability", "handedness_splits",
        "pitch_type_matchup", "matchup_history", "environment", "pa_opportunity",
        "batter_pa_prior", "batter_recent_pa_10", "batter_hr_rate_prior",
        "batter_recent_hr_rate_10", "batter_recent_hr_rate_20",
        "batter_barrel_rate_prior", "batter_hard_hit_rate_prior", "batter_avg_ev_prior",
        "batter_hr_rate_vs_hand_prior", "pitcher_hr_rate_allowed_prior",
        "pitcher_recent_hr_allowed_rate_10", "pitcher_barrel_rate_allowed_prior",
        "pitcher_hard_hit_rate_allowed_prior", "pitcher_k_rate_prior",
        "matchup_pa_prior", "matchup_hr_prior", "matchup_hr_rate_prior",
        "pitch_fit_score_prior", "platoon_advantage", "temp_f", "wind_speed_mph",
        "weather_blowing_out", "wind_out_to_pull_flag", "pull_wind_mph",
        "wind_to_lf_mph", "wind_to_cf_mph", "wind_to_rf_mph",
        "relative_humidity", "is_roofed_no_wind", "park_factor",
    ]
    records = [
        {key: clean(value) for key, value in row.items()}
        for row in frame[[c for c in columns if c in frame.columns]].to_dict("records")
    ]
    if not frame.empty and "target_date" in frame:
        target_date = str(frame["target_date"].iloc[0])
    elif board_path is not None:
        target_date = board_path.stem[-10:]
    else:
        target_date = os.getenv("TARGET_DATE", datetime.now(timezone.utc).date().isoformat())
    payload = {
        "targetDate": target_date,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "featuredCount": min(40, len(records)),
        "backtest": build_backtest_payload(),
        "rows": records,
    }
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    (SITE / "data" / "board.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    frame.to_csv(SITE / "data" / "latest-board.csv", index=False)


if __name__ == "__main__":
    main()
