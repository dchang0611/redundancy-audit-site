from __future__ import annotations

import json
import os
import re
import unicodedata
from urllib.request import urlopen
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
PREFIX = "trained_hr_model"
HISTORY = SITE / "data" / "history"


def restore_history() -> None:
    """Carry prior deployed snapshots into the next immutable Pages artifact."""
    base = os.getenv("HISTORY_BASE_URL", "https://dchang0611.github.io/dc-daily-home-run-board").rstrip("/")
    HISTORY.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(f"{base}/data/history/index.json", timeout=15) as response:
            index = json.load(response)
        for slate_date in index.get("dates", []):
            if not str(slate_date).replace("-", "").isdigit():
                continue
            with urlopen(f"{base}/data/history/{slate_date}.json", timeout=15) as response:
                (HISTORY / f"{slate_date}.json").write_bytes(response.read())
        try:
            with urlopen(f"{base}/data/model-impact-test.json", timeout=15) as response:
                (SITE / "data" / "model-impact-test.json").write_bytes(response.read())
        except Exception:
            pass
    except Exception as exc:
        print(f"No prior history archive restored: {exc}")


def live_backtest_fallback() -> dict:
    try:
        with urlopen("https://dchang0611.github.io/dc-daily-home-run-board/data/board.json", timeout=15) as response:
            return json.load(response).get("backtest", {"summary": [], "daily": [], "drivers": []})
    except Exception:
        return {"summary": [], "daily": [], "drivers": []}


def normalize_player_name(value) -> str:
    """Normalize display names so archived boards can be joined to game results."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\s*\([LRS]\)\s*$", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def build_backtest_payload() -> dict:
    """Grade the exact archived boards; use model test rows only as outcomes."""
    scored_path = ROOT / f"{PREFIX}_scored_test_rows.csv"
    history_paths = sorted(HISTORY.glob("????-??-??.json"))
    if not scored_path.exists() or not history_paths:
        return live_backtest_fallback()

    scored = pd.read_csv(scored_path)
    scored["game_date"] = pd.to_datetime(scored["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    scored = scored.dropna(subset=["game_date"]).copy()
    scored["outcome"] = pd.to_numeric(scored.get("home_run_game"), errors="coerce")
    scored = scored.dropna(subset=["outcome"])
    name_col = next((c for c in ["batter_name", "batter_name_hand", "player_name"] if c in scored), None)
    scored["name_key"] = scored[name_col].map(normalize_player_name) if name_col else ""
    if "batter" in scored:
        scored["batter_key"] = pd.to_numeric(scored["batter"], errors="coerce").astype("Int64")
    else:
        scored["batter_key"] = pd.Series(pd.NA, index=scored.index, dtype="Int64")
    if "game_pk" in scored:
        scored["game_pk_key"] = pd.to_numeric(scored["game_pk"], errors="coerce").astype("Int64")
    else:
        scored["game_pk_key"] = pd.Series(pd.NA, index=scored.index, dtype="Int64")
    scored["outcome"] = (scored["outcome"] > 0).astype(int)
    id_rows = scored.dropna(subset=["batter_key"])
    name_rows = scored[scored["name_key"] != ""]
    outcome_by_game_id = id_rows.dropna(subset=["game_pk_key"]).groupby(
        ["game_date", "game_pk_key", "batter_key"]
    )["outcome"].max().to_dict()
    outcome_by_game_name = name_rows.dropna(subset=["game_pk_key"]).groupby(
        ["game_date", "game_pk_key", "name_key"]
    )["outcome"].max().to_dict()
    outcome_by_id = id_rows.groupby(["game_date", "batter_key"])["outcome"].max().to_dict()
    outcome_by_name = name_rows.groupby(["game_date", "name_key"])["outcome"].max().to_dict()

    board_rows = []
    for path in history_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Could not read historical board {path.name}: {exc}")
            continue
        game_date = str(payload.get("targetDate") or path.stem)
        for fallback_rank, row in enumerate(payload.get("rows", []), start=1):
            record = dict(row)
            record["game_date"] = game_date
            ranking = pd.to_numeric(record.get("ranking"), errors="coerce")
            record["ranking"] = int(ranking) if pd.notna(ranking) else fallback_rank
            probability = pd.to_numeric(
                record.get("final_hr_probability", record.get("calibrated_hr_probability")), errors="coerce"
            )
            record["probability"] = probability / 100 if pd.notna(probability) and probability > 1 else probability
            batter_id = pd.to_numeric(record.get("batter"), errors="coerce")
            game_pk = pd.to_numeric(record.get("game_pk"), errors="coerce")
            display_name = record.get("batter_name_hand", record.get("batter_name", ""))
            outcome = None
            name_key = normalize_player_name(display_name)
            if pd.notna(game_pk) and pd.notna(batter_id):
                outcome = outcome_by_game_id.get((game_date, int(game_pk), int(batter_id)))
            if outcome is None and pd.notna(game_pk):
                outcome = outcome_by_game_name.get((game_date, int(game_pk), name_key))
            if pd.notna(batter_id):
                outcome = outcome if outcome is not None else outcome_by_id.get((game_date, int(batter_id)))
            if outcome is None:
                outcome = outcome_by_name.get((game_date, name_key))
            record["outcome"] = outcome
            record["display_name"] = re.sub(r"\s*\([LRS]\)\s*$", "", str(display_name), flags=re.IGNORECASE)
            board_rows.append(record)

    if not board_rows:
        return live_backtest_fallback()

    board = pd.DataFrame(board_rows).sort_values(["game_date", "ranking"])
    daily_records = []
    driver_records = []
    summary_records = []
    driver_columns = {
        "batter_power_score_prior": "Batter power", "batter_recent_hr_rate_10": "Recent HR rate",
        "batter_barrel_rate_prior": "Barrel rate", "batter_hard_hit_rate_prior": "Hard-hit rate",
        "pitcher_damage_score_prior": "Pitcher vulnerability", "pitcher_hr_rate_allowed_prior": "Pitcher HR rate allowed",
        "pitcher_recent_hr_allowed_rate_10": "Recent pitcher HR rate allowed", "pitcher_k_rate_prior": "Pitcher strikeout rate",
        "park_factor": "Park factor", "temp_f": "Temperature", "pull_wind_mph": "Pull-side wind",
        "batter_recent_pa_10": "Recent plate appearances",
    }
    for top_n in [10, 20, 30, 40]:
        selected = board.groupby("game_date", as_index=False, group_keys=False).head(top_n).copy()
        ranked = selected[selected["outcome"].notna()].copy()
        if ranked.empty:
            continue
        ranked["outcome"] = pd.to_numeric(ranked["outcome"], errors="coerce")
        homer_hitters = {
            game_date: [
                {"name": clean(row.display_name), "rank": int(row.ranking), "probability": clean(row.probability)}
                for row in group.sort_values("ranking").itertuples()
            ]
            for game_date, group in ranked[ranked["outcome"] > 0].groupby("game_date")
        }
        daily = ranked.groupby("game_date", as_index=False).agg(
            players=("outcome", "count"), homers=("outcome", "sum"), avg_model_prob=("probability", "mean")
        ).sort_values("game_date")
        daily["hit_rate"] = daily["homers"] / daily["players"].replace(0, pd.NA)
        daily["cumulative_players"] = daily["players"].cumsum()
        daily["cumulative_homers"] = daily["homers"].cumsum()
        daily["cumulative_hit_rate"] = daily["cumulative_homers"] / daily["cumulative_players"]
        daily["top_n"] = top_n
        for row in daily.to_dict("records"):
            record = {key: clean(value) for key, value in row.items()}
            record["home_run_hitters"] = homer_hitters.get(record["game_date"], [])
            daily_records.append(record)
        summary_records.append({
            "top_n": top_n,
            "overall_hit_rate": clean(ranked["outcome"].mean()),
            "avg_daily_hit_rate": clean(daily["hit_rate"].mean()),
            "days": int(len(daily)),
            "total_players": int(daily["players"].sum()),
            "total_homers": int(daily["homers"].sum()),
            "avg_model_prob": clean(ranked["probability"].mean()),
        })
        available = [c for c in driver_columns if c in ranked.columns]
        if available and len(daily) >= 8:
            ranked[available] = ranked[available].apply(pd.to_numeric, errors="coerce")
            outcomes = daily.copy()
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
    restore_history()
    try:
        board_path = latest_board()
        frame = pd.read_csv(board_path).sort_values("ranking")
    except FileNotFoundError:
        board_path = None
        frame = pd.DataFrame()
    columns = [
        "ranking", "game_pk", "commence_time", "batter", "batter_name_hand", "batting_team",
        "fielding_team", "is_home_batter", "game_matchup", "pitcher_name_hand",
        "lineup_confirmed", "lineup_status",
        "final_hr_probability", "calibrated_hr_probability", "bet_quality_score",
        "batter_power", "recent_form", "pitcher_vulnerability", "handedness_splits",
        "pitch_type_matchup", "matchup_history", "environment", "pa_opportunity",
        "batter_pa_prior", "batter_recent_pa_10", "batter_hr_rate_prior",
        "batter_recent_hr_rate_10", "batter_recent_hr_rate_20",
        "batter_barrel_rate_prior", "batter_hard_hit_rate_prior", "batter_avg_ev_prior",
        "batter_power_score_prior",
        "batter_hr_rate_vs_hand_prior", "pitcher_hr_rate_allowed_prior",
        "pitcher_recent_hr_allowed_rate_10", "pitcher_barrel_rate_allowed_prior",
        "pitcher_hard_hit_rate_allowed_prior", "pitcher_k_rate_prior",
        "pitcher_damage_score_prior",
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
    archive_payload = {
        "targetDate": target_date,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "featuredCount": min(40, len(records)),
        "rows": records,
    }
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    if records:
        (HISTORY / f"{target_date}.json").write_text(json.dumps(archive_payload, indent=2), encoding="utf-8")
    payload = dict(archive_payload)
    payload["backtest"] = build_backtest_payload()
    (SITE / "data" / "board.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history_dates = sorted((p.stem for p in HISTORY.glob("????-??-??.json")), reverse=True)
    (HISTORY / "index.json").write_text(json.dumps({"dates": history_dates}, indent=2), encoding="utf-8")
    frame.to_csv(SITE / "data" / "latest-board.csv", index=False)


if __name__ == "__main__":
    main()
