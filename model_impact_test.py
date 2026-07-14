from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.base import clone


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache_trained_model"
STARTER_CACHE = CACHE_DIR / "impact_starting_batters.json"
OUTPUT = ROOT / "trained_hr_model_change_impact.json"


def _clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_starter_cache() -> dict[str, list[int]]:
    if not STARTER_CACHE.exists():
        return {}
    try:
        return json.loads(STARTER_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _starting_batters(game_pk: int, cache: dict[str, list[int]]) -> set[int]:
    key = str(game_pk)
    if key in cache:
        return {int(value) for value in cache[key]}

    ids: set[int] = set()
    try:
        response = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            timeout=30,
        )
        response.raise_for_status()
        boxscore = response.json().get("liveData", {}).get("boxscore", {})
        for side in ("away", "home"):
            players = boxscore.get("teams", {}).get(side, {}).get("players", {})
            for player in players.values():
                order = pd.to_numeric(player.get("battingOrder"), errors="coerce")
                player_id = player.get("person", {}).get("id")
                if player_id is not None and pd.notna(order) and int(order) % 100 == 0:
                    ids.add(int(player_id))
    except Exception as exc:
        print(f"Could not retrieve starting lineup for game {game_pk}: {exc}")

    cache[key] = sorted(ids)
    return ids


def _add_confirmed_starter_flag(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cache = _load_starter_cache()
    starter_keys: set[tuple[int, int]] = set()
    covered_games = 0
    game_ids = sorted(pd.to_numeric(frame["game_pk"], errors="coerce").dropna().astype(int).unique())
    for game_pk in game_ids:
        ids = _starting_batters(game_pk, cache)
        if ids:
            covered_games += 1
            starter_keys.update((game_pk, batter) for batter in ids)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STARTER_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    out = frame.copy()
    out["game_pk_key"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["batter_key"] = pd.to_numeric(out["batter"], errors="coerce").astype("Int64")
    out["confirmed_starter"] = [
        int(pd.notna(game_pk) and pd.notna(batter) and (int(game_pk), int(batter)) in starter_keys)
        for game_pk, batter in zip(out["game_pk_key"], out["batter_key"])
    ]
    coverage = {
        "games_requested": len(game_ids),
        "games_with_lineups": covered_games,
        "coverage_rate": covered_games / len(game_ids) if game_ids else 0.0,
    }
    return out, coverage


def _summarize_variant(frame: pd.DataFrame, probability_col: str, variant: str) -> tuple[list[dict], list[dict]]:
    base = frame.dropna(subset=[probability_col, "game_date", "batter"]).copy()
    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    base = base.sort_values(["game_date", probability_col], ascending=[True, False])
    base = base.drop_duplicates(["game_date", "batter"], keep="first")
    summaries: list[dict] = []
    daily_records: list[dict] = []

    for top_n in (10, 20, 30, 40):
        ranked = base.groupby("game_date", as_index=False, group_keys=False).head(top_n).copy()
        daily = ranked.groupby("game_date", as_index=False).agg(
            players=("batter", "count"),
            homers=("home_run_game", "sum"),
            avg_model_prob=(probability_col, "mean"),
        )
        daily["hit_rate"] = daily["homers"] / daily["players"]
        daily["variant"] = variant
        daily["top_n"] = top_n
        daily_records.extend(
            {key: _clean(value) for key, value in row.items()}
            for row in daily.to_dict("records")
        )
        summaries.append({
            "variant": variant,
            "top_n": top_n,
            "days": int(daily["game_date"].nunique()),
            "total_players": int(daily["players"].sum()),
            "total_homers": int(daily["homers"].sum()),
            "overall_hit_rate": _clean(daily["homers"].sum() / daily["players"].sum()),
            "avg_daily_hit_rate": _clean(daily["hit_rate"].mean()),
            "avg_model_prob": _clean(ranked[probability_col].mean()),
        })
    return summaries, daily_records


def run_model_impact_test(
    frozen_model,
    model_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    sample_weight_fn,
) -> dict:
    print("\n=== RUNNING 30-DAY MODEL CHANGE IMPACT TEST ===")
    native = test_df.copy()
    native["fresh_frozen_prob"] = np.clip(
        frozen_model.predict_proba(native[feature_columns])[:, 1], 0.0, 1.0
    )
    native, lineup_coverage = _add_confirmed_starter_flag(native)
    confirmed = native[native["confirmed_starter"] == 1].copy()

    rolling_parts = []
    model_dates = pd.to_datetime(model_df["game_date"], errors="coerce")
    confirmed["game_date_dt"] = pd.to_datetime(confirmed["game_date"], errors="coerce")
    for game_date, day in confirmed.groupby("game_date_dt", sort=True):
        history = model_df[model_dates < game_date].copy()
        if history.empty or day.empty:
            continue
        rolling_model = clone(frozen_model)
        weights = sample_weight_fn(history)
        rolling_model.fit(
            history[feature_columns],
            history["home_run_game"].astype(int),
            sample_weight=weights,
        )
        scored_day = day.copy()
        scored_day["fresh_rolling_prob"] = np.clip(
            rolling_model.predict_proba(scored_day[feature_columns])[:, 1], 0.0, 1.0
        )
        rolling_parts.append(scored_day)
        print(f"Rolling replay complete: {game_date.date()} ({len(scored_day)} confirmed starters)")

    rolling = pd.concat(rolling_parts, ignore_index=True) if rolling_parts else pd.DataFrame()
    summaries: list[dict] = []
    daily: list[dict] = []
    variants = [
        (native, "fresh_frozen_prob", "fresh_rows_all_appearances"),
        (confirmed, "fresh_frozen_prob", "fresh_rows_confirmed_starters"),
        (rolling, "fresh_rolling_prob", "fresh_starters_rolling_refit"),
    ]
    for frame, probability_col, variant in variants:
        if frame.empty:
            continue
        variant_summary, variant_daily = _summarize_variant(frame, probability_col, variant)
        summaries.extend(variant_summary)
        daily.extend(variant_daily)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Thirty-day historical replay using pregame feature rows. Confirmed-starter variants use official historical starting lineups. Rolling refit trains only on dates before each replay date.",
        "caveats": [
            "The all-appearances variant is an upper-bound diagnostic because it includes players who entered as substitutes.",
            "The confirmed-starter variants approximate a board run after official lineups were available.",
            "This test does not yet include the proposed per-plate-appearance model.",
        ],
        "lineup_coverage": lineup_coverage,
        "summaries": summaries,
        "daily": daily,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT.name}")
    return payload
