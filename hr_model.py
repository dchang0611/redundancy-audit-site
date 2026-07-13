from __future__ import annotations

from pybaseball import statcast, playerid_reverse_lookup
import requests
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.inspection import permutation_importance


# =========================================================
# SETTINGS
# =========================================================
TODAY = date.today()
FULL_DATA_START_DATE = os.getenv("FULL_DATA_START_DATE", "2025-04-01")
FULL_DATA_END_DATE = os.getenv("FULL_DATA_END_DATE", (TODAY - timedelta(days=1)).isoformat())

# For daily forward scoring
TARGET_DATE = os.getenv("TARGET_DATE", TODAY.isoformat())

# Earliest date allowed for model rows after pregame features are built.
# This helps avoid tiny-sample rows early in the season.
MODEL_ROW_START_DATE = "2025-05-01"

# Time split for validation / backtest
TRAIN_END_DATE = os.getenv("TRAIN_END_DATE", (TODAY - timedelta(days=60)).isoformat())
VALID_END_DATE = os.getenv("VALID_END_DATE", (TODAY - timedelta(days=30)).isoformat())

# Recency weighting for model training rows.
# Keeps 2024 in the sample for stability, but lets newer MLB run environments / player skill matter more.
USE_RECENCY_SAMPLE_WEIGHTS = True
RECENCY_WEIGHT_2024 = 0.60
RECENCY_WEIGHT_2025 = 0.90
RECENCY_WEIGHT_2026 = 1.40

# If True, build historical out-of-sample backtest rows for all dates > VALID_END_DATE
RUN_BACKTEST = True

# If True, also create a forward-looking board for TARGET_DATE
RUN_FORWARD_BOARD = True

# If True, grade the forward board against official MLB boxscores for TARGET_DATE
RUN_HR_CHECK = os.getenv("RUN_HR_CHECK", "false").lower() == "true"
HR_CHECK_TOP_NS = [10, 20, 30]

# Cache
USE_CACHE = True
REFRESH_CACHE = False
# If True, append only missing Statcast dates to the existing PA cache instead of re-pulling the full range.
INCREMENTAL_STATCAST_CACHE_UPDATE = True
# Internal flag: set when the PA cache changed, so the engineered model dataset cache is rebuilt.
STATCAST_CACHE_UPDATED = False

# Incremental engineered-feature cache.
# This avoids rebuilding the full model dataset after a daily Statcast append.
INCREMENTAL_MODEL_DATASET_UPDATE = True
MODEL_FEATURE_WARMUP_DAYS = 120
MODEL_DATASET_CACHE_VERSION = "k_fb_gb_incremental_v2"
CACHE_DIR = "cache_trained_model"

# Daily board options
MAX_BATTERS_PER_TEAM = 9
MIN_FORWARD_RECENT_PA_10 = 2
TOP_N_OUTPUT = 50

# Feature audit / redundancy pruning
# Keeps the printed/CSV output the same, but trains the probability model on a cleaner feature set.
RUN_FEATURE_AUDIT = True
APPLY_FEATURE_PRUNING = True
FEATURE_CORR_PRUNE_THRESHOLD = 0.92
FEATURE_MIN_PERM_IMPORTANCE = -0.00005
FEATURE_AUDIT_VALID_SAMPLE_ROWS = 6000
FEATURE_AUDIT_PERM_REPEATS = 3
FEATURE_AUDIT_RANDOM_STATE = 42

# Environmental defaults / forecast fallback
DEFAULT_TEMP_F = 72.0
DEFAULT_WIND_SPEED_MPH = 5.0
DEFAULT_WIND_DIRECTION_DEG = 0.0
DEFAULT_REL_HUMIDITY = 50.0
DEFAULT_PARK_FACTOR = 1.00

DYNAMIC_PARK_FACTOR_MIN = 0.85
DYNAMIC_PARK_FACTOR_MAX = 1.25

# =========================================================
# BALLPARK DATA
# Approximate venue coordinates + rough HR park factors.
# =========================================================
TEAM_CONTEXT = {
    # Bearings are compass azimuths in degrees for the line from home plate through 2B/CF.
    # LF/RF are derived as +/-45 degree outfield-sector approximations from the verified CF/home-to-2B azimuth.
    # Source basis: MLB API azimuth list cross-checked against Google Earth collection posted by CaddoxDox; see chat citations.
    "Arizona Diamondbacks": {"lat": 33.4455, "lon": -112.0667, "park_factor": 1.03, "lf_bearing": 314.749168, "cf_bearing": 359.749168, "rf_bearing": 44.749168, "out_to_cf_bearing": 359.749168, "bearing_source": "verified_azimuth"},
    "Atlanta Braves": {"lat": 33.8908, "lon": -84.4677, "park_factor": 1.02, "lf_bearing": 112.646870, "cf_bearing": 157.646870, "rf_bearing": 202.646870, "out_to_cf_bearing": 157.646870, "bearing_source": "verified_azimuth"},
    "Baltimore Orioles": {"lat": 39.2841, "lon": -76.6215, "park_factor": 0.95, "lf_bearing": 346.339785, "cf_bearing": 31.339785, "rf_bearing": 76.339785, "out_to_cf_bearing": 31.339785, "bearing_source": "verified_azimuth"},
    "Boston Red Sox": {"lat": 42.3467, "lon": -71.0972, "park_factor": 1.05, "lf_bearing": 359.174241, "cf_bearing": 44.174241, "rf_bearing": 89.174241, "out_to_cf_bearing": 44.174241, "bearing_source": "verified_azimuth"},
    "Chicago Cubs": {"lat": 41.9484, "lon": -87.6553, "park_factor": 1.03, "lf_bearing": 352.615729, "cf_bearing": 37.615729, "rf_bearing": 82.615729, "out_to_cf_bearing": 37.615729, "bearing_source": "verified_azimuth"},
    "Chicago White Sox": {"lat": 41.8299, "lon": -87.6338, "park_factor": 1.04, "lf_bearing": 82.060672, "cf_bearing": 127.060672, "rf_bearing": 172.060672, "out_to_cf_bearing": 127.060672, "bearing_source": "verified_azimuth"},
    "Cincinnati Reds": {"lat": 39.0979, "lon": -84.5082, "park_factor": 1.08, "lf_bearing": 77.342735, "cf_bearing": 122.342735, "rf_bearing": 167.342735, "out_to_cf_bearing": 122.342735, "bearing_source": "verified_azimuth"},
    "Cleveland Guardians": {"lat": 41.4962, "lon": -81.6852, "park_factor": 0.98, "lf_bearing": 314.254792, "cf_bearing": 359.254792, "rf_bearing": 44.254792, "out_to_cf_bearing": 359.254792, "bearing_source": "verified_azimuth"},
    "Colorado Rockies": {"lat": 39.7559, "lon": -104.9942, "park_factor": 1.18, "lf_bearing": 319.962491, "cf_bearing": 4.962491, "rf_bearing": 49.962491, "out_to_cf_bearing": 4.962491, "bearing_source": "verified_azimuth"},
    "Detroit Tigers": {"lat": 42.3390, "lon": -83.0485, "park_factor": 0.96, "lf_bearing": 106.227049, "cf_bearing": 151.227049, "rf_bearing": 196.227049, "out_to_cf_bearing": 151.227049, "bearing_source": "verified_azimuth"},
    "Houston Astros": {"lat": 29.7573, "lon": -95.3555, "park_factor": 0.99, "lf_bearing": 298.173841, "cf_bearing": 343.173841, "rf_bearing": 28.173841, "out_to_cf_bearing": 343.173841, "bearing_source": "verified_azimuth"},
    "Kansas City Royals": {"lat": 39.0517, "lon": -94.4803, "park_factor": 1.00, "lf_bearing": 1.710182, "cf_bearing": 46.710182, "rf_bearing": 91.710182, "out_to_cf_bearing": 46.710182, "bearing_source": "verified_azimuth"},
    "Los Angeles Angels": {"lat": 33.8003, "lon": -117.8827, "park_factor": 1.00, "lf_bearing": 359.171365, "cf_bearing": 44.171365, "rf_bearing": 89.171365, "out_to_cf_bearing": 44.171365, "bearing_source": "verified_azimuth"},
    "Los Angeles Dodgers": {"lat": 34.0739, "lon": -118.2400, "park_factor": 1.02, "lf_bearing": 341.484540, "cf_bearing": 26.484540, "rf_bearing": 71.484540, "out_to_cf_bearing": 26.484540, "bearing_source": "verified_azimuth"},
    "Miami Marlins": {"lat": 25.7781, "lon": -80.2197, "park_factor": 0.94, "lf_bearing": 83.132189, "cf_bearing": 128.132189, "rf_bearing": 173.132189, "out_to_cf_bearing": 128.132189, "bearing_source": "verified_azimuth"},
    "Milwaukee Brewers": {"lat": 43.0280, "lon": -87.9712, "park_factor": 1.01, "lf_bearing": 83.853705, "cf_bearing": 128.853705, "rf_bearing": 173.853705, "out_to_cf_bearing": 128.853705, "bearing_source": "verified_azimuth"},
    "Minnesota Twins": {"lat": 44.9817, "lon": -93.2776, "park_factor": 1.00, "lf_bearing": 45.503946, "cf_bearing": 90.503946, "rf_bearing": 135.503946, "out_to_cf_bearing": 90.503946, "bearing_source": "verified_azimuth"},
    "New York Mets": {"lat": 40.7571, "lon": -73.8458, "park_factor": 0.94, "lf_bearing": 328.991142, "cf_bearing": 13.991142, "rf_bearing": 58.991142, "out_to_cf_bearing": 13.991142, "bearing_source": "verified_azimuth"},
    "New York Yankees": {"lat": 40.8296, "lon": -73.9262, "park_factor": 1.10, "lf_bearing": 30.617667, "cf_bearing": 75.617667, "rf_bearing": 120.617667, "out_to_cf_bearing": 75.617667, "bearing_source": "verified_azimuth"},
    "Athletics": {"lat": 38.5804, "lon": -121.5139, "park_factor": 1.00, "lf_bearing": 9.511163, "cf_bearing": 54.511163, "rf_bearing": 99.511163, "out_to_cf_bearing": 54.511163, "bearing_source": "verified_azimuth"},
    "Oakland Athletics": {"lat": 38.5804, "lon": -121.5139, "park_factor": 1.00, "lf_bearing": 9.511163, "cf_bearing": 54.511163, "rf_bearing": 99.511163, "out_to_cf_bearing": 54.511163, "bearing_source": "verified_azimuth"},
    "Philadelphia Phillies": {"lat": 39.9057, "lon": -75.1665, "park_factor": 1.05, "lf_bearing": 325.070520, "cf_bearing": 10.070520, "rf_bearing": 55.070520, "out_to_cf_bearing": 10.070520, "bearing_source": "verified_azimuth"},
    "Pittsburgh Pirates": {"lat": 40.4469, "lon": -80.0057, "park_factor": 0.98, "lf_bearing": 71.576424, "cf_bearing": 116.576424, "rf_bearing": 161.576424, "out_to_cf_bearing": 116.576424, "bearing_source": "verified_azimuth"},
    "San Diego Padres": {"lat": 32.7073, "lon": -117.1573, "park_factor": 0.96, "lf_bearing": 314.737475, "cf_bearing": 359.737475, "rf_bearing": 44.737475, "out_to_cf_bearing": 359.737475, "bearing_source": "verified_azimuth"},
    "San Francisco Giants": {"lat": 37.7786, "lon": -122.3893, "park_factor": 0.90, "lf_bearing": 40.151947, "cf_bearing": 85.151947, "rf_bearing": 130.151947, "out_to_cf_bearing": 85.151947, "bearing_source": "verified_azimuth"},
    "Seattle Mariners": {"lat": 47.5914, "lon": -122.3325, "park_factor": 0.92, "lf_bearing": 4.308841, "cf_bearing": 49.308841, "rf_bearing": 94.308841, "out_to_cf_bearing": 49.308841, "bearing_source": "verified_azimuth"},
    "St. Louis Cardinals": {"lat": 38.6226, "lon": -90.1928, "park_factor": 0.99, "lf_bearing": 17.696291, "cf_bearing": 62.696291, "rf_bearing": 107.696291, "out_to_cf_bearing": 62.696291, "bearing_source": "verified_azimuth"},
    "Tampa Bay Rays": {"lat": 27.7682, "lon": -82.6534, "park_factor": 0.95, "lf_bearing": 4.000000, "cf_bearing": 49.000000, "rf_bearing": 94.000000, "out_to_cf_bearing": 49.000000, "bearing_source": "mlb_api_azimuth"},
    "Texas Rangers": {"lat": 32.7513, "lon": -97.0825, "park_factor": 1.05, "lf_bearing": 347.802834, "cf_bearing": 32.802834, "rf_bearing": 77.802834, "out_to_cf_bearing": 32.802834, "bearing_source": "verified_azimuth"},
    "Toronto Blue Jays": {"lat": 43.6414, "lon": -79.3894, "park_factor": 1.02, "lf_bearing": 299.090411, "cf_bearing": 344.090411, "rf_bearing": 29.090411, "out_to_cf_bearing": 344.090411, "bearing_source": "verified_azimuth"},
    "Washington Nationals": {"lat": 38.8730, "lon": -77.0074, "park_factor": 1.01, "lf_bearing": 343.948194, "cf_bearing": 28.948194, "rf_bearing": 73.948194, "out_to_cf_bearing": 28.948194, "bearing_source": "verified_azimuth"},

}

# Parks where this model should treat wind as having no HR impact.
# Includes fixed domes plus retractable-roof parks because the public schedule/weather
# feed used here does not reliably tell us roof-open vs roof-closed at scoring time.
ROOFED_NO_WIND_TEAMS = {
    "Arizona Diamondbacks",   # Chase Field - retractable roof
    "Houston Astros",         # Daikin Park / Minute Maid Park - retractable roof
    "Miami Marlins",          # loanDepot park - retractable roof
    "Milwaukee Brewers",      # American Family Field - retractable roof
    "Seattle Mariners",       # T-Mobile Park - retractable roof
    "Tampa Bay Rays",         # Tropicana Field - fixed dome
    "Texas Rangers",          # Globe Life Field - retractable roof
    "Toronto Blue Jays",      # Rogers Centre - retractable roof
}

def is_roofed_no_wind_team(team: str) -> bool:
    """Return True when wind should be neutralized for this venue/team."""
    if pd.isna(team):
        return False
    return str(team).strip() in ROOFED_NO_WIND_TEAMS


def neutralize_wind_for_roofed_venue(weather: dict, team: str) -> dict:
    """
    For roofed/domed parks, keep temperature/humidity if available, but force all
    wind-based model inputs to zero so wind cannot increase/decrease HR probability.
    """
    out = dict(weather)
    if is_roofed_no_wind_team(team):
        out["wind_speed_mph"] = 0.0
        out["wind_direction_deg"] = DEFAULT_WIND_DIRECTION_DEG
        out["wind_out_flag"] = 0
        out["weather_blowing_out"] = 0
        out["is_roofed_no_wind"] = 1
    else:
        out["is_roofed_no_wind"] = 0
    return out

STRICT_BALLPARK_BEARINGS = True

def validate_team_context() -> None:
    """Fail loudly instead of silently using fake wind bearings."""
    required = ["lat", "lon", "park_factor", "lf_bearing", "cf_bearing", "rf_bearing"]
    problems = []
    for team, ctx in TEAM_CONTEXT.items():
        for key in required:
            val = ctx.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                problems.append(f"{team}: missing {key}")
        for key in ["lf_bearing", "cf_bearing", "rf_bearing"]:
            val = ctx.get(key)
            if val is not None and not (0 <= float(val) < 360):
                problems.append(f"{team}: {key} outside [0, 360): {val}")
    if problems and STRICT_BALLPARK_BEARINGS:
        raise ValueError("Ballpark bearing validation failed; refusing to use guessed wind directions: " + "; ".join(problems))
# HELPERS
# =========================================================
def ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def cache_path(name: str) -> str:
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, name)


def get_json(url: str, params: dict | None = None, timeout: int = 30):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠️ API error: {url}")
        print(str(e))
        return None


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    text = str(name).lower().strip()
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n", "ü": "u", "'": "", ".": ""
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return " ".join(text.split())


def safe_name_lookup(ids, id_col_name, output_name_col):
    ids = pd.Series(ids).dropna().astype(int).unique().tolist()
    if len(ids) == 0:
        return pd.DataFrame(columns=[id_col_name, output_name_col])

    lookup = playerid_reverse_lookup(ids, key_type="mlbam")
    if lookup is None or len(lookup) == 0:
        return pd.DataFrame(columns=[id_col_name, output_name_col])

    lookup = lookup.copy()
    lookup[id_col_name] = pd.to_numeric(lookup["key_mlbam"], errors="coerce")
    lookup[output_name_col] = (
        lookup["name_first"].fillna("").str.strip()
        + " "
        + lookup["name_last"].fillna("").str.strip()
    ).str.strip()

    return lookup[[id_col_name, output_name_col]].drop_duplicates()


def angular_difference(a: float, b: float) -> float:
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def wind_out_flag(wind_direction_deg: float, out_to_cf_bearing: float) -> int:
    wind_to = (wind_direction_deg + 180) % 360
    return int(angular_difference(wind_to, out_to_cf_bearing) <= 45)


def directional_wind_mph(wind_speed_mph: float, wind_direction_deg: float, out_bearing: float) -> float:
    """
    Positive = wind blowing toward that field direction.
    Negative = wind blowing in from that field direction.
    """
    wind_speed_mph = float(wind_speed_mph if pd.notna(wind_speed_mph) else DEFAULT_WIND_SPEED_MPH)
    wind_direction_deg = float(wind_direction_deg if pd.notna(wind_direction_deg) else DEFAULT_WIND_DIRECTION_DEG)
    wind_to = (wind_direction_deg + 180) % 360
    angle = angular_difference(wind_to, float(out_bearing))
    return float(wind_speed_mph * np.cos(np.deg2rad(angle)))


def add_directional_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add LF/CF/RF wind components using the venue context."""
    if df.empty:
        return df
    out = df.copy()
    team_col = "venue_team" if "venue_team" in out.columns else ("home_team" if "home_team" in out.columns else "fielding_team")

    for col in ["lf_bearing", "cf_bearing", "rf_bearing"]:
        if col not in out.columns:
            out[col] = out[team_col].map(lambda t: TEAM_CONTEXT.get(t, {}).get(col, np.nan))

    out["lf_bearing"] = pd.to_numeric(out["lf_bearing"], errors="coerce").fillna(320)
    out["cf_bearing"] = pd.to_numeric(out["cf_bearing"], errors="coerce").fillna(20)
    out["rf_bearing"] = pd.to_numeric(out["rf_bearing"], errors="coerce").fillna(80)

    # Dome/retractable-roof handling: wind should not help or hurt in roofed parks.
    out["is_roofed_no_wind"] = out[team_col].map(is_roofed_no_wind_team).astype(int)
    out.loc[out["is_roofed_no_wind"].eq(1), "wind_speed_mph"] = 0.0
    out.loc[out["is_roofed_no_wind"].eq(1), "wind_direction_deg"] = DEFAULT_WIND_DIRECTION_DEG

    out["wind_to_lf_mph"] = out.apply(lambda r: directional_wind_mph(r["wind_speed_mph"], r["wind_direction_deg"], r["lf_bearing"]), axis=1)
    out["wind_to_cf_mph"] = out.apply(lambda r: directional_wind_mph(r["wind_speed_mph"], r["wind_direction_deg"], r["cf_bearing"]), axis=1)
    out["wind_to_rf_mph"] = out.apply(lambda r: directional_wind_mph(r["wind_speed_mph"], r["wind_direction_deg"], r["rf_bearing"]), axis=1)
    return out


def add_pull_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert venue wind into hitter-specific pull/oppo wind.
      RHB pull field = LF, oppo = RF
      LHB pull field = RF, oppo = LF
      Switch/unknown = use CF as neutral proxy
    """
    if df.empty:
        return df
    out = add_directional_wind_components(df)
    hand = out.get("batter_hand", pd.Series("U", index=out.index)).fillna("U").astype(str).str.upper()
    out["pull_wind_mph"] = np.select(
        [hand.eq("R"), hand.eq("L")],
        [out["wind_to_lf_mph"], out["wind_to_rf_mph"]],
        default=out["wind_to_cf_mph"],
    )
    out["oppo_wind_mph"] = np.select(
        [hand.eq("R"), hand.eq("L")],
        [out["wind_to_rf_mph"], out["wind_to_lf_mph"]],
        default=out["wind_to_cf_mph"],
    )
    out["cross_wind_abs_mph"] = np.sqrt(np.maximum(out["wind_speed_mph"] ** 2 - out["wind_to_cf_mph"] ** 2, 0))
    out["wind_out_to_pull_flag"] = (out["pull_wind_mph"] >= 3.0).astype(int)
    out.loc[out.get("is_roofed_no_wind", 0).eq(1), "wind_out_to_pull_flag"] = 0

    # Game-level flag: is the venue weather blowing out toward CF?
    # For roofed/domed parks this is always 0 because wind is neutralized.
    if "wind_out_flag" in out.columns:
        out["weather_blowing_out"] = pd.to_numeric(out["wind_out_flag"], errors="coerce").fillna(0).astype(int)
    else:
        out["weather_blowing_out"] = (out["wind_to_cf_mph"] >= 3.0).astype(int)
    out.loc[out.get("is_roofed_no_wind", 0).eq(1), ["wind_out_flag", "weather_blowing_out"]] = 0

    # Backward-compatible name; now means hitter pull-side wind on player rows.
    if "batter_hand" in out.columns:
        out["wind_out_flag"] = out["wind_out_to_pull_flag"]
    return out

def dynamic_park_factor(
    base_park_factor: float,
    temp_f: float,
    wind_speed_mph: float,
    wind_direction_deg: float,
    relative_humidity: float,
    out_to_cf_bearing: float,
) -> float:
    """
    Weather-adjust the baseline HR park factor for the daily board.
    Positive adjustments:
      - warmer temperatures
      - wind blowing out
      - modest added humidity
    Negative adjustments:
      - cold temperatures
      - wind blowing in
    """
    base_park_factor = float(base_park_factor if pd.notna(base_park_factor) else DEFAULT_PARK_FACTOR)
    temp_f = float(temp_f if pd.notna(temp_f) else DEFAULT_TEMP_F)
    wind_speed_mph = float(wind_speed_mph if pd.notna(wind_speed_mph) else DEFAULT_WIND_SPEED_MPH)
    wind_direction_deg = float(wind_direction_deg if pd.notna(wind_direction_deg) else DEFAULT_WIND_DIRECTION_DEG)
    relative_humidity = float(relative_humidity if pd.notna(relative_humidity) else DEFAULT_REL_HUMIDITY)

    wind_to = (wind_direction_deg + 180) % 360
    angle = angular_difference(wind_to, out_to_cf_bearing)

    # Directional wind scalar: +1 blowing out, -1 blowing in, ~0 crosswind
    wind_alignment = np.cos(np.deg2rad(angle))

    temp_adj = 0.0015 * (temp_f - 70.0)
    wind_adj = 0.0040 * wind_speed_mph * wind_alignment
    humidity_adj = 0.0005 * (relative_humidity - 50.0) / 10.0

    adjusted = base_park_factor + temp_adj + wind_adj + humidity_adj
    return float(np.clip(adjusted, DYNAMIC_PARK_FACTOR_MIN, DYNAMIC_PARK_FACTOR_MAX))


def map_pitch_group(pitch_type: str) -> str:
    pitch_map = {
        "FF": "FF",
        "FA": "FF",
        "SI": "SI",
        "FT": "SI",
        "FC": "FC",
        "SL": "SL",
        "CH": "CH",
        "FS": "FS",
        "CU": "CU",
        "KC": "CU",
        "SV": "CU",
    }
    return pitch_map.get(pitch_type, "OTHER")


def rate(numer, denom):
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    out = np.where((denom > 0) & np.isfinite(denom), numer / denom, np.nan)
    return out


def trailing_mean(series: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(level=0)
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


# =========================================================
# 1) LOAD STATCAST PA DATA
# =========================================================
def get_training_cache_path(start_date: str, end_date: str) -> str:
    return cache_path(f"statcast_pa_batted_ball_{start_date}_to_{end_date}.parquet")


def _find_best_statcast_cache(start_date: str, end_date: str) -> Optional[str]:
    """Find a compatible existing Statcast cache so only missing dates need to be appended."""
    ensure_cache_dir()
    requested_start = pd.to_datetime(start_date)
    candidates = []

    for fname in os.listdir(CACHE_DIR):
        if not (fname.startswith("statcast_pa") and fname.endswith(".parquet") and "_to_" in fname):
            continue
        stem = fname.replace(".parquet", "")
        try:
            date_part = stem.split("statcast_pa_batted_ball_", 1)[-1]
            if date_part == stem:
                date_part = stem.split("statcast_pa_", 1)[-1]
            cached_start_str, cached_end_str = date_part.split("_to_", 1)
            cached_start = pd.to_datetime(cached_start_str)
            cached_end = pd.to_datetime(cached_end_str)
        except Exception:
            continue
        if cached_start <= requested_start and cached_end >= requested_start:
            candidates.append((cached_end, os.path.join(CACHE_DIR, fname)))

    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]


def _process_raw_statcast_to_pa(df: pd.DataFrame) -> pd.DataFrame:
    pa_df = df[df["events"].notna()].copy()
    print(f"Rows after plate-appearance events filter: {len(pa_df)}")

    needed_cols = [
        "game_date",
        "game_pk",
        "batter",
        "pitcher",
        "events",
        "launch_speed",
        "launch_angle",
        "stand",
        "p_throws",
        "pitch_type",
        "bb_type",
        "home_team",
        "away_team",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
    ]

    missing_cols = [c for c in needed_cols if c not in pa_df.columns]
    if missing_cols:
        raise ValueError(f"Statcast pull is missing required columns: {missing_cols}")

    pa_df = pa_df[needed_cols].copy()

    pa_df["game_date"] = pd.to_datetime(pa_df["game_date"])
    pa_df["batter"] = pd.to_numeric(pa_df["batter"], errors="coerce")
    pa_df["pitcher"] = pd.to_numeric(pa_df["pitcher"], errors="coerce")
    pa_df = pa_df.dropna(subset=["batter", "pitcher", "game_pk"]).copy()
    pa_df["batter"] = pa_df["batter"].astype(int)
    pa_df["pitcher"] = pa_df["pitcher"].astype(int)
    pa_df["game_pk"] = pd.to_numeric(pa_df["game_pk"], errors="coerce").astype(int)

    pa_df["pitch_type"] = pa_df["pitch_type"].fillna("UNK")
    pa_df["pitch_group"] = pa_df["pitch_type"].apply(map_pitch_group)

    pa_df["barrel"] = (
        (pa_df["launch_speed"].fillna(0) >= 98)
        & (pa_df["launch_angle"].fillna(-999).between(20, 35))
    ).astype(int)
    pa_df["hard_hit"] = (pa_df["launch_speed"].fillna(0) >= 95).astype(int)
    pa_df["home_run"] = (pa_df["events"] == "home_run").astype(int)
    pa_df["is_top"] = (pa_df["inning_topbot"] == "Top").astype(int)
    pa_df["batted_ball"] = pa_df["bb_type"].notna().astype(int)
    pa_df["fly_ball"] = (pa_df["bb_type"] == "fly_ball").astype(int)
    pa_df["ground_ball"] = (pa_df["bb_type"] == "ground_ball").astype(int)
    return pa_df


def _pull_statcast_pa_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Pulling Statcast PA data for {start_date} -> {end_date} ...")
    df = statcast(start_dt=start_date, end_dt=end_date)
    print(f"Raw Statcast rows pulled: {len(df)}")
    return _process_raw_statcast_to_pa(df)


def _dedupe_pa_cache(pa_df: pd.DataFrame) -> pd.DataFrame:
    if pa_df.empty:
        return pa_df
    out = pa_df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    dedupe_cols = ["game_pk", "batter", "pitcher", "at_bat_number", "pitch_number", "events"]
    existing_cols = [c for c in dedupe_cols if c in out.columns]
    out = out.drop_duplicates(subset=existing_cols, keep="last")
    out = out.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)
    return out


def load_statcast_pa(start_date: str, end_date: str, use_cache: bool = True, refresh_cache: bool = False) -> pd.DataFrame:
    global STATCAST_CACHE_UPDATED
    STATCAST_CACHE_UPDATED = False

    path = get_training_cache_path(start_date, end_date)
    requested_start = pd.to_datetime(start_date)
    requested_end = pd.to_datetime(end_date)

    def append_missing_dates(pa_df: pd.DataFrame) -> pd.DataFrame:
        global STATCAST_CACHE_UPDATED
        if pa_df.empty:
            return pa_df
        pa_df["game_date"] = pd.to_datetime(pa_df["game_date"], errors="coerce")
        cached_max_date = pa_df["game_date"].max()
        if (
            INCREMENTAL_STATCAST_CACHE_UPDATE
            and pd.notna(cached_max_date)
            and cached_max_date.normalize() < requested_end.normalize()
        ):
            append_start = (cached_max_date.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            append_end = requested_end.strftime("%Y-%m-%d")
            new_pa = _pull_statcast_pa_chunk(append_start, append_end)
            if not new_pa.empty:
                pa_df = _dedupe_pa_cache(pd.concat([pa_df, new_pa], ignore_index=True))
                STATCAST_CACHE_UPDATED = True
                print(f"Appended Statcast PA rows for {append_start} -> {append_end}.")
            else:
                print("No new Statcast PA rows found to append.")
        return pa_df

    if use_cache and not refresh_cache and os.path.exists(path):
        print(f"Loading cached Statcast PA data from {path} ...")
        pa_df = pd.read_parquet(path)
        pa_df = append_missing_dates(pa_df)
        if STATCAST_CACHE_UPDATED:
            pa_df.to_parquet(path, index=False)
            print(f"Updated Statcast cache saved to {path}")
        print(f"Cached rows loaded: {len(pa_df)}")
        return pa_df

    if use_cache and not refresh_cache and INCREMENTAL_STATCAST_CACHE_UPDATE:
        best_cache_path = _find_best_statcast_cache(start_date, end_date)
        if best_cache_path is not None:
            print(f"Found compatible Statcast cache for incremental update: {best_cache_path}")
            pa_df = pd.read_parquet(best_cache_path)
            pa_df["game_date"] = pd.to_datetime(pa_df["game_date"], errors="coerce")
            pa_df = pa_df[(pa_df["game_date"] >= requested_start) & (pa_df["game_date"] <= requested_end)].copy()
            pa_df = append_missing_dates(pa_df)
            pa_df = _dedupe_pa_cache(pa_df)
            pa_df.to_parquet(path, index=False)
            STATCAST_CACHE_UPDATED = True
            print(f"Saved updated Statcast cache to {path}")
            print(f"Cached rows loaded: {len(pa_df)}")
            return pa_df

    pa_df = _pull_statcast_pa_chunk(start_date, end_date)

    if use_cache:
        pa_df.to_parquet(path, index=False)
        STATCAST_CACHE_UPDATED = True
        print(f"Saved Statcast cache to {path}")

    return pa_df


# =========================================================
# 2) BUILD HISTORICAL DATASETS
# =========================================================
def build_batter_game_dataset(pa_df: pd.DataFrame) -> pd.DataFrame:
    print("Building batter-game dataset...")

    sort_cols = ["game_date", "game_pk", "batter", "at_bat_number", "pitch_number"]
    temp = pa_df.sort_values(sort_cols).copy()

    first_pa = (
        temp.groupby(["game_date", "game_pk", "batter"], as_index=False)
        .first()[[
            "game_date", "game_pk", "batter", "pitcher", "stand", "p_throws",
            "home_team", "away_team", "inning_topbot"
        ]]
        .rename(columns={
            "pitcher": "starter_pitcher",
            "stand": "batter_hand",
            "p_throws": "starter_pitcher_hand",
        })
    )

    batter_games = (
        pa_df.groupby(["game_date", "game_pk", "batter"], as_index=False)
        .agg(
            pa=("events", "count"),
            hr_count=("home_run", "sum"),
            barrels=("barrel", "sum"),
            hard_hits=("hard_hit", "sum"),
            avg_ev=("launch_speed", "mean"),
            unique_pitchers_faced=("pitcher", "nunique"),
        )
    )

    batter_games = batter_games.merge(
        first_pa,
        on=["game_date", "game_pk", "batter"],
        how="left",
    )

    batter_games["home_run_game"] = (batter_games["hr_count"] > 0).astype(int)
    batter_games["avg_ev"] = batter_games["avg_ev"].fillna(0)

    batter_games["batting_team"] = np.where(
        batter_games["inning_topbot"] == "Top",
        batter_games["away_team"],
        batter_games["home_team"],
    )
    batter_games["fielding_team"] = np.where(
        batter_games["inning_topbot"] == "Top",
        batter_games["home_team"],
        batter_games["away_team"],
    )
    batter_games["is_home_batter"] = (batter_games["batting_team"] == batter_games["home_team"]).astype(int)

    batter_games["starter_pitcher"] = pd.to_numeric(batter_games["starter_pitcher"], errors="coerce")
    batter_games = batter_games.dropna(subset=["starter_pitcher"]).copy()
    batter_games["starter_pitcher"] = batter_games["starter_pitcher"].astype(int)

    batter_lookup = safe_name_lookup(batter_games["batter"], "batter", "batter_name")
    pitcher_lookup = safe_name_lookup(batter_games["starter_pitcher"], "starter_pitcher", "pitcher_name")

    batter_games = batter_games.merge(batter_lookup, on="batter", how="left")
    batter_games = batter_games.merge(pitcher_lookup, on="starter_pitcher", how="left")

    batter_games["batter_name"] = batter_games["batter_name"].fillna("Unknown Batter")
    batter_games["pitcher_name"] = batter_games["pitcher_name"].fillna("Unknown Pitcher")
    batter_games["batter_name_norm"] = batter_games["batter_name"].apply(normalize_name)

    batter_games = batter_games.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
    return batter_games


def build_pitcher_game_dataset(pa_df: pd.DataFrame) -> pd.DataFrame:
    print("Building pitcher-game dataset...")

    pitcher_games = (
        pa_df.groupby(["game_date", "game_pk", "pitcher"], as_index=False)
        .agg(
            pitcher_pa=("events", "count"),
            hr_allowed=("home_run", "sum"),
            barrels_allowed=("barrel", "sum"),
            hard_hits_allowed=("hard_hit", "sum"),
            avg_ev_allowed=("launch_speed", "mean"),
            batters_faced=("batter", "nunique"),
            strikeouts=("events", lambda x: x.isin(["strikeout", "strikeout_double_play"]).sum()),
            batted_balls_allowed=("batted_ball", "sum"),
            fly_balls_allowed=("fly_ball", "sum"),
            ground_balls_allowed=("ground_ball", "sum"),
        )
    )

    hand_map = (
        pa_df.groupby(["game_date", "game_pk", "pitcher"], as_index=False)["p_throws"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "U")
        .rename(columns={"p_throws": "pitcher_hand"})
    )

    pitcher_games = pitcher_games.merge(hand_map, on=["game_date", "game_pk", "pitcher"], how="left")
    pitcher_games["avg_ev_allowed"] = pitcher_games["avg_ev_allowed"].fillna(0)
    pitcher_games["hr_allowed_game"] = (pitcher_games["hr_allowed"] > 0).astype(int)

    pitcher_games = pitcher_games.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    return pitcher_games


def build_current_season_pitcher_stats(pa_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Build official season-to-date pitcher context for the slate table.

    Uses MLB Stats API /stats endpoint, which is more reliable for pulling all
    qualified/current pitching stat rows than hydrating every player record.
    Returns:
      - pitcher
      - season_era
      - season_hr_allowed
    """
    target_ts = pd.to_datetime(target_date, errors="coerce")
    season = int(target_ts.year) if pd.notna(target_ts) else pd.Timestamp.today().year

    rows = []
    offset = 0
    limit = 500

    while True:
        payload = get_json(
            "https://statsapi.mlb.com/api/v1/stats",
            params={
                "stats": "season",
                "group": "pitching",
                "playerPool": "ALL",
                "season": season,
                "limit": limit,
                "offset": offset,
            },
        )

        if not payload:
            break

        splits = payload.get("stats", [{}])[0].get("splits", [])
        if not splits:
            break

        for split in splits:
            player = split.get("player", {})
            stat = split.get("stat", {})
            pitcher_id = player.get("id")

            if pitcher_id is None:
                continue

            era = pd.to_numeric(stat.get("era", np.nan), errors="coerce")
            hr_allowed = pd.to_numeric(
                stat.get("homeRuns", stat.get("homeRunsAllowed", np.nan)),
                errors="coerce",
            )

            rows.append({
                "pitcher": int(pitcher_id),
                "season_era": era,
                "season_hr_allowed": hr_allowed,
            })

        # Stop if the API returned fewer rows than requested.
        if len(splits) < limit:
            break

        offset += limit

    out = pd.DataFrame(rows)
    if out.empty:
        print("⚠️ Official pitcher season stats returned no rows; ERA/HR columns will be blank.")
        return pd.DataFrame(columns=["pitcher", "season_hr_allowed", "season_era"])

    out = out.drop_duplicates("pitcher", keep="first").copy()
    out["season_era"] = pd.to_numeric(out["season_era"], errors="coerce").round(2)
    out["season_hr_allowed"] = pd.to_numeric(out["season_hr_allowed"], errors="coerce").fillna(0).astype(int)

    print(f"Loaded official {season} pitcher season stats rows: {len(out):,}")
    return out[["pitcher", "season_hr_allowed", "season_era"]]


def build_matchup_game_history(pa_df: pd.DataFrame) -> pd.DataFrame:
    print("Building batter-vs-pitcher matchup history...")

    matchup_games = (
        pa_df.groupby(["game_date", "game_pk", "batter", "pitcher"], as_index=False)
        .agg(
            matchup_pa_game=("events", "count"),
            matchup_hr_game=("home_run", "sum"),
            matchup_barrels_game=("barrel", "sum"),
            matchup_hard_hits_game=("hard_hit", "sum"),
            matchup_avg_ev_game=("launch_speed", "mean"),
        )
    )

    matchup_games["matchup_avg_ev_game"] = matchup_games["matchup_avg_ev_game"].fillna(0)
    matchup_games["matchup_hr_game_flag"] = (matchup_games["matchup_hr_game"] > 0).astype(int)
    matchup_games = matchup_games.sort_values(["batter", "pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    return matchup_games


def add_matchup_pregame_features(matchup_games: pd.DataFrame) -> pd.DataFrame:
    df = matchup_games.sort_values(["batter", "pitcher", "game_date", "game_pk"]).copy()
    g = df.groupby(["batter", "pitcher"], sort=False)

    df["matchup_games_prior"] = g.cumcount()
    df["matchup_pa_prior"] = g["matchup_pa_game"].cumsum() - df["matchup_pa_game"]
    df["matchup_hr_prior"] = g["matchup_hr_game"].cumsum() - df["matchup_hr_game"]
    df["matchup_barrels_prior"] = g["matchup_barrels_game"].cumsum() - df["matchup_barrels_game"]
    df["matchup_hard_hits_prior"] = g["matchup_hard_hits_game"].cumsum() - df["matchup_hard_hits_game"]

    df["_matchup_ev_sum_game"] = df["matchup_avg_ev_game"] * df["matchup_pa_game"]
    df["matchup_ev_sum_prior"] = g["_matchup_ev_sum_game"].cumsum() - df["_matchup_ev_sum_game"]

    df["matchup_hr_rate_prior"] = rate(df["matchup_hr_prior"], df["matchup_pa_prior"])
    df["matchup_barrel_rate_prior"] = rate(df["matchup_barrels_prior"], df["matchup_pa_prior"])
    df["matchup_hard_hit_rate_prior"] = rate(df["matchup_hard_hits_prior"], df["matchup_pa_prior"])
    df["matchup_avg_ev_prior"] = rate(df["matchup_ev_sum_prior"], df["matchup_pa_prior"])

    df["_matchup_hr_game_shift"] = g["matchup_hr_game_flag"].shift(1)
    df["_matchup_barrel_rate_game_shift"] = rate(df["matchup_barrels_game"], df["matchup_pa_game"])
    df["_matchup_barrel_rate_game_shift"] = g["_matchup_barrel_rate_game_shift"].shift(1)
    df["_matchup_avg_ev_game_shift"] = g["matchup_avg_ev_game"].shift(1)
    df["_matchup_pa_game_shift"] = g["matchup_pa_game"].shift(1)

    idx = pd.MultiIndex.from_arrays([
        pd.Series(list(zip(df["batter"], df["pitcher"]))).astype(str),
        np.arange(len(df))
    ])
    for col_name, source, window in [
        ("matchup_recent_hr_rate_3", "_matchup_hr_game_shift", 3),
        ("matchup_recent_hr_rate_5", "_matchup_hr_game_shift", 5),
        ("matchup_recent_barrel_rate_3", "_matchup_barrel_rate_game_shift", 3),
        ("matchup_recent_avg_ev_3", "_matchup_avg_ev_game_shift", 3),
        ("matchup_recent_pa_3", "_matchup_pa_game_shift", 3),
    ]:
        s = pd.Series(df[source].values, index=idx)
        df[col_name] = trailing_mean(s, window).values

    df["matchup_history_score_prior"] = (
        0.35 * df["matchup_hr_rate_prior"].fillna(0)
        + 0.20 * df["matchup_barrel_rate_prior"].fillna(0)
        + 0.10 * df["matchup_hard_hit_rate_prior"].fillna(0)
        + 0.10 * (df["matchup_avg_ev_prior"].fillna(0) / 100.0)
        + 0.15 * df["matchup_recent_hr_rate_3"].fillna(0)
        + 0.10 * df["matchup_recent_barrel_rate_3"].fillna(0)
    )

    drop_cols = [
        c for c in df.columns
        if c.startswith("_matchup_")
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


# =========================================================
# 3) PREGAME FEATURE ENGINEERING (NO LEAKAGE)
# =========================================================
def add_batter_pregame_features(batter_games: pd.DataFrame) -> pd.DataFrame:
    df = batter_games.sort_values(["batter", "game_date", "game_pk"]).copy()

    g = df.groupby("batter", sort=False)

    df["batter_games_prior"] = g.cumcount()
    df["batter_pa_prior"] = g["pa"].cumsum() - df["pa"]
    df["batter_hr_prior"] = g["hr_count"].cumsum() - df["hr_count"]
    df["batter_barrels_prior"] = g["barrels"].cumsum() - df["barrels"]
    df["batter_hard_hits_prior"] = g["hard_hits"].cumsum() - df["hard_hits"]

    ev_pa = df["avg_ev"] * df["pa"]
    df["batter_ev_sum_prior"] = g[ev_pa.name if ev_pa.name else "avg_ev"].transform(lambda s: 0)
    df["_batter_ev_sum_game"] = ev_pa
    df["batter_ev_sum_prior"] = g["_batter_ev_sum_game"].cumsum() - df["_batter_ev_sum_game"]

    df["batter_hr_rate_prior"] = rate(df["batter_hr_prior"], df["batter_pa_prior"])
    df["batter_barrel_rate_prior"] = rate(df["batter_barrels_prior"], df["batter_pa_prior"])
    df["batter_hard_hit_rate_prior"] = rate(df["batter_hard_hits_prior"], df["batter_pa_prior"])
    df["batter_avg_ev_prior"] = rate(df["batter_ev_sum_prior"], df["batter_pa_prior"])

    df["_hr_game_shift"] = g["home_run_game"].shift(1)
    df["_barrel_rate_game_shift"] = (df["barrels"] / df["pa"].replace(0, np.nan))
    df["_barrel_rate_game_shift"] = g["_barrel_rate_game_shift"].shift(1)
    df["_hard_hit_rate_game_shift"] = (df["hard_hits"] / df["pa"].replace(0, np.nan))
    df["_hard_hit_rate_game_shift"] = g["_hard_hit_rate_game_shift"].shift(1)
    df["_avg_ev_game_shift"] = g["avg_ev"].shift(1)
    df["_pa_game_shift"] = g["pa"].shift(1)

    idx = pd.MultiIndex.from_arrays([df["batter"], np.arange(len(df))])
    for col_name, source, window in [
        ("batter_recent_hr_rate_10", "_hr_game_shift", 10),
        ("batter_recent_hr_rate_20", "_hr_game_shift", 20),
        ("batter_recent_barrel_rate_10", "_barrel_rate_game_shift", 10),
        ("batter_recent_hard_hit_rate_10", "_hard_hit_rate_game_shift", 10),
        ("batter_recent_avg_ev_10", "_avg_ev_game_shift", 10),
        ("batter_recent_pa_10", "_pa_game_shift", 10),
    ]:
        s = pd.Series(df[source].values, index=idx)
        df[col_name] = trailing_mean(s, window).values

    df["batter_power_score_prior"] = (
        0.50 * df["batter_barrel_rate_prior"].fillna(0)
        + 0.30 * df["batter_hard_hit_rate_prior"].fillna(0)
        + 0.20 * (df["batter_avg_ev_prior"].fillna(0) / 100.0)
    )

    drop_cols = [c for c in df.columns if c.startswith("_batter_") or c.startswith("_hr_") or c.startswith("_barrel_") or c.startswith("_hard_hit_") or c.startswith("_avg_ev_") or c.startswith("_pa_")]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def add_pitcher_pregame_features(pitcher_games: pd.DataFrame) -> pd.DataFrame:
    df = pitcher_games.sort_values(["pitcher", "game_date", "game_pk"]).copy()

    g = df.groupby("pitcher", sort=False)

    df["pitcher_games_prior"] = g.cumcount()
    df["pitcher_pa_prior"] = g["pitcher_pa"].cumsum() - df["pitcher_pa"]
    df["pitcher_hr_allowed_prior"] = g["hr_allowed"].cumsum() - df["hr_allowed"]
    df["pitcher_barrels_allowed_prior"] = g["barrels_allowed"].cumsum() - df["barrels_allowed"]
    df["pitcher_hard_hits_allowed_prior"] = g["hard_hits_allowed"].cumsum() - df["hard_hits_allowed"]
    df["pitcher_k_prior"] = g["strikeouts"].cumsum() - df["strikeouts"]
    df["pitcher_bbe_allowed_prior"] = g["batted_balls_allowed"].cumsum() - df["batted_balls_allowed"]
    df["pitcher_fb_allowed_prior"] = g["fly_balls_allowed"].cumsum() - df["fly_balls_allowed"]
    df["pitcher_gb_allowed_prior"] = g["ground_balls_allowed"].cumsum() - df["ground_balls_allowed"]

    df["_pitcher_ev_sum_game"] = df["avg_ev_allowed"] * df["pitcher_pa"]
    df["pitcher_ev_sum_prior"] = g["_pitcher_ev_sum_game"].cumsum() - df["_pitcher_ev_sum_game"]

    df["pitcher_hr_rate_allowed_prior"] = rate(df["pitcher_hr_allowed_prior"], df["pitcher_pa_prior"])
    df["pitcher_barrel_rate_allowed_prior"] = rate(df["pitcher_barrels_allowed_prior"], df["pitcher_pa_prior"])
    df["pitcher_hard_hit_rate_allowed_prior"] = rate(df["pitcher_hard_hits_allowed_prior"], df["pitcher_pa_prior"])
    df["pitcher_avg_ev_allowed_prior"] = rate(df["pitcher_ev_sum_prior"], df["pitcher_pa_prior"])
    df["pitcher_k_rate_prior"] = rate(df["pitcher_k_prior"], df["pitcher_pa_prior"])
    df["pitcher_fb_rate_allowed_prior"] = rate(df["pitcher_fb_allowed_prior"], df["pitcher_bbe_allowed_prior"])
    df["pitcher_gb_rate_allowed_prior"] = rate(df["pitcher_gb_allowed_prior"], df["pitcher_bbe_allowed_prior"])

    df["_hr_allowed_game_shift"] = g["hr_allowed_game"].shift(1)
    df["_barrel_allowed_rate_game_shift"] = (df["barrels_allowed"] / df["pitcher_pa"].replace(0, np.nan))
    df["_barrel_allowed_rate_game_shift"] = g["_barrel_allowed_rate_game_shift"].shift(1)
    df["_avg_ev_allowed_game_shift"] = g["avg_ev_allowed"].shift(1)
    df["_k_rate_game_shift"] = (df["strikeouts"] / df["pitcher_pa"].replace(0, np.nan))
    df["_k_rate_game_shift"] = g["_k_rate_game_shift"].shift(1)
    df["_fb_rate_game_shift"] = (df["fly_balls_allowed"] / df["batted_balls_allowed"].replace(0, np.nan))
    df["_fb_rate_game_shift"] = g["_fb_rate_game_shift"].shift(1)
    df["_gb_rate_game_shift"] = (df["ground_balls_allowed"] / df["batted_balls_allowed"].replace(0, np.nan))
    df["_gb_rate_game_shift"] = g["_gb_rate_game_shift"].shift(1)

    idx = pd.MultiIndex.from_arrays([df["pitcher"], np.arange(len(df))])
    for col_name, source, window in [
        ("pitcher_recent_hr_allowed_rate_10", "_hr_allowed_game_shift", 10),
        ("pitcher_recent_hr_allowed_rate_20", "_hr_allowed_game_shift", 20),
        ("pitcher_recent_barrel_allowed_rate_10", "_barrel_allowed_rate_game_shift", 10),
        ("pitcher_recent_avg_ev_allowed_10", "_avg_ev_allowed_game_shift", 10),
        ("pitcher_recent_k_rate_5", "_k_rate_game_shift", 5),
        ("pitcher_recent_k_rate_10", "_k_rate_game_shift", 10),
        ("pitcher_recent_fb_rate_allowed_5", "_fb_rate_game_shift", 5),
        ("pitcher_recent_fb_rate_allowed_10", "_fb_rate_game_shift", 10),
        ("pitcher_recent_gb_rate_allowed_5", "_gb_rate_game_shift", 5),
        ("pitcher_recent_gb_rate_allowed_10", "_gb_rate_game_shift", 10),
    ]:
        s = pd.Series(df[source].values, index=idx)
        df[col_name] = trailing_mean(s, window).values

    # Higher K-rate and ground-ball rate should reduce HR opportunity.
    # Higher fly-ball rate should increase HR opportunity because HRs require elevation.
    df["pitcher_damage_score_prior"] = (
        0.34 * df["pitcher_barrel_rate_allowed_prior"].fillna(0)
        + 0.21 * df["pitcher_hard_hit_rate_allowed_prior"].fillna(0)
        + 0.12 * (df["pitcher_avg_ev_allowed_prior"].fillna(0) / 100.0)
        + 0.10 * df["pitcher_fb_rate_allowed_prior"].fillna(0)
        + 0.08 * df["pitcher_recent_fb_rate_allowed_10"].fillna(0)
        - 0.10 * df["pitcher_gb_rate_allowed_prior"].fillna(0)
        - 0.06 * df["pitcher_recent_gb_rate_allowed_10"].fillna(0)
        - 0.09 * df["pitcher_k_rate_prior"].fillna(0)
        - 0.04 * df["pitcher_recent_k_rate_10"].fillna(0)
    )

    drop_cols = [c for c in df.columns if c.startswith("_pitcher_") or c.startswith("_hr_allowed_") or c.startswith("_barrel_allowed_") or c.startswith("_avg_ev_allowed_") or c.startswith("_k_rate_") or c.startswith("_fb_rate_") or c.startswith("_gb_rate_")]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def build_batter_hand_history(pa_df: pd.DataFrame) -> pd.DataFrame:
    hand_games = (
        pa_df.groupby(["game_date", "game_pk", "batter", "p_throws"], as_index=False)
        .agg(
            pa_vs_hand=("events", "count"),
            hr_vs_hand=("home_run", "sum"),
            barrels_vs_hand=("barrel", "sum"),
            hard_hits_vs_hand=("hard_hit", "sum"),
            avg_ev_vs_hand=("launch_speed", "mean"),
        )
        .rename(columns={"p_throws": "pitcher_hand"})
    )
    hand_games["avg_ev_vs_hand"] = hand_games["avg_ev_vs_hand"].fillna(0)

    hand_games = hand_games.sort_values(["batter", "pitcher_hand", "game_date", "game_pk"]).copy()
    g = hand_games.groupby(["batter", "pitcher_hand"], sort=False)

    hand_games["batter_pa_vs_hand_prior"] = g["pa_vs_hand"].cumsum() - hand_games["pa_vs_hand"]
    hand_games["batter_hr_vs_hand_prior"] = g["hr_vs_hand"].cumsum() - hand_games["hr_vs_hand"]
    hand_games["batter_barrels_vs_hand_prior"] = g["barrels_vs_hand"].cumsum() - hand_games["barrels_vs_hand"]
    hand_games["batter_hard_hits_vs_hand_prior"] = g["hard_hits_vs_hand"].cumsum() - hand_games["hard_hits_vs_hand"]

    hand_games["_ev_sum_vs_hand_game"] = hand_games["avg_ev_vs_hand"] * hand_games["pa_vs_hand"]
    hand_games["batter_ev_sum_vs_hand_prior"] = g["_ev_sum_vs_hand_game"].cumsum() - hand_games["_ev_sum_vs_hand_game"]

    hand_games["batter_hr_rate_vs_hand_prior"] = rate(hand_games["batter_hr_vs_hand_prior"], hand_games["batter_pa_vs_hand_prior"])
    hand_games["batter_barrel_rate_vs_hand_prior"] = rate(hand_games["batter_barrels_vs_hand_prior"], hand_games["batter_pa_vs_hand_prior"])
    hand_games["batter_hard_hit_rate_vs_hand_prior"] = rate(hand_games["batter_hard_hits_vs_hand_prior"], hand_games["batter_pa_vs_hand_prior"])
    hand_games["batter_avg_ev_vs_hand_prior"] = rate(hand_games["batter_ev_sum_vs_hand_prior"], hand_games["batter_pa_vs_hand_prior"])

    keep_cols = [
        "game_date", "game_pk", "batter", "pitcher_hand",
        "batter_pa_vs_hand_prior", "batter_hr_rate_vs_hand_prior",
        "batter_barrel_rate_vs_hand_prior", "batter_hard_hit_rate_vs_hand_prior",
        "batter_avg_ev_vs_hand_prior"
    ]
    return hand_games[keep_cols].drop_duplicates(["game_date", "game_pk", "batter", "pitcher_hand"])


def build_pitcher_hand_history(pa_df: pd.DataFrame) -> pd.DataFrame:
    hand_games = (
        pa_df.groupby(["game_date", "game_pk", "pitcher", "stand"], as_index=False)
        .agg(
            pa_vs_batter_hand=("events", "count"),
            hr_vs_batter_hand=("home_run", "sum"),
            barrels_vs_batter_hand=("barrel", "sum"),
            hard_hits_vs_batter_hand=("hard_hit", "sum"),
            avg_ev_vs_batter_hand=("launch_speed", "mean"),
        )
        .rename(columns={"stand": "batter_hand"})
    )
    hand_games["avg_ev_vs_batter_hand"] = hand_games["avg_ev_vs_batter_hand"].fillna(0)

    hand_games = hand_games.sort_values(["pitcher", "batter_hand", "game_date", "game_pk"]).copy()
    g = hand_games.groupby(["pitcher", "batter_hand"], sort=False)

    hand_games["pitcher_pa_vs_batter_hand_prior"] = g["pa_vs_batter_hand"].cumsum() - hand_games["pa_vs_batter_hand"]
    hand_games["pitcher_hr_vs_batter_hand_prior"] = g["hr_vs_batter_hand"].cumsum() - hand_games["hr_vs_batter_hand"]
    hand_games["pitcher_barrels_vs_batter_hand_prior"] = g["barrels_vs_batter_hand"].cumsum() - hand_games["barrels_vs_batter_hand"]
    hand_games["pitcher_hard_hits_vs_batter_hand_prior"] = g["hard_hits_vs_batter_hand"].cumsum() - hand_games["hard_hits_vs_batter_hand"]

    hand_games["_ev_sum_vs_batter_hand_game"] = hand_games["avg_ev_vs_batter_hand"] * hand_games["pa_vs_batter_hand"]
    hand_games["pitcher_ev_sum_vs_batter_hand_prior"] = g["_ev_sum_vs_batter_hand_game"].cumsum() - hand_games["_ev_sum_vs_batter_hand_game"]

    hand_games["pitcher_hr_rate_vs_batter_hand_prior"] = rate(hand_games["pitcher_hr_vs_batter_hand_prior"], hand_games["pitcher_pa_vs_batter_hand_prior"])
    hand_games["pitcher_barrel_rate_vs_batter_hand_prior"] = rate(hand_games["pitcher_barrels_vs_batter_hand_prior"], hand_games["pitcher_pa_vs_batter_hand_prior"])
    hand_games["pitcher_hard_hit_rate_vs_batter_hand_prior"] = rate(hand_games["pitcher_hard_hits_vs_batter_hand_prior"], hand_games["pitcher_pa_vs_batter_hand_prior"])
    hand_games["pitcher_avg_ev_vs_batter_hand_prior"] = rate(hand_games["pitcher_ev_sum_vs_batter_hand_prior"], hand_games["pitcher_pa_vs_batter_hand_prior"])

    keep_cols = [
        "game_date", "game_pk", "pitcher", "batter_hand",
        "pitcher_pa_vs_batter_hand_prior", "pitcher_hr_rate_vs_batter_hand_prior",
        "pitcher_barrel_rate_vs_batter_hand_prior", "pitcher_hard_hit_rate_vs_batter_hand_prior",
        "pitcher_avg_ev_vs_batter_hand_prior"
    ]
    return hand_games[keep_cols].drop_duplicates(["game_date", "game_pk", "pitcher", "batter_hand"])


def build_batter_pitch_group_history(pa_df: pd.DataFrame) -> pd.DataFrame:
    bp = (
        pa_df.groupby(["game_date", "game_pk", "batter", "pitch_group"], as_index=False)
        .agg(
            pa_pitch=("events", "count"),
            hr_pitch=("home_run", "sum"),
            barrels_pitch=("barrel", "sum"),
            hard_hits_pitch=("hard_hit", "sum"),
            avg_ev_pitch=("launch_speed", "mean"),
        )
    )
    bp["avg_ev_pitch"] = bp["avg_ev_pitch"].fillna(0)

    bp = bp.sort_values(["batter", "pitch_group", "game_date", "game_pk"]).copy()
    g = bp.groupby(["batter", "pitch_group"], sort=False)

    bp["batter_pa_pitch_prior"] = g["pa_pitch"].cumsum() - bp["pa_pitch"]
    bp["batter_hr_pitch_prior"] = g["hr_pitch"].cumsum() - bp["hr_pitch"]
    bp["batter_barrels_pitch_prior"] = g["barrels_pitch"].cumsum() - bp["barrels_pitch"]
    bp["batter_hard_hits_pitch_prior"] = g["hard_hits_pitch"].cumsum() - bp["hard_hits_pitch"]
    bp["_batter_ev_pitch_sum_game"] = bp["avg_ev_pitch"] * bp["pa_pitch"]
    bp["batter_ev_pitch_sum_prior"] = g["_batter_ev_pitch_sum_game"].cumsum() - bp["_batter_ev_pitch_sum_game"]

    bp["batter_hr_rate_pitch_prior"] = rate(bp["batter_hr_pitch_prior"], bp["batter_pa_pitch_prior"])
    bp["batter_barrel_rate_pitch_prior"] = rate(bp["batter_barrels_pitch_prior"], bp["batter_pa_pitch_prior"])
    bp["batter_hard_hit_rate_pitch_prior"] = rate(bp["batter_hard_hits_pitch_prior"], bp["batter_pa_pitch_prior"])
    bp["batter_avg_ev_pitch_prior"] = rate(bp["batter_ev_pitch_sum_prior"], bp["batter_pa_pitch_prior"])

    bp["batter_pitch_score_prior"] = (
        0.45 * bp["batter_barrel_rate_pitch_prior"].fillna(0)
        + 0.25 * bp["batter_hard_hit_rate_pitch_prior"].fillna(0)
        + 0.20 * bp["batter_hr_rate_pitch_prior"].fillna(0)
        + 0.10 * (bp["batter_avg_ev_pitch_prior"].fillna(0) / 100.0)
    )

    keep = ["game_date", "game_pk", "batter", "pitch_group", "batter_pitch_score_prior"]
    return bp[keep].drop_duplicates(["game_date", "game_pk", "batter", "pitch_group"])


def build_pitcher_pitch_mix_history(pa_df: pd.DataFrame) -> pd.DataFrame:
    pm = (
        pa_df.groupby(["game_date", "game_pk", "pitcher", "pitch_group"], as_index=False)
        .size()
        .rename(columns={"size": "pitch_count"})
    )

    pm = pm.sort_values(["pitcher", "pitch_group", "game_date", "game_pk"]).copy()
    g_pg = pm.groupby(["pitcher", "pitch_group"], sort=False)
    pm["pitch_count_prior_group"] = g_pg["pitch_count"].cumsum() - pm["pitch_count"]

    totals = (
        pm.groupby(["game_date", "game_pk", "pitcher"], as_index=False)["pitch_count"]
        .sum()
        .rename(columns={"pitch_count": "pitch_count_game_total"})
    )

    pm = pm.merge(totals, on=["game_date", "game_pk", "pitcher"], how="left")
    g_p = pm.groupby("pitcher", sort=False)
    pm["pitch_count_prior_total"] = g_p["pitch_count_game_total"].cumsum() - pm["pitch_count_game_total"]
    pm["pitch_usage_prior"] = rate(pm["pitch_count_prior_group"], pm["pitch_count_prior_total"])

    keep = ["game_date", "game_pk", "pitcher", "pitch_group", "pitch_usage_prior"]
    return pm[keep].drop_duplicates(["game_date", "game_pk", "pitcher", "pitch_group"])



def _add_game_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    date_int = pd.to_numeric(out["game_date"].dt.strftime("%Y%m%d"), errors="coerce").fillna(0).astype(np.int64)
    game_pk_int = pd.to_numeric(out["game_pk"], errors="coerce").fillna(0).astype(np.int64)
    out["game_sort_key"] = date_int * np.int64(10_000_000) + game_pk_int
    return out


def build_pitch_fit_for_games(
    batter_games: pd.DataFrame,
    batter_pitch_hist: pd.DataFrame,
    pitcher_pitch_hist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build pitch-fit for each batter/starter matchup by joining:
      - the batter's latest prior skill vs each pitch group
      - the starter's latest prior pitch-group usage

    This version avoids pandas merge_asof grouped-sort issues by doing
    the asof lookup inside each group.
    """
    base = batter_games[["game_date", "game_pk", "batter", "starter_pitcher"]].drop_duplicates().copy()
    base = _add_game_sort_key(base)

    pitch_groups = ["FF", "SI", "FC", "SL", "CH", "FS", "CU", "OTHER"]
    rows = []

    batter_hist = _add_game_sort_key(
        batter_pitch_hist[["game_date", "game_pk", "batter", "pitch_group", "batter_pitch_score_prior"]].copy()
    )
    pitcher_hist = _add_game_sort_key(
        pitcher_pitch_hist.rename(columns={"pitcher": "starter_pitcher"})[
            ["game_date", "game_pk", "starter_pitcher", "pitch_group", "pitch_usage_prior"]
        ].copy()
    )

    def asof_batter_group(left_group: pd.DataFrame, right_group: pd.DataFrame) -> pd.DataFrame:
        left_group = left_group.sort_values("game_sort_key").reset_index(drop=True)
        right_group = right_group.sort_values("game_sort_key").reset_index(drop=True)
        if right_group.empty:
            left_group["batter_pitch_score_prior"] = np.nan
            return left_group
        return pd.merge_asof(
            left_group,
            right_group[["game_sort_key", "batter_pitch_score_prior"]],
            on="game_sort_key",
            direction="backward",
            allow_exact_matches=False,
        )

    def asof_pitcher_group(left_group: pd.DataFrame, right_group: pd.DataFrame) -> pd.DataFrame:
        left_group = left_group.sort_values("game_sort_key").reset_index(drop=True)
        right_group = right_group.sort_values("game_sort_key").reset_index(drop=True)
        if right_group.empty:
            left_group["pitch_usage_prior"] = np.nan
            return left_group
        return pd.merge_asof(
            left_group,
            right_group[["game_sort_key", "pitch_usage_prior"]],
            on="game_sort_key",
            direction="backward",
            allow_exact_matches=False,
        )

    for pg in pitch_groups:
        left = base.copy()
        left["pitch_group"] = pg

        batter_pg = batter_hist[batter_hist["pitch_group"] == pg].copy()
        pitcher_pg = pitcher_hist[pitcher_hist["pitch_group"] == pg].copy()

        batter_chunks = []
        for batter_id, left_group in left.groupby("batter", sort=False):
            right_group = batter_pg[batter_pg["batter"] == batter_id]
            batter_chunks.append(asof_batter_group(left_group, right_group))

        merged = pd.concat(batter_chunks, ignore_index=True) if batter_chunks else left.copy()

        pitcher_chunks = []
        for pitcher_id, left_group in merged.groupby("starter_pitcher", sort=False):
            right_group = pitcher_pg[pitcher_pg["starter_pitcher"] == pitcher_id]
            pitcher_chunks.append(asof_pitcher_group(left_group, right_group))

        merged = pd.concat(pitcher_chunks, ignore_index=True) if pitcher_chunks else merged.copy()

        merged["batter_pitch_score_prior"] = pd.to_numeric(
            merged["batter_pitch_score_prior"], errors="coerce"
        ).fillna(0)
        merged["pitch_usage_prior"] = pd.to_numeric(
            merged["pitch_usage_prior"], errors="coerce"
        ).fillna(0)
        merged["weighted_fit"] = merged["batter_pitch_score_prior"] * merged["pitch_usage_prior"]

        rows.append(
            merged[["game_date", "game_pk", "batter", "starter_pitcher", "weighted_fit", "pitch_usage_prior"]]
        )

    all_fit = pd.concat(rows, ignore_index=True)
    fit = (
        all_fit.groupby(["game_date", "game_pk", "batter", "starter_pitcher"], as_index=False)
        .agg(
            pitch_fit_score_prior=("weighted_fit", "sum"),
            pitch_fit_coverage_prior=("pitch_usage_prior", "sum"),
        )
    )
    return fit


def build_latest_batter_pitch_group_snapshot(pa_df: pd.DataFrame) -> pd.DataFrame:
    batter_pitch_hist = build_batter_pitch_group_history(pa_df)
    latest = (
        batter_pitch_hist.sort_values(["batter", "pitch_group", "game_date", "game_pk"])
        .groupby(["batter", "pitch_group"], as_index=False)
        .tail(1)
        .copy()
    )
    return latest[["batter", "pitch_group", "batter_pitch_score_prior"]].drop_duplicates(["batter", "pitch_group"])


def build_latest_pitcher_pitch_group_snapshot(pa_df: pd.DataFrame) -> pd.DataFrame:
    pitcher_pitch_hist = build_pitcher_pitch_mix_history(pa_df)
    latest = (
        pitcher_pitch_hist.sort_values(["pitcher", "pitch_group", "game_date", "game_pk"])
        .groupby(["pitcher", "pitch_group"], as_index=False)
        .tail(1)
        .copy()
    )
    return latest[["pitcher", "pitch_group", "pitch_usage_prior"]].drop_duplicates(["pitcher", "pitch_group"])


def compute_forward_pitch_fit(
    board: pd.DataFrame,
    batter_pitch_snapshot: pd.DataFrame,
    pitcher_pitch_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    if board.empty:
        return board

    pitch_groups = ["FF", "SI", "FC", "SL", "CH", "FS", "CU", "OTHER"]
    rows = []

    base = board.copy()
    for pg in pitch_groups:
        left = base[["batter", "pitcher"]].copy()
        left["pitch_group"] = pg

        merged = left.merge(
            batter_pitch_snapshot,
            on=["batter", "pitch_group"],
            how="left",
        ).merge(
            pitcher_pitch_snapshot,
            on=["pitcher", "pitch_group"],
            how="left",
        )

        merged["batter_pitch_score_prior"] = pd.to_numeric(
            merged["batter_pitch_score_prior"], errors="coerce"
        ).fillna(0)
        merged["pitch_usage_prior"] = pd.to_numeric(
            merged["pitch_usage_prior"], errors="coerce"
        ).fillna(0)
        merged["weighted_fit"] = merged["batter_pitch_score_prior"] * merged["pitch_usage_prior"]
        rows.append(merged[["batter", "pitcher", "weighted_fit", "pitch_usage_prior"]])

    all_fit = pd.concat(rows, ignore_index=True)
    fit = (
        all_fit.groupby(["batter", "pitcher"], as_index=False)
        .agg(
            pitch_fit_score_prior=("weighted_fit", "sum"),
            pitch_fit_coverage_prior=("pitch_usage_prior", "sum"),
        )
    )

    out = board.drop(columns=["pitch_fit_score_prior", "pitch_fit_coverage_prior"], errors="ignore").merge(
        fit,
        on=["batter", "pitcher"],
        how="left",
    )
    out["pitch_fit_score_prior"] = pd.to_numeric(out["pitch_fit_score_prior"], errors="coerce").fillna(0)
    out["pitch_fit_coverage_prior"] = pd.to_numeric(out["pitch_fit_coverage_prior"], errors="coerce").fillna(0)
    return out


def _build_model_dataset_full(pa_df: pd.DataFrame, force_rebuild: bool = False, cache_suffix: str = "") -> pd.DataFrame:
    dataset_path = cache_path(
        f"model_dataset_{MODEL_DATASET_CACHE_VERSION}_{FULL_DATA_START_DATE}_to_{FULL_DATA_END_DATE}{cache_suffix}.parquet"
    )
    if (not force_rebuild) and USE_CACHE and not REFRESH_CACHE and not STATCAST_CACHE_UPDATED and os.path.exists(dataset_path):
        print(f"Loading cached model dataset from {dataset_path} ...")
        cached = pd.read_parquet(dataset_path)
        cached = add_pull_wind_features(cached)
        for c in ["pull_wind_mph", "oppo_wind_mph", "cross_wind_abs_mph", "wind_out_to_pull_flag"]:
            if c not in cached.columns:
                cached[c] = 0
        return cached

    batter_games = build_batter_game_dataset(pa_df)
    pitcher_games = build_pitcher_game_dataset(pa_df)

    batter_games = add_batter_pregame_features(batter_games)
    pitcher_games = add_pitcher_pregame_features(pitcher_games)

    matchup_games = build_matchup_game_history(pa_df)
    matchup_games = add_matchup_pregame_features(matchup_games)

    batter_hand_hist = build_batter_hand_history(pa_df)
    pitcher_hand_hist = build_pitcher_hand_history(pa_df)
    batter_pitch_hist = build_batter_pitch_group_history(pa_df)
    pitcher_pitch_hist = build_pitcher_pitch_mix_history(pa_df)
    pitch_fit = build_pitch_fit_for_games(batter_games, batter_pitch_hist, pitcher_pitch_hist)

    df = batter_games.merge(
        pitcher_games[[
            "game_date", "game_pk", "pitcher",
            "pitcher_games_prior", "pitcher_pa_prior", "pitcher_hr_allowed_prior",
            "pitcher_barrel_rate_allowed_prior", "pitcher_hard_hit_rate_allowed_prior",
            "pitcher_avg_ev_allowed_prior", "pitcher_hr_rate_allowed_prior",
            "pitcher_k_rate_prior", "pitcher_recent_k_rate_5", "pitcher_recent_k_rate_10",
            "pitcher_fb_rate_allowed_prior", "pitcher_gb_rate_allowed_prior",
            "pitcher_recent_fb_rate_allowed_5", "pitcher_recent_fb_rate_allowed_10",
            "pitcher_recent_gb_rate_allowed_5", "pitcher_recent_gb_rate_allowed_10",
            "pitcher_recent_hr_allowed_rate_10", "pitcher_recent_hr_allowed_rate_20",
            "pitcher_recent_barrel_allowed_rate_10", "pitcher_recent_avg_ev_allowed_10",
            "pitcher_damage_score_prior", "pitcher_hand"
        ]],
        left_on=["game_date", "game_pk", "starter_pitcher"],
        right_on=["game_date", "game_pk", "pitcher"],
        how="left"
    )

    df = df.merge(
        batter_hand_hist,
        left_on=["game_date", "game_pk", "batter", "starter_pitcher_hand"],
        right_on=["game_date", "game_pk", "batter", "pitcher_hand"],
        how="left",
        suffixes=("", "_drop1")
    )
    df = df.drop(columns=[c for c in df.columns if c.endswith("_drop1")], errors="ignore")

    df = df.merge(
        pitcher_hand_hist,
        left_on=["game_date", "game_pk", "starter_pitcher", "batter_hand"],
        right_on=["game_date", "game_pk", "pitcher", "batter_hand"],
        how="left",
        suffixes=("", "_drop2")
    )
    df = df.drop(columns=[c for c in df.columns if c.endswith("_drop2")], errors="ignore")

    df = df.merge(
        pitch_fit,
        on=["game_date", "game_pk", "batter", "starter_pitcher"],
        how="left"
    )

    df = df.merge(
        matchup_games.rename(columns={"pitcher": "starter_pitcher"})[[
            "game_date", "game_pk", "batter", "starter_pitcher",
            "matchup_games_prior", "matchup_pa_prior", "matchup_hr_prior",
            "matchup_barrel_rate_prior", "matchup_hard_hit_rate_prior",
            "matchup_avg_ev_prior", "matchup_hr_rate_prior",
            "matchup_recent_hr_rate_3", "matchup_recent_hr_rate_5",
            "matchup_recent_barrel_rate_3", "matchup_recent_avg_ev_3",
            "matchup_recent_pa_3", "matchup_history_score_prior"
        ]],
        on=["game_date", "game_pk", "batter", "starter_pitcher"],
        how="left"
    )

    historical_weather = build_historical_weather_context(
        df,
        use_cache=USE_CACHE,
        refresh_cache=REFRESH_CACHE,
    )
    if not historical_weather.empty:
        df = df.merge(
            historical_weather,
            on=["game_date", "game_pk"],
            how="left",
        )

    if "park_factor_static" not in df.columns:
        df["park_factor_static"] = df["home_team"].map(
            lambda t: TEAM_CONTEXT.get(t, {}).get("park_factor", DEFAULT_PARK_FACTOR)
        )
    if "park_factor_dynamic" not in df.columns:
        df["park_factor_dynamic"] = df["park_factor_static"]

    df["park_factor"] = pd.to_numeric(
        df.get("park_factor_dynamic", df.get("park_factor_static", DEFAULT_PARK_FACTOR)),
        errors="coerce",
    ).fillna(df["park_factor_static"]).fillna(DEFAULT_PARK_FACTOR)

    for env_col, default_val in [
        ("temp_f", DEFAULT_TEMP_F),
        ("wind_speed_mph", DEFAULT_WIND_SPEED_MPH),
        ("wind_direction_deg", DEFAULT_WIND_DIRECTION_DEG),
        ("relative_humidity", DEFAULT_REL_HUMIDITY),
    ]:
        if env_col not in df.columns:
            df[env_col] = default_val
        df[env_col] = pd.to_numeric(df[env_col], errors="coerce").fillna(default_val)

    if "wind_out_flag" not in df.columns:
        df["wind_out_flag"] = 0
    df["wind_out_flag"] = pd.to_numeric(df["wind_out_flag"], errors="coerce").fillna(0).astype(int)
    df = add_pull_wind_features(df)

    df["platoon_advantage"] = (
        (df["batter_hand"].isin(["L", "R"]))
        & (df["starter_pitcher_hand"].isin(["L", "R"]))
        & (df["batter_hand"] != df["starter_pitcher_hand"])
    ).astype(int)
    df.loc[df["batter_hand"] == "B", "platoon_advantage"] = 1

    fill_zero_cols = [
        "batter_games_prior", "batter_pa_prior", "batter_hr_prior",
        "batter_hr_rate_prior", "batter_barrel_rate_prior", "batter_hard_hit_rate_prior", "batter_avg_ev_prior",
        "batter_recent_hr_rate_10", "batter_recent_hr_rate_20",
        "batter_recent_barrel_rate_10", "batter_recent_hard_hit_rate_10",
        "batter_recent_avg_ev_10", "batter_recent_pa_10",
        "batter_power_score_prior",
        "pitcher_games_prior", "pitcher_pa_prior", "pitcher_hr_allowed_prior",
        "pitcher_hr_rate_allowed_prior", "pitcher_barrel_rate_allowed_prior",
        "pitcher_hard_hit_rate_allowed_prior", "pitcher_avg_ev_allowed_prior",
        "pitcher_k_rate_prior", "pitcher_recent_k_rate_5", "pitcher_recent_k_rate_10",
        "pitcher_fb_rate_allowed_prior", "pitcher_gb_rate_allowed_prior",
        "pitcher_recent_fb_rate_allowed_5", "pitcher_recent_fb_rate_allowed_10",
        "pitcher_recent_gb_rate_allowed_5", "pitcher_recent_gb_rate_allowed_10",
        "pitcher_recent_hr_allowed_rate_10", "pitcher_recent_hr_allowed_rate_20",
        "pitcher_recent_barrel_allowed_rate_10", "pitcher_recent_avg_ev_allowed_10",
        "pitcher_damage_score_prior",
        "batter_pa_vs_hand_prior", "batter_hr_rate_vs_hand_prior",
        "batter_barrel_rate_vs_hand_prior", "batter_hard_hit_rate_vs_hand_prior", "batter_avg_ev_vs_hand_prior",
        "pitcher_pa_vs_batter_hand_prior", "pitcher_hr_rate_vs_batter_hand_prior",
        "pitcher_barrel_rate_vs_batter_hand_prior", "pitcher_hard_hit_rate_vs_batter_hand_prior",
        "pitcher_avg_ev_vs_batter_hand_prior",
        "pitch_fit_score_prior", "pitch_fit_coverage_prior",
        "matchup_games_prior", "matchup_pa_prior", "matchup_hr_prior",
        "matchup_hr_rate_prior", "matchup_barrel_rate_prior", "matchup_hard_hit_rate_prior",
        "matchup_avg_ev_prior", "matchup_recent_hr_rate_3", "matchup_recent_hr_rate_5",
        "matchup_recent_barrel_rate_3", "matchup_recent_avg_ev_3", "matchup_recent_pa_3",
        "matchup_history_score_prior",
    ]
    for c in fill_zero_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df[df["game_date"] >= pd.to_datetime(MODEL_ROW_START_DATE)].copy()
    df = df[(df["batter_pa_prior"] >= 20) & (df["pitcher_pa_prior"] >= 20)].copy()

    df["log_batter_pa_prior"] = np.log1p(df["batter_pa_prior"])
    df["log_pitcher_pa_prior"] = np.log1p(df["pitcher_pa_prior"])
    df["interaction_hr_rates"] = df["batter_hr_rate_prior"] * df["pitcher_hr_rate_allowed_prior"]
    df["interaction_barrel_rates"] = df["batter_barrel_rate_prior"] * df["pitcher_barrel_rate_allowed_prior"]
    df["interaction_power_damage"] = df["batter_power_score_prior"] * df["pitcher_damage_score_prior"]
    df["interaction_power_vs_k"] = df["batter_power_score_prior"] * (1 - df["pitcher_recent_k_rate_10"].clip(0, 0.5))
    df["interaction_power_vs_elevation"] = df["batter_power_score_prior"] * (
        df["pitcher_recent_fb_rate_allowed_10"].fillna(0) - df["pitcher_recent_gb_rate_allowed_10"].fillna(0)
    )
    df["interaction_matchup_pitchfit"] = df["matchup_history_score_prior"] * df["pitch_fit_score_prior"]
    df["interaction_matchup_power"] = df["matchup_history_score_prior"] * df["batter_power_score_prior"]

    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if USE_CACHE and not force_rebuild:
        df.to_parquet(dataset_path, index=False)
        print(f"Saved model dataset cache to {dataset_path}")

    return df


def _model_dataset_cache_path() -> str:
    return cache_path(
        f"model_dataset_{MODEL_DATASET_CACHE_VERSION}_{FULL_DATA_START_DATE}_to_{FULL_DATA_END_DATE}.parquet"
    )


def build_model_dataset(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build or update the engineered model dataset.

    Normal cached run:
      - load the saved engineered model dataset.

    Daily incremental run after Statcast PA cache appended:
      - load the existing engineered model dataset.
      - rebuild only a warmup window around the newly added dates.
      - append only rows after the old engineered max date.

    Why a warmup window:
      rolling/recent features need prior games to be available. Keeping a 120-day
      warmup window gives the new rows enough history for recent 3/5/10/20-game
      form without rebuilding the full 2024+ dataset.
    """
    dataset_path = _model_dataset_cache_path()

    if USE_CACHE and not REFRESH_CACHE and os.path.exists(dataset_path):
        cached = pd.read_parquet(dataset_path)
        cached["game_date"] = pd.to_datetime(cached["game_date"], errors="coerce")

        if not STATCAST_CACHE_UPDATED or not INCREMENTAL_MODEL_DATASET_UPDATE:
            print(f"Loading cached model dataset from {dataset_path} ...")
            cached = add_pull_wind_features(cached)
            for c in ["pull_wind_mph", "oppo_wind_mph", "cross_wind_abs_mph", "wind_out_to_pull_flag"]:
                if c not in cached.columns:
                    cached[c] = 0
            return cached

        cached_max_date = cached["game_date"].max()
        pa_df = pa_df.copy()
        pa_df["game_date"] = pd.to_datetime(pa_df["game_date"], errors="coerce")
        pa_max_date = pa_df["game_date"].max()

        if pd.notna(cached_max_date) and pd.notna(pa_max_date) and pa_max_date.normalize() <= cached_max_date.normalize():
            print(f"Engineered model dataset is already current through {cached_max_date.date()}.")
            return cached

        warmup_start = (cached_max_date.normalize() - pd.Timedelta(days=MODEL_FEATURE_WARMUP_DAYS))
        pa_recent = pa_df[pa_df["game_date"] >= warmup_start].copy()
        print(
            f"Incrementally rebuilding engineered features from {warmup_start.date()} onward "
            f"and appending rows after {cached_max_date.date()} ..."
        )

        recent_model = _build_model_dataset_full(pa_recent, force_rebuild=True, cache_suffix="_incremental_tmp")
        recent_model["game_date"] = pd.to_datetime(recent_model["game_date"], errors="coerce")
        new_rows = recent_model[recent_model["game_date"] > cached_max_date].copy()

        if new_rows.empty:
            print("No new engineered model rows were created; using existing cache.")
            return cached

        combined = pd.concat([cached, new_rows], ignore_index=True, sort=False)
        dedupe_cols = [c for c in ["game_date", "game_pk", "batter", "starter_pitcher"] if c in combined.columns]
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
        combined = combined.sort_values(["game_date", "game_pk", "batter"], kind="stable").reset_index(drop=True)

        combined = add_pull_wind_features(combined)
        for c in FEATURE_COLUMNS:
            if c not in combined.columns:
                combined[c] = 0
            combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(0)

        combined.to_parquet(dataset_path, index=False)
        print(f"Updated engineered model dataset cache saved to {dataset_path}")
        print(f"Appended engineered rows: {len(new_rows):,}")
        return combined

    return _build_model_dataset_full(pa_df, force_rebuild=False)


# =========================================================
# 4) MODEL TRAINING
# =========================================================
FEATURE_COLUMNS = [
    "is_home_batter",
    "park_factor",
    "temp_f",
    "wind_speed_mph",
    "pull_wind_mph",
    "oppo_wind_mph",
    "cross_wind_abs_mph",
    "wind_out_to_pull_flag",
    "relative_humidity",
    "platoon_advantage",
    "unique_pitchers_faced",

    "batter_games_prior",
    "batter_pa_prior",
    "batter_hr_rate_prior",
    "batter_barrel_rate_prior",
    "batter_hard_hit_rate_prior",
    "batter_avg_ev_prior",
    "batter_recent_hr_rate_10",
    "batter_recent_hr_rate_20",
    "batter_recent_barrel_rate_10",
    "batter_recent_hard_hit_rate_10",
    "batter_recent_avg_ev_10",
    "batter_recent_pa_10",
    "batter_power_score_prior",

    "pitcher_games_prior",
    "pitcher_pa_prior",
    "pitcher_hr_rate_allowed_prior",
    "pitcher_barrel_rate_allowed_prior",
    "pitcher_hard_hit_rate_allowed_prior",
    "pitcher_avg_ev_allowed_prior",
    "pitcher_k_rate_prior",
    "pitcher_recent_k_rate_5",
    "pitcher_recent_k_rate_10",
    "pitcher_fb_rate_allowed_prior",
    "pitcher_gb_rate_allowed_prior",
    "pitcher_recent_fb_rate_allowed_5",
    "pitcher_recent_fb_rate_allowed_10",
    "pitcher_recent_gb_rate_allowed_5",
    "pitcher_recent_gb_rate_allowed_10",
    "pitcher_recent_hr_allowed_rate_10",
    "pitcher_recent_hr_allowed_rate_20",
    "pitcher_recent_barrel_allowed_rate_10",
    "pitcher_recent_avg_ev_allowed_10",
    "pitcher_damage_score_prior",

    "batter_pa_vs_hand_prior",
    "batter_hr_rate_vs_hand_prior",
    "batter_barrel_rate_vs_hand_prior",
    "batter_hard_hit_rate_vs_hand_prior",
    "batter_avg_ev_vs_hand_prior",

    "pitcher_pa_vs_batter_hand_prior",
    "pitcher_hr_rate_vs_batter_hand_prior",
    "pitcher_barrel_rate_vs_batter_hand_prior",
    "pitcher_hard_hit_rate_vs_batter_hand_prior",
    "pitcher_avg_ev_vs_batter_hand_prior",

    "pitch_fit_score_prior",
    "pitch_fit_coverage_prior",

    "matchup_games_prior",
    "matchup_pa_prior",
    "matchup_hr_rate_prior",
    "matchup_barrel_rate_prior",
    "matchup_hard_hit_rate_prior",
    "matchup_avg_ev_prior",
    "matchup_recent_hr_rate_3",
    "matchup_recent_hr_rate_5",
    "matchup_recent_barrel_rate_3",
    "matchup_recent_avg_ev_3",
    "matchup_recent_pa_3",
    "matchup_history_score_prior",

    "log_batter_pa_prior",
    "log_pitcher_pa_prior",
    "interaction_hr_rates",
    "interaction_barrel_rates",
    "interaction_power_damage",
    "interaction_power_vs_k",
    "interaction_power_vs_elevation",
    "interaction_matchup_pitchfit",
    "interaction_matchup_power",
]

# Preserve the full engineered feature list for dataset construction, while allowing
# the trained model to use a pruned subset after the feature audit.
FULL_FEATURE_COLUMNS = FEATURE_COLUMNS.copy()
ACTIVE_FEATURE_COLUMNS = FEATURE_COLUMNS.copy()

def get_model_feature_columns() -> List[str]:
    return ACTIVE_FEATURE_COLUMNS

# ---------------------------------------------------------
# ORIGINAL FEATURE SET RESTORED
# ---------------------------------------------------------
# Reverted the hybrid/defensible feature exclusions after the OOS backtest showed
# they suppressed useful top-N HR signal. BvP and recent HR outcome features are
# back in FEATURE_COLUMNS, matching the stronger original ranking behavior.

def _sort_col(df: pd.DataFrame) -> str:
    # Reverted: rank by raw model probability, not manual hybrid bet_quality_score.
    return "pred_hr_prob"

def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)



def batter_hand_label(hand) -> str:
    """Readable batter handedness label for printed output."""
    if pd.isna(hand):
        return "U"
    hand = str(hand).strip().upper()
    if hand == "R":
        return "R"
    if hand == "L":
        return "L"
    if hand in {"B", "S", "SW"}:
        return "SW"
    return hand or "U"


def pitcher_hand_label(hand) -> str:
    """Readable pitcher handedness label for printed output."""
    if pd.isna(hand):
        return "U"
    hand = str(hand).strip().upper()
    if hand == "R":
        return "R"
    if hand == "L":
        return "L"
    return hand or "U"


def add_macro_board_columns(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return board

    out = board.copy()

    batter_power_raw = (
        0.35 * _safe_series(out, "batter_hr_rate_prior")
        + 0.20 * _safe_series(out, "batter_barrel_rate_prior")
        + 0.15 * _safe_series(out, "batter_hard_hit_rate_prior")
        + 0.15 * (_safe_series(out, "batter_avg_ev_prior") / 100.0)
        + 0.15 * _safe_series(out, "batter_power_score_prior")
    )

    recent_form_raw = (
        0.35 * _safe_series(out, "batter_recent_hr_rate_10")
        + 0.25 * _safe_series(out, "batter_recent_hr_rate_20")
        + 0.20 * _safe_series(out, "batter_recent_barrel_rate_10")
        + 0.10 * _safe_series(out, "batter_recent_hard_hit_rate_10")
        + 0.10 * (_safe_series(out, "batter_recent_avg_ev_10") / 100.0)
    )

    pitcher_vulnerability_raw = (
        0.24 * _safe_series(out, "pitcher_hr_rate_allowed_prior")
        + 0.18 * _safe_series(out, "pitcher_barrel_rate_allowed_prior")
        + 0.12 * _safe_series(out, "pitcher_hard_hit_rate_allowed_prior")
        + 0.08 * (_safe_series(out, "pitcher_avg_ev_allowed_prior") / 100.0)
        + 0.12 * _safe_series(out, "pitcher_recent_hr_allowed_rate_10")
        + 0.08 * _safe_series(out, "pitcher_fb_rate_allowed_prior")
        + 0.07 * _safe_series(out, "pitcher_recent_fb_rate_allowed_10")
        + 0.10 * _safe_series(out, "pitcher_damage_score_prior")
        - 0.07 * _safe_series(out, "pitcher_gb_rate_allowed_prior")
        - 0.06 * _safe_series(out, "pitcher_recent_gb_rate_allowed_10")
        - 0.07 * _safe_series(out, "pitcher_k_rate_prior")
        - 0.05 * _safe_series(out, "pitcher_recent_k_rate_10")
    )

    handedness_splits_raw = (
        0.30 * _safe_series(out, "batter_hr_rate_vs_hand_prior")
        + 0.15 * _safe_series(out, "batter_barrel_rate_vs_hand_prior")
        + 0.10 * _safe_series(out, "batter_hard_hit_rate_vs_hand_prior")
        + 0.15 * (_safe_series(out, "batter_avg_ev_vs_hand_prior") / 100.0)
        + 0.20 * _safe_series(out, "pitcher_hr_rate_vs_batter_hand_prior")
        + 0.10 * _safe_series(out, "platoon_advantage")
    )

    pitch_type_matchup_raw = (
        0.80 * _safe_series(out, "pitch_fit_score_prior")
        + 0.20 * _safe_series(out, "pitch_fit_coverage_prior")
    )

    matchup_history_raw = (
        0.35 * _safe_series(out, "matchup_hr_rate_prior")
        + 0.20 * _safe_series(out, "matchup_barrel_rate_prior")
        + 0.10 * _safe_series(out, "matchup_hard_hit_rate_prior")
        + 0.10 * (_safe_series(out, "matchup_avg_ev_prior") / 100.0)
        + 0.10 * _safe_series(out, "matchup_recent_hr_rate_3")
        + 0.05 * _safe_series(out, "matchup_recent_hr_rate_5")
        + 0.10 * _safe_series(out, "matchup_recent_barrel_rate_3")
    )

    environment_raw = (
        0.40 * _safe_series(out, "park_factor")
        + 0.20 * (_safe_series(out, "temp_f") / 100.0)
        + 0.20 * ((_safe_series(out, "pull_wind_mph") + 20.0) / 40.0)
        + 0.10 * _safe_series(out, "wind_out_to_pull_flag")
        + 0.10 * (_safe_series(out, "relative_humidity") / 100.0)
    )

    raw_map = {
        "batter_power": batter_power_raw,
        "recent_form": recent_form_raw,
        "pitcher_vulnerability": pitcher_vulnerability_raw,
        "handedness_splits": handedness_splits_raw,
        "pitch_type_matchup": pitch_type_matchup_raw,
        "matchup_history": matchup_history_raw,
        "environment": environment_raw,
    }

    for name, raw in raw_map.items():
        out[f"{name}_raw"] = raw
        ranks = raw.rank(method="average", pct=True).fillna(0) * 100
        out[name] = ranks.round(1)

    # Final probability is the RAW model probability because it preserves player-to-player separation.
    # Calibrated probability is still shown separately as a conservative reference.
    out["final_hr_probability"] = (_safe_series(out, "pred_hr_prob") * 100).round(1)
    if "calibrated_hr_prob" in out.columns:
        out["calibrated_hr_probability"] = (_safe_series(out, "calibrated_hr_prob") * 100).round(1)

    if "batter_hand" in out.columns:
        out["batter_hand_label"] = out["batter_hand"].apply(batter_hand_label)
        out["batter_name_hand"] = out.apply(
            lambda x: f"{x['batter_name']} ({x['batter_hand_label']})"
            if pd.notna(x.get("batter_name")) and str(x.get("batter_name")).strip()
            else f"Unknown Batter ({x['batter_hand_label']})",
            axis=1,
        )

    if "starter_pitcher_hand" in out.columns:
        out["pitcher_hand_label"] = out["starter_pitcher_hand"].apply(pitcher_hand_label)
        out["pitcher_name_hand"] = out.apply(
            lambda x: f"{x['pitcher_name']} ({x['pitcher_hand_label']})"
            if pd.notna(x.get("pitcher_name")) and str(x.get("pitcher_name")).strip()
            else f"Unknown Pitcher ({x['pitcher_hand_label']})",
            axis=1,
        )

    out["game_matchup"] = out.apply(
        lambda x: " vs. ".join(sorted([str(x["batting_team"]), str(x["fielding_team"])])),
        axis=1,
    )
    return out

def print_forward_board_tables(board: pd.DataFrame, top_n: int) -> None:
    if board.empty:
        print("\nNo forward board available.")
        return

    ranked = (
        board.sort_values(_sort_col(board), ascending=False)
        .drop_duplicates(subset=["batter"])
        .head(top_n)
        .copy()
    )

    ranked["ranking"] = np.arange(1, len(ranked) + 1)

    summary_cols = [
        "ranking",
        "batter_name_hand",
        "batting_team",
        "pitcher_name_hand",
        "bet_quality_score",
        "batter_power",
        "recent_form",
        "pitcher_vulnerability",
        "handedness_splits",
        "pitch_type_matchup",
        "matchup_history",
        "environment",
        "pa_opportunity",
        "final_hr_probability",
    ]
    summary_cols = [c for c in summary_cols if c in ranked.columns]

    print("\n=== TOP FORWARD HR PLAYS (TOP 30) ===")
    print(ranked[summary_cols].to_string(index=False))

    matchup_cols = [
        "ranking",
        "game_matchup",
        "batter_name_hand",
        "batting_team",
        "pitcher_name_hand",
        "final_hr_probability",
        "wind_out_to_pull_flag",
        "commence_time",
        "temp_f",
        "wind_speed_mph",
        "is_roofed_no_wind",
        "weather_blowing_out",
    ]
    matchup_cols = [c for c in matchup_cols if c in ranked.columns]
    matchup_view = ranked[matchup_cols].copy()
    matchup_view["commence_time"] = pd.to_datetime(matchup_view["commence_time"], utc=True, errors="coerce")
    matchup_view = matchup_view.sort_values(["commence_time", "game_matchup", "ranking"], ascending=[True, True, True])

    print("\n=== TOP FORWARD HR PLAYS BY MATCHUP ===")
    for game, group in matchup_view.groupby("game_matchup", sort=False):
        first = group.iloc[0]
        temp = pd.to_numeric(first.get("temp_f", np.nan), errors="coerce")
        wind = pd.to_numeric(first.get("wind_speed_mph", np.nan), errors="coerce")
        roofed = bool(first.get("is_roofed_no_wind", False))
        blowing_out = bool(first.get("weather_blowing_out", False))

        temp_txt = "NA" if pd.isna(temp) else f"{temp:.0f}F"
        wind_txt = "NA" if pd.isna(wind) else f"{wind:.1f} mph"
        roof_txt = "YES - wind disabled" if roofed else "NO"
        blowing_txt = "YES" if blowing_out else "NO"

        print(f"\n=== {game} ===")
        print(f"Conditions: Temp {temp_txt} | Wind {wind_txt} | Domed: {roof_txt} | Wind blowing out: {blowing_txt}")

        game_cols = [
            "ranking",
            "batter_name_hand",
            "batting_team",
            "pitcher_name_hand",
            "final_hr_probability",
            "wind_out_to_pull_flag",
        ]
        game_cols = [c for c in game_cols if c in group.columns]
        print(group[game_cols].to_string(index=False))


def split_model_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["game_date"] <= pd.to_datetime(TRAIN_END_DATE)].copy()
    valid_df = df[
        (df["game_date"] > pd.to_datetime(TRAIN_END_DATE))
        & (df["game_date"] <= pd.to_datetime(VALID_END_DATE))
    ].copy()
    test_df = df[df["game_date"] > pd.to_datetime(VALID_END_DATE)].copy()

    print("\nDataset splits:")
    for name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        if part.empty:
            print(f"  {name}: 0 rows")
        else:
            print(
                f"  {name}: {len(part):,} rows | "
                f"{part['game_date'].min().date()} -> {part['game_date'].max().date()} | "
                f"HR rate={part['home_run_game'].mean():.4f}"
            )

    return train_df, valid_df, test_df


def evaluate_probs(y_true: pd.Series, probs: np.ndarray, label: str) -> None:
    probs = np.clip(np.asarray(probs), 1e-6, 1 - 1e-6)
    ll = log_loss(y_true, probs)
    bs = brier_score_loss(y_true, probs)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = np.nan
    print(f"{label}: logloss={ll:.6f} | brier={bs:.6f} | roc_auc={auc:.4f}")


def print_feature_importance(model) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        print("\nModel does not expose feature_importances_.")
        return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

    fi = pd.DataFrame({
        "feature": get_model_feature_columns(),
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    total = fi["importance"].sum()
    fi["importance_pct"] = np.where(total > 0, fi["importance"] / total * 100.0, 0.0)

    print("\n=== FEATURE IMPORTANCE ===")
    print(fi.head(25).to_string(index=False))

    fi.to_csv("trained_hr_model_feature_importance.csv", index=False)
    print("Saved: trained_hr_model_feature_importance.csv")
    return fi



def add_recency_sample_weights(df: pd.DataFrame) -> pd.Series:
    """
    Return row-level sample weights for model fitting.

    Why:
      - 2024 adds useful HR/weather/park/pitch-mix sample size.
      - 2025 should be the baseline for current skill/run environment.
      - 2026 gets a modest boost, but not too much because early-season samples are noisy.
    """
    if df.empty:
        return pd.Series(dtype=float)

    weights = pd.Series(1.0, index=df.index, dtype=float)

    game_dates = pd.to_datetime(df["game_date"], errors="coerce")
    weights.loc[game_dates < pd.Timestamp("2025-01-01")] = RECENCY_WEIGHT_2024
    weights.loc[
        (game_dates >= pd.Timestamp("2025-01-01"))
        & (game_dates < pd.Timestamp("2026-01-01"))
    ] = RECENCY_WEIGHT_2025
    weights.loc[game_dates >= pd.Timestamp("2026-01-01")] = RECENCY_WEIGHT_2026

    weights = weights.fillna(1.0)
    return weights


def print_sample_weight_summary(train_df: pd.DataFrame, sample_weight: pd.Series) -> None:
    if train_df.empty or sample_weight.empty:
        return

    temp = train_df[["game_date", "home_run_game"]].copy()
    temp["game_date"] = pd.to_datetime(temp["game_date"], errors="coerce")
    temp["season"] = temp["game_date"].dt.year
    temp["sample_weight"] = sample_weight.values

    summary = (
        temp.groupby("season", as_index=False)
        .agg(
            rows=("home_run_game", "count"),
            hr_rate=("home_run_game", "mean"),
            avg_sample_weight=("sample_weight", "mean"),
            weighted_rows=("sample_weight", "sum"),
        )
    )

    print("\n=== RECENCY SAMPLE WEIGHTS ===")
    print(summary.to_string(index=False))


def _safe_abs_corr_with_target(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[target_col], errors="coerce")
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        if x.nunique(dropna=True) <= 1:
            corr = 0.0
        else:
            corr = x.corr(y)
        rows.append({"feature": col, "target_corr_abs": 0.0 if pd.isna(corr) else abs(float(corr))})
    return pd.DataFrame(rows)


def _feature_family(feature: str) -> str:
    """Coarse feature family label for easier audit reading."""
    prefixes = [
        "batter_recent", "batter_pa_vs", "batter_hr_rate_vs", "batter_barrel_rate_vs",
        "batter_hard_hit_rate_vs", "batter_avg_ev_vs", "batter_",
        "pitcher_recent", "pitcher_pa_vs", "pitcher_hr_rate_vs", "pitcher_barrel_rate_vs",
        "pitcher_hard_hit_rate_vs", "pitcher_avg_ev_vs", "pitcher_",
        "matchup_recent", "matchup_", "interaction_", "pitch_fit", "wind_",
        "pull_wind", "oppo_wind", "cross_wind", "park_", "temp_", "relative_",
    ]
    for prefix in prefixes:
        if feature.startswith(prefix):
            return prefix.rstrip("_")
    return "other"


def run_feature_audit_and_select(
    model,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
) -> List[str]:
    """
    Quantify feature redundancy and validation importance, then choose a cleaner
    feature subset for the final model fit.

    The audit intentionally does not change printed forward-board columns. It only
    changes which backend features the probability model uses.
    """
    print("\n=== FEATURE AUDIT / REDUNDANCY PRUNING ===")

    available = [c for c in feature_cols if c in train_df.columns and c in valid_df.columns]
    if not available:
        print("No available feature columns for audit; using original feature set.")
        return feature_cols

    x_train = train_df[available].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = train_df["home_run_game"].astype(int)
    x_valid = valid_df[available].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_valid = valid_df["home_run_game"].astype(int)

    # 1) Redundancy map: absolute feature-feature correlation on train rows.
    corr = x_train.corr().abs().fillna(0)
    corr_long_rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = float(corr.loc[a, b])
            if val >= 0.70:
                corr_long_rows.append({
                    "feature_a": a,
                    "feature_b": b,
                    "abs_corr": val,
                    "family_a": _feature_family(a),
                    "family_b": _feature_family(b),
                })
    corr_long = pd.DataFrame(corr_long_rows).sort_values("abs_corr", ascending=False) if corr_long_rows else pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr", "family_a", "family_b"])
    corr_long.to_csv("trained_hr_model_feature_redundancy_pairs.csv", index=False)

    # 2) Simple target relationship, useful when choosing between correlated twins.
    target_corr = _safe_abs_corr_with_target(train_df, available, "home_run_game")

    # 3) True validation value: permutation importance using validation log loss.
    # Sampling keeps this from becoming painfully slow on large validation windows.
    if len(x_valid) > FEATURE_AUDIT_VALID_SAMPLE_ROWS:
        sample_idx = x_valid.sample(FEATURE_AUDIT_VALID_SAMPLE_ROWS, random_state=FEATURE_AUDIT_RANDOM_STATE).index
        x_perm = x_valid.loc[sample_idx]
        y_perm = y_valid.loc[sample_idx]
    else:
        x_perm = x_valid
        y_perm = y_valid

    try:
        perm = permutation_importance(
            model,
            x_perm,
            y_perm,
            scoring="neg_log_loss",
            n_repeats=FEATURE_AUDIT_PERM_REPEATS,
            random_state=FEATURE_AUDIT_RANDOM_STATE,
            n_jobs=1,
        )
        perm_df = pd.DataFrame({
            "feature": available,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        })
    except Exception as e:
        print(f"⚠️ Permutation importance failed; falling back to target correlation only: {e}")
        perm_df = pd.DataFrame({
            "feature": available,
            "perm_importance_mean": 0.0,
            "perm_importance_std": 0.0,
        })

    audit = (
        pd.DataFrame({"feature": available})
        .merge(perm_df, on="feature", how="left")
        .merge(target_corr, on="feature", how="left")
    )
    audit["family"] = audit["feature"].map(_feature_family)
    audit["nonzero_rate_train"] = [(x_train[c] != 0).mean() for c in available]
    audit["nunique_train"] = [x_train[c].nunique(dropna=True) for c in available]
    audit["audit_score"] = (
        pd.to_numeric(audit["perm_importance_mean"], errors="coerce").fillna(0)
        + 0.05 * pd.to_numeric(audit["target_corr_abs"], errors="coerce").fillna(0)
    )

    # Constant features are pure noise for this split.
    constant_features = set(audit.loc[audit["nunique_train"] <= 1, "feature"])

    # Low/negative permutation importance features are candidates for removal.
    low_value_features = set(audit.loc[
        pd.to_numeric(audit["perm_importance_mean"], errors="coerce").fillna(0) < FEATURE_MIN_PERM_IMPORTANCE,
        "feature"
    ])

    # Redundancy pruning: within highly correlated pairs, keep the feature with
    # stronger validation importance / target relationship.
    score_map = audit.set_index("feature")["audit_score"].to_dict()
    removed_for_corr = set()
    kept_preference = []
    if not corr_long.empty:
        high_corr = corr_long[corr_long["abs_corr"] >= FEATURE_CORR_PRUNE_THRESHOLD].copy()
        for _, row in high_corr.iterrows():
            a, b = row["feature_a"], row["feature_b"]
            if a in removed_for_corr or b in removed_for_corr:
                continue
            sa = float(score_map.get(a, 0.0))
            sb = float(score_map.get(b, 0.0))
            # Tie-breaker: prefer raw/base features over hand-built composites only when scores are close.
            if abs(sa - sb) < 1e-8:
                composite_tokens = ["score", "interaction"]
                a_is_composite = any(tok in a for tok in composite_tokens)
                b_is_composite = any(tok in b for tok in composite_tokens)
                if a_is_composite and not b_is_composite:
                    drop, keep = a, b
                elif b_is_composite and not a_is_composite:
                    drop, keep = b, a
                else:
                    drop, keep = b, a
            elif sa >= sb:
                keep, drop = a, b
            else:
                keep, drop = b, a
            removed_for_corr.add(drop)
            kept_preference.append({
                "kept_feature": keep,
                "dropped_feature": drop,
                "abs_corr": float(row["abs_corr"]),
                "kept_score": float(score_map.get(keep, 0.0)),
                "dropped_score": float(score_map.get(drop, 0.0)),
            })

    removal_reasons = []
    to_remove = set()
    for f in constant_features:
        to_remove.add(f)
        removal_reasons.append({"feature": f, "remove_reason": "constant_or_single_value"})
    for f in low_value_features:
        if f not in to_remove:
            to_remove.add(f)
            removal_reasons.append({"feature": f, "remove_reason": "negative_or_near_zero_validation_importance"})
    for f in removed_for_corr:
        if f not in to_remove:
            to_remove.add(f)
            removal_reasons.append({"feature": f, "remove_reason": "highly_correlated_with_stronger_feature"})

    selected = [f for f in available if f not in to_remove]

    # Safety guard: do not accidentally collapse the model if a weird split makes
    # permutation importance noisy. Keep at least half the original features.
    min_keep = max(20, int(len(available) * 0.50))
    if len(selected) < min_keep:
        ranked = audit.sort_values(["audit_score", "target_corr_abs"], ascending=False)["feature"].tolist()
        selected = []
        for f in ranked:
            if f not in selected:
                selected.append(f)
            if len(selected) >= min_keep:
                break
        print(f"Feature pruning hit safety guard; keeping top {len(selected)} audited features.")

    audit["selected_for_model"] = audit["feature"].isin(selected).astype(int)
    reason_df = pd.DataFrame(removal_reasons)
    audit = audit.merge(reason_df, on="feature", how="left") if not reason_df.empty else audit.assign(remove_reason="")
    audit = audit.sort_values(["selected_for_model", "audit_score"], ascending=[False, False])

    audit.to_csv("trained_hr_model_feature_audit.csv", index=False)
    pd.DataFrame(kept_preference).to_csv("trained_hr_model_feature_correlation_pruning.csv", index=False)
    pd.DataFrame({"feature": selected}).to_csv("trained_hr_model_selected_features.csv", index=False)

    print(f"Original features: {len(available)}")
    print(f"Selected features: {len(selected)}")
    print(f"Dropped features: {len(available) - len(selected)}")
    print("Saved:")
    print(" - trained_hr_model_feature_audit.csv")
    print(" - trained_hr_model_feature_redundancy_pairs.csv")
    print(" - trained_hr_model_feature_correlation_pruning.csv")
    print(" - trained_hr_model_selected_features.csv")

    print("\nTop selected features by audit score:")
    print(audit[audit["selected_for_model"].eq(1)][["feature", "family", "perm_importance_mean", "target_corr_abs", "audit_score"]].head(25).to_string(index=False))

    return selected if APPLY_FEATURE_PRUNING else available

def fit_calibrated_hgb(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    feature_cols = get_model_feature_columns()
    x_train = train_df[feature_cols].copy()
    y_train = train_df["home_run_game"].astype(int).copy()

    x_valid = valid_df[feature_cols].copy()
    y_valid = valid_df["home_run_game"].astype(int).copy()

    if USE_RECENCY_SAMPLE_WEIGHTS:
        train_sample_weight = add_recency_sample_weights(train_df)
        print_sample_weight_summary(train_df, train_sample_weight)
    else:
        train_sample_weight = None

    param_grid = [
        {"learning_rate": 0.03, "max_depth": 3, "max_iter": 250, "min_samples_leaf": 50, "l2_regularization": 0.0},
        {"learning_rate": 0.03, "max_depth": 4, "max_iter": 300, "min_samples_leaf": 50, "l2_regularization": 0.0},
        {"learning_rate": 0.02, "max_depth": 4, "max_iter": 400, "min_samples_leaf": 40, "l2_regularization": 0.0},
        {"learning_rate": 0.02, "max_depth": 5, "max_iter": 450, "min_samples_leaf": 40, "l2_regularization": 0.01},
        {"learning_rate": 0.015, "max_depth": 5, "max_iter": 600, "min_samples_leaf": 30, "l2_regularization": 0.05},
    ]

    best_model = None
    best_params = None
    best_valid_ll = np.inf

    print("\nTuning HistGradientBoostingClassifier...")
    for i, params in enumerate(param_grid, start=1):
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            random_state=42,
            **params
        )
        model.fit(x_train, y_train, sample_weight=train_sample_weight)
        valid_raw = model.predict_proba(x_valid)[:, 1]
        valid_ll = log_loss(y_valid, np.clip(valid_raw, 1e-6, 1 - 1e-6))
        print(f"  candidate {i}: params={params} | valid_logloss={valid_ll:.6f}")

        if valid_ll < best_valid_ll:
            best_valid_ll = valid_ll
            best_model = model
            best_params = params

    print("\nBest raw model params:")
    print(best_params)

    train_raw = best_model.predict_proba(x_train)[:, 1]
    valid_raw = best_model.predict_proba(x_valid)[:, 1]

    print("\nRaw model performance:")
    evaluate_probs(y_train, train_raw, "train_raw")
    evaluate_probs(y_valid, valid_raw, "valid_raw")

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(valid_raw, y_valid)
    valid_cal = calibrator.transform(valid_raw)

    print("\nCalibrated validation performance:")
    evaluate_probs(y_valid, valid_cal, "valid_calibrated")

    return best_model, calibrator


def predict_raw(model, df: pd.DataFrame) -> np.ndarray:
    """Raw model probability. Use this for ranking because it preserves separation."""
    raw = model.predict_proba(df[get_model_feature_columns()])[:, 1]
    return np.clip(raw, 0.0, 1.0)


def predict_calibrated(model, calibrator, df: pd.DataFrame) -> np.ndarray:
    """Calibrated probability. Useful as a conservative reference, but isotonic may bucket values."""
    raw = predict_raw(model, df)
    cal = calibrator.transform(raw)
    return np.clip(cal, 0.0, 1.0)


# =========================================================
# 5) BACKTEST REPORTING
# =========================================================
def summarize_top_n(df: pd.DataFrame, top_n: int) -> dict:
    ranked = (
        df.sort_values(_sort_col(df), ascending=False)
        .drop_duplicates(subset=["game_date", "batter"])
        .groupby("game_date", as_index=False, group_keys=False)
        .head(top_n)
        .copy()
    )

    daily = (
        ranked.groupby("game_date", as_index=False)
        .agg(players=("batter", "count"), homers=("home_run_game", "sum"))
    )
    daily["hit_rate"] = daily["homers"] / daily["players"]

    return {
        "top_n": top_n,
        "days": int(daily["game_date"].nunique()) if not daily.empty else 0,
        "total_players": int(daily["players"].sum()) if not daily.empty else 0,
        "total_homers": int(daily["homers"].sum()) if not daily.empty else 0,
        "avg_daily_hit_rate": float(daily["hit_rate"].mean()) if not daily.empty else np.nan,
        "overall_hit_rate": float(daily["homers"].sum() / daily["players"].sum()) if not daily.empty else np.nan,
        "avg_model_prob": float(ranked["pred_hr_prob"].mean()) if not ranked.empty else np.nan,
    }


def run_backtest(model, calibrator, test_df: pd.DataFrame) -> None:
    if test_df.empty:
        print("\nNo test rows available for backtest.")
        return

    scored = test_df.copy()
    scored["raw_hr_prob"] = predict_raw(model, scored)
    scored["calibrated_hr_prob"] = predict_calibrated(model, calibrator, scored)
    # Use raw probability for ranking/top-N separation.
    scored["pred_hr_prob"] = scored["raw_hr_prob"]
    scored = add_macro_board_columns(scored)

    evaluate_probs(scored["home_run_game"], scored["raw_hr_prob"], "\nTest set raw")
    evaluate_probs(scored["home_run_game"], scored["calibrated_hr_prob"], "Test set calibrated")

    summaries = []
    for n in [10, 20, 50]:
        summaries.append(summarize_top_n(scored, n))

    summary_df = pd.DataFrame(summaries)
    print("\n=== OUT-OF-SAMPLE TOP-N BACKTEST ===")
    print(summary_df.to_string(index=False))

    scored.to_csv("trained_hr_model_scored_test_rows.csv", index=False)
    summary_df.to_csv("trained_hr_model_backtest_summary.csv", index=False)

    print("\nSaved:")
    print(" - trained_hr_model_scored_test_rows.csv")
    print(" - trained_hr_model_backtest_summary.csv")


# =========================================================
# 6) FORWARD-LOOKING DAILY BOARD
# =========================================================
def get_daily_slate(target_date: str) -> pd.DataFrame:
    print(f"\nPulling MLB schedule for {target_date} ...")
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": target_date,
        "hydrate": "probablePitcher,team"
    }
    payload = get_json(url, params=params)

    rows = []
    if not payload:
        return pd.DataFrame()

    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            venue = game.get("venue", {})

            home_team = home.get("team", {}).get("name")
            away_team = away.get("team", {}).get("name")
            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})

            rows.append({
                "game_pk": game.get("gamePk"),
                "commence_time": game.get("gameDate"),
                "venue_name": venue.get("name"),
                "home_team": home_team,
                "away_team": away_team,
                "home_pitcher_id": home_pitcher.get("id"),
                "home_pitcher_name": home_pitcher.get("fullName"),
                "away_pitcher_id": away_pitcher.get("id"),
                "away_pitcher_name": away_pitcher.get("fullName"),
            })

    return pd.DataFrame(rows)


def fetch_weather_for_game(lat: float, lon: float, start_iso_utc: str) -> dict:
    if pd.isna(lat) or pd.isna(lon) or not start_iso_utc:
        return {
            "temp_f": DEFAULT_TEMP_F,
            "wind_speed_mph": DEFAULT_WIND_SPEED_MPH,
            "wind_direction_deg": DEFAULT_WIND_DIRECTION_DEG,
            "relative_humidity": DEFAULT_REL_HUMIDITY,
        }

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "forecast_days": 3,
    }
    data = get_json(url, params=params)
    if data is None:
        return {
            "temp_f": DEFAULT_TEMP_F,
            "wind_speed_mph": DEFAULT_WIND_SPEED_MPH,
            "wind_direction_deg": DEFAULT_WIND_DIRECTION_DEG,
            "relative_humidity": DEFAULT_REL_HUMIDITY,
        }

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {
            "temp_f": DEFAULT_TEMP_F,
            "wind_speed_mph": DEFAULT_WIND_SPEED_MPH,
            "wind_direction_deg": DEFAULT_WIND_DIRECTION_DEG,
            "relative_humidity": DEFAULT_REL_HUMIDITY,
        }

    target_dt = pd.to_datetime(start_iso_utc, utc=True)
    weather_df = pd.DataFrame({
        "time": pd.to_datetime(times, utc=True),
        "temp_f": hourly.get("temperature_2m", []),
        "relative_humidity": hourly.get("relative_humidity_2m", []),
        "wind_speed_mph": hourly.get("wind_speed_10m", []),
        "wind_direction_deg": hourly.get("wind_direction_10m", []),
    })
    weather_df["time_diff"] = (weather_df["time"] - target_dt).abs()
    row = weather_df.sort_values("time_diff").iloc[0]
    return {
        "temp_f": float(row["temp_f"]),
        "wind_speed_mph": float(row["wind_speed_mph"]),
        "wind_direction_deg": float(row["wind_direction_deg"]),
        "relative_humidity": float(row["relative_humidity"]),
    }




def get_historical_weather_cache_path(start_date: str, end_date: str) -> str:
    return cache_path(f"historical_game_weather_{start_date}_to_{end_date}.parquet")


def fetch_historical_weather_for_game(lat: float, lon: float, start_iso_utc: str) -> dict:
    if pd.isna(lat) or pd.isna(lon) or not start_iso_utc:
        return {
            "temp_f": DEFAULT_TEMP_F,
            "wind_speed_mph": DEFAULT_WIND_SPEED_MPH,
            "wind_direction_deg": DEFAULT_WIND_DIRECTION_DEG,
            "relative_humidity": DEFAULT_REL_HUMIDITY,
        }

    target_dt = pd.to_datetime(start_iso_utc, utc=True, errors="coerce")
    if pd.isna(target_dt):
        return {
            "temp_f": DEFAULT_TEMP_F,
            "wind_speed_mph": DEFAULT_WIND_SPEED_MPH,
            "wind_direction_deg": DEFAULT_WIND_DIRECTION_DEG,
            "relative_humidity": DEFAULT_REL_HUMIDITY,
        }

    date_str = target_dt.strftime("%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    }
    data = get_json(url, params=params)
    if data is None:
        return {
            "temp_f": DEFAULT_TEMP_F,
            "wind_speed_mph": DEFAULT_WIND_SPEED_MPH,
            "wind_direction_deg": DEFAULT_WIND_DIRECTION_DEG,
            "relative_humidity": DEFAULT_REL_HUMIDITY,
        }

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {
            "temp_f": DEFAULT_TEMP_F,
            "wind_speed_mph": DEFAULT_WIND_SPEED_MPH,
            "wind_direction_deg": DEFAULT_WIND_DIRECTION_DEG,
            "relative_humidity": DEFAULT_REL_HUMIDITY,
        }

    weather_df = pd.DataFrame({
        "time": pd.to_datetime(times, utc=True),
        "temp_f": hourly.get("temperature_2m", []),
        "relative_humidity": hourly.get("relative_humidity_2m", []),
        "wind_speed_mph": hourly.get("wind_speed_10m", []),
        "wind_direction_deg": hourly.get("wind_direction_10m", []),
    })
    weather_df["time_diff"] = (weather_df["time"] - target_dt).abs()
    row = weather_df.sort_values("time_diff").iloc[0]
    return {
        "temp_f": float(row["temp_f"]),
        "wind_speed_mph": float(row["wind_speed_mph"]),
        "wind_direction_deg": float(row["wind_direction_deg"]),
        "relative_humidity": float(row["relative_humidity"]),
    }


def build_historical_weather_context(df: pd.DataFrame, use_cache: bool = True, refresh_cache: bool = False) -> pd.DataFrame:
    weather_cache_path = get_historical_weather_cache_path(FULL_DATA_START_DATE, FULL_DATA_END_DATE)

    if use_cache and not refresh_cache and os.path.exists(weather_cache_path):
        print(f"Loading cached historical weather context from {weather_cache_path} ...")
        weather = pd.read_parquet(weather_cache_path)
        return weather

    print("Building historical weather context from MLB schedule + Open-Meteo archive...")

    base_games = (
        df[["game_date", "game_pk", "home_team", "fielding_team"]]
        .drop_duplicates()
        .copy()
    )
    base_games["game_date"] = pd.to_datetime(base_games["game_date"]).dt.normalize()
    # Venue is always the home team/ballpark. Do not use fielding_team here,
    # because home hitters face the away fielding team but still play in the home park.
    base_games["venue_team"] = base_games["home_team"]

    rows = []
    schedule_cache: Dict[str, dict] = {}

    for _, game in base_games.sort_values(["game_date", "game_pk"]).iterrows():
        date_str = pd.to_datetime(game["game_date"]).strftime("%Y-%m-%d")
        if date_str not in schedule_cache:
            schedule_cache[date_str] = get_json(
                "https://statsapi.mlb.com/api/v1/schedule",
                params={"sportId": 1, "date": date_str},
            ) or {}

        payload = schedule_cache[date_str]
        game_lookup = None
        for date_block in payload.get("dates", []):
            for scheduled_game in date_block.get("games", []):
                if int(scheduled_game.get("gamePk", -1)) == int(game["game_pk"]):
                    game_lookup = scheduled_game
                    break
            if game_lookup is not None:
                break

        start_iso_utc = None
        if game_lookup is not None:
            start_iso_utc = game_lookup.get("gameDate")

        venue_team = game["venue_team"]
        ctx = TEAM_CONTEXT.get(venue_team, {})
        lat = ctx.get("lat")
        lon = ctx.get("lon")
        base_pf = ctx.get("park_factor", DEFAULT_PARK_FACTOR)
        out_bearing = ctx.get("cf_bearing", ctx.get("out_to_cf_bearing", 0))

        weather = fetch_historical_weather_for_game(lat, lon, start_iso_utc)
        weather = neutralize_wind_for_roofed_venue(weather, venue_team)

        dynamic_pf = dynamic_park_factor(
            base_park_factor=base_pf,
            temp_f=weather["temp_f"],
            wind_speed_mph=weather["wind_speed_mph"],
            wind_direction_deg=weather["wind_direction_deg"],
            relative_humidity=weather["relative_humidity"],
            out_to_cf_bearing=out_bearing,
        )

        rows.append({
            "game_date": game["game_date"],
            "game_pk": int(game["game_pk"]),
            "park_factor_static": base_pf,
            "park_factor_dynamic": dynamic_pf,
            "park_factor": dynamic_pf,
            "temp_f": weather["temp_f"],
            "wind_speed_mph": weather["wind_speed_mph"],
            "wind_direction_deg": weather["wind_direction_deg"],
            "relative_humidity": weather["relative_humidity"],
            "is_roofed_no_wind": weather.get("is_roofed_no_wind", 0),
            "weather_blowing_out": 0 if weather.get("is_roofed_no_wind", 0) else wind_out_flag(weather["wind_direction_deg"], out_bearing),
            "lf_bearing": ctx.get("lf_bearing", np.nan),
            "cf_bearing": ctx.get("cf_bearing", out_bearing),
            "rf_bearing": ctx.get("rf_bearing", np.nan),
            "wind_to_lf_mph": directional_wind_mph(weather["wind_speed_mph"], weather["wind_direction_deg"], ctx.get("lf_bearing", out_bearing)),
            "wind_to_cf_mph": directional_wind_mph(weather["wind_speed_mph"], weather["wind_direction_deg"], ctx.get("cf_bearing", out_bearing)),
            "wind_to_rf_mph": directional_wind_mph(weather["wind_speed_mph"], weather["wind_direction_deg"], ctx.get("rf_bearing", out_bearing)),
            "wind_out_flag": wind_out_flag(weather["wind_direction_deg"], out_bearing),
        })

    weather_df = pd.DataFrame(rows)
    if use_cache and not weather_df.empty:
        weather_df.to_parquet(weather_cache_path, index=False)
        print(f"Saved historical weather cache to {weather_cache_path}")

    return weather_df

def add_real_weather_and_park(slate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, game in slate.iterrows():
        ctx = TEAM_CONTEXT.get(game["home_team"], {})
        lat = ctx.get("lat")
        lon = ctx.get("lon")
        park_factor = ctx.get("park_factor", DEFAULT_PARK_FACTOR)
        out_bearing = ctx.get("cf_bearing", ctx.get("out_to_cf_bearing", 0))

        w = fetch_weather_for_game(lat, lon, game["commence_time"])
        w = neutralize_wind_for_roofed_venue(w, game["home_team"])

        dynamic_pf = dynamic_park_factor(
            base_park_factor=park_factor,
            temp_f=w["temp_f"],
            wind_speed_mph=w["wind_speed_mph"],
            wind_direction_deg=w["wind_direction_deg"],
            relative_humidity=w["relative_humidity"],
            out_to_cf_bearing=out_bearing,
        )

        rows.append({
            "game_pk": game["game_pk"],
            "park_factor_static": park_factor,
            "park_factor": dynamic_pf,
            "park_factor_dynamic": dynamic_pf,
            "temp_f": w["temp_f"],
            "wind_speed_mph": w["wind_speed_mph"],
            "wind_direction_deg": w["wind_direction_deg"],
            "relative_humidity": w["relative_humidity"],
            "is_roofed_no_wind": w.get("is_roofed_no_wind", 0),
            "weather_blowing_out": 0 if w.get("is_roofed_no_wind", 0) else wind_out_flag(w["wind_direction_deg"], out_bearing),
            "wind_out_flag": 0 if w.get("is_roofed_no_wind", 0) else wind_out_flag(w["wind_direction_deg"], out_bearing),
            "lf_bearing": ctx.get("lf_bearing", np.nan),
            "cf_bearing": ctx.get("cf_bearing", out_bearing),
            "rf_bearing": ctx.get("rf_bearing", np.nan),
            "wind_to_lf_mph": 0.0 if w.get("is_roofed_no_wind", 0) else directional_wind_mph(w["wind_speed_mph"], w["wind_direction_deg"], ctx.get("lf_bearing", out_bearing)),
            "wind_to_cf_mph": 0.0 if w.get("is_roofed_no_wind", 0) else directional_wind_mph(w["wind_speed_mph"], w["wind_direction_deg"], ctx.get("cf_bearing", out_bearing)),
            "wind_to_rf_mph": 0.0 if w.get("is_roofed_no_wind", 0) else directional_wind_mph(w["wind_speed_mph"], w["wind_direction_deg"], ctx.get("rf_bearing", out_bearing)),
        })

    weather = pd.DataFrame(rows)
    return slate.merge(weather, on="game_pk", how="left")


def print_slate_weather_table(slate: pd.DataFrame) -> None:
    if slate.empty:
        print("\nNo slate / weather context available.")
        return

    display = slate.copy()

    display["commence_time_local"] = pd.to_datetime(
        display["commence_time"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert("America/Los_Angeles")

    display["game_time"] = (
        display["commence_time_local"]
        .dt.strftime("%I:%M %p")
        .str.lstrip("0")
    )

    display["matchup"] = (
        display["away_team"].fillna("")
        + " @ "
        + display["home_team"].fillna("")
    )

    # ---------- ROUND NUMERIC COLUMNS ----------
    for c in [
        "temp_f",
        "wind_speed_mph",
        "park_factor_static",
        "park_factor_dynamic",
        "park_factor",
        "away_era",
        "home_era",
    ]:
        if c in display.columns:
            display[c] = pd.to_numeric(
                display[c],
                errors="coerce"
            ).round(2)

    # ---------- HR ALLOWED AS WHOLE NUMBERS ----------
    for c in [
        "away_hr_allowed",
        "home_hr_allowed",
    ]:
        if c in display.columns:
            display[c] = (
                pd.to_numeric(display[c], errors="coerce")
                .fillna(0)
                .astype(int)
            )

    cols = [
        "game_time",
        "matchup",
        "away_pitcher_name",
        "away_era",
        "away_hr_allowed",
        "home_pitcher_name",
        "home_era",
        "home_hr_allowed",
        "temp_f",
        "wind_speed_mph",
        "wind_out_flag",
        "park_factor_dynamic",
    ]
    cols = [c for c in cols if c in display.columns]
    if "park_factor_dynamic" in display.columns and "park_factor_static" in display.columns:
        display["park_factor_delta"] = (display["park_factor_dynamic"] - display["park_factor_static"]).round(3)
        if "park_factor_dynamic" in cols:
            insert_at = cols.index("park_factor_dynamic") + 1
            cols.insert(insert_at, "park_factor_delta")

    print("\n=== PARK CONDITIONS / SLATE CONTEXT ===")
    print(
        display.sort_values(["commence_time_local", "matchup"])[cols].to_string(index=False)
    )


def get_team_id_map() -> Dict[str, int]:
    payload = get_json("https://statsapi.mlb.com/api/v1/teams", params={"sportId": 1})
    if not payload:
        return {}
    return {t["name"]: int(t["id"]) for t in payload.get("teams", [])}


def get_active_roster_ids(team_id: int) -> List[int]:
    payload = get_json(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster", params={"rosterType": "active"})
    if not payload:
        return []
    ids = []
    for p in payload.get("roster", []):
        pid = p.get("person", {}).get("id")
        if pid is not None:
            ids.append(int(pid))
    return ids


def build_latest_batter_snapshot(model_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "batter", "batter_name", "batter_name_norm", "batter_hand",
        "batter_games_prior", "batter_pa_prior", "batter_hr_rate_prior",
        "batter_barrel_rate_prior", "batter_hard_hit_rate_prior", "batter_avg_ev_prior",
        "batter_recent_hr_rate_10", "batter_recent_hr_rate_20",
        "batter_recent_barrel_rate_10", "batter_recent_hard_hit_rate_10",
        "batter_recent_avg_ev_10", "batter_recent_pa_10", "batter_power_score_prior",
        "log_batter_pa_prior"
    ]
    latest = (
        model_df.sort_values(["batter", "game_date", "game_pk"])
        .groupby("batter", as_index=False)
        .tail(1)
        .copy()
    )
    return latest[keep_cols].drop_duplicates("batter")


def build_latest_pitcher_snapshot(model_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "pitcher", "pitcher_hand", "pitcher_games_prior", "pitcher_pa_prior",
        "pitcher_hr_rate_allowed_prior", "pitcher_barrel_rate_allowed_prior",
        "pitcher_hard_hit_rate_allowed_prior", "pitcher_avg_ev_allowed_prior",
        "pitcher_k_rate_prior", "pitcher_recent_k_rate_5", "pitcher_recent_k_rate_10",
        "pitcher_fb_rate_allowed_prior", "pitcher_gb_rate_allowed_prior",
        "pitcher_recent_fb_rate_allowed_5", "pitcher_recent_fb_rate_allowed_10",
        "pitcher_recent_gb_rate_allowed_5", "pitcher_recent_gb_rate_allowed_10",
        "pitcher_recent_hr_allowed_rate_10", "pitcher_recent_hr_allowed_rate_20",
        "pitcher_recent_barrel_allowed_rate_10", "pitcher_recent_avg_ev_allowed_10",
        "pitcher_damage_score_prior", "log_pitcher_pa_prior"
    ]
    latest = (
        model_df.sort_values(["pitcher", "game_date", "game_pk"])
        .groupby("pitcher", as_index=False)
        .tail(1)
        .copy()
    )
    return latest[keep_cols].drop_duplicates("pitcher")


def build_latest_batter_hand_snapshot(model_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "batter", "starter_pitcher_hand",
        "batter_pa_vs_hand_prior", "batter_hr_rate_vs_hand_prior",
        "batter_barrel_rate_vs_hand_prior", "batter_hard_hit_rate_vs_hand_prior",
        "batter_avg_ev_vs_hand_prior"
    ]
    latest = (
        model_df.sort_values(["batter", "starter_pitcher_hand", "game_date", "game_pk"])
        .groupby(["batter", "starter_pitcher_hand"], as_index=False)
        .tail(1)
        .copy()
    )
    return latest[cols].drop_duplicates(["batter", "starter_pitcher_hand"])


def build_latest_pitcher_hand_snapshot(model_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "pitcher", "batter_hand",
        "pitcher_pa_vs_batter_hand_prior", "pitcher_hr_rate_vs_batter_hand_prior",
        "pitcher_barrel_rate_vs_batter_hand_prior", "pitcher_hard_hit_rate_vs_batter_hand_prior",
        "pitcher_avg_ev_vs_batter_hand_prior"
    ]
    latest = (
        model_df.sort_values(["pitcher", "batter_hand", "game_date", "game_pk"])
        .groupby(["pitcher", "batter_hand"], as_index=False)
        .tail(1)
        .copy()
    )
    return latest[cols].drop_duplicates(["pitcher", "batter_hand"])


def build_latest_pitch_fit_snapshot(model_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["batter", "pitcher", "pitch_fit_score_prior", "pitch_fit_coverage_prior"]
    latest = (
        model_df.sort_values(["batter", "pitcher", "game_date", "game_pk"])
        .groupby(["batter", "pitcher"], as_index=False)
        .tail(1)
        .copy()
    )
    return latest[cols].drop_duplicates(["batter", "pitcher"])


def build_latest_matchup_snapshot(model_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "batter", "pitcher",
        "matchup_games_prior", "matchup_pa_prior", "matchup_hr_prior",
        "matchup_hr_rate_prior", "matchup_barrel_rate_prior", "matchup_hard_hit_rate_prior",
        "matchup_avg_ev_prior", "matchup_recent_hr_rate_3", "matchup_recent_hr_rate_5",
        "matchup_recent_barrel_rate_3", "matchup_recent_avg_ev_3", "matchup_recent_pa_3",
        "matchup_history_score_prior",
    ]
    latest = (
        model_df.sort_values(["batter", "pitcher", "game_date", "game_pk"])
        .groupby(["batter", "pitcher"], as_index=False)
        .tail(1)
        .copy()
    )
    return latest[cols].drop_duplicates(["batter", "pitcher"])


def build_forward_board_input(model_df: pd.DataFrame, pa_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    slate = get_daily_slate(target_date)
    if slate.empty:
        return pd.DataFrame()

    slate = add_real_weather_and_park(slate)

    # Add official season-to-date probable pitcher context to the slate table.
    season_pitcher_stats = build_current_season_pitcher_stats(pa_df, target_date)
    if not season_pitcher_stats.empty:
        away_stats = season_pitcher_stats.rename(columns={
            "pitcher": "away_pitcher_id",
            "season_era": "away_era",
            "season_hr_allowed": "away_hr_allowed",
        })
        home_stats = season_pitcher_stats.rename(columns={
            "pitcher": "home_pitcher_id",
            "season_era": "home_era",
            "season_hr_allowed": "home_hr_allowed",
        })

        slate["away_pitcher_id"] = pd.to_numeric(slate["away_pitcher_id"], errors="coerce").astype("Int64")
        slate["home_pitcher_id"] = pd.to_numeric(slate["home_pitcher_id"], errors="coerce").astype("Int64")
        away_stats["away_pitcher_id"] = pd.to_numeric(away_stats["away_pitcher_id"], errors="coerce").astype("Int64")
        home_stats["home_pitcher_id"] = pd.to_numeric(home_stats["home_pitcher_id"], errors="coerce").astype("Int64")

        slate = slate.merge(away_stats, on="away_pitcher_id", how="left")
        slate = slate.merge(home_stats, on="home_pitcher_id", how="left")

        matched_away = int(slate["away_era"].notna().sum()) if "away_era" in slate.columns else 0
        matched_home = int(slate["home_era"].notna().sum()) if "home_era" in slate.columns else 0
        print(f"Slate pitcher stat matches: away={matched_away}/{len(slate)}, home={matched_home}/{len(slate)}")
    else:
        slate["away_era"] = np.nan
        slate["away_hr_allowed"] = np.nan
        slate["home_era"] = np.nan
        slate["home_hr_allowed"] = np.nan

    slate["commence_time"] = pd.to_datetime(slate["commence_time"], utc=True, errors="coerce")
    slate = slate.sort_values(["commence_time", "away_team", "home_team"]).reset_index(drop=True)

    print_slate_weather_table(slate)

    team_id_map = get_team_id_map()
    batter_snapshot = build_latest_batter_snapshot(model_df)
    pitcher_snapshot = build_latest_pitcher_snapshot(model_df)
    batter_hand_snapshot = build_latest_batter_hand_snapshot(model_df)
    pitcher_hand_snapshot = build_latest_pitcher_hand_snapshot(model_df)
    batter_pitch_snapshot = build_latest_batter_pitch_group_snapshot(pa_df)
    pitcher_pitch_snapshot = build_latest_pitcher_pitch_group_snapshot(pa_df)
    matchup_snapshot = build_latest_matchup_snapshot(model_df)

    rows = []
    for _, game in slate.iterrows():
        away_team = game["away_team"]
        home_team = game["home_team"]

        away_team_id = team_id_map.get(away_team)
        home_team_id = team_id_map.get(home_team)
        if away_team_id is None or home_team_id is None:
            continue

        away_roster = get_active_roster_ids(away_team_id)
        home_roster = get_active_roster_ids(home_team_id)

        away_hitters = batter_snapshot[batter_snapshot["batter"].isin(away_roster)].copy()
        home_hitters = batter_snapshot[batter_snapshot["batter"].isin(home_roster)].copy()

        away_hitters["batter_recent_pa_10"] = pd.to_numeric(away_hitters["batter_recent_pa_10"], errors="coerce").fillna(0)
        home_hitters["batter_recent_pa_10"] = pd.to_numeric(home_hitters["batter_recent_pa_10"], errors="coerce").fillna(0)

        away_hitters = away_hitters[
            away_hitters["batter_recent_pa_10"] >= MIN_FORWARD_RECENT_PA_10
        ].copy()
        home_hitters = home_hitters[
            home_hitters["batter_recent_pa_10"] >= MIN_FORWARD_RECENT_PA_10
        ].copy()

        away_hitters = away_hitters.sort_values(
            ["batter_recent_pa_10", "batter_hr_rate_prior", "batter_power_score_prior"],
            ascending=False
        ).head(MAX_BATTERS_PER_TEAM)

        home_hitters = home_hitters.sort_values(
            ["batter_recent_pa_10", "batter_hr_rate_prior", "batter_power_score_prior"],
            ascending=False
        ).head(MAX_BATTERS_PER_TEAM)

        home_pitcher_id = game.get("home_pitcher_id")
        away_pitcher_id = game.get("away_pitcher_id")

        if pd.notna(home_pitcher_id):
            home_pitcher_id = int(home_pitcher_id)
            hp = pitcher_snapshot[pitcher_snapshot["pitcher"] == home_pitcher_id]
            if not hp.empty:
                hp_row = hp.iloc[0]
                for _, hitter in away_hitters.iterrows():
                    rows.append({
                        "target_date": target_date,
                        "game_pk": game["game_pk"],
                        "commence_time": game["commence_time"],
                        "batter": hitter["batter"],
                        "batter_name": hitter["batter_name"],
                        "batter_name_norm": hitter["batter_name_norm"],
                        "batter_hand": hitter["batter_hand"],
                        "pitcher": home_pitcher_id,
                        "pitcher_name": game.get("home_pitcher_name"),
                        "starter_pitcher_hand": hp_row["pitcher_hand"],
                        "batting_team": away_team,
                        "fielding_team": home_team,
                        "venue_team": home_team,
                        "is_home_batter": 0,
                        "park_factor": game["park_factor"],
                        "temp_f": game["temp_f"],
                        "wind_speed_mph": game["wind_speed_mph"],
                        "wind_direction_deg": game["wind_direction_deg"],
                        "relative_humidity": game["relative_humidity"],
                        "is_roofed_no_wind": game.get("is_roofed_no_wind", 0),
                        "weather_blowing_out": game.get("weather_blowing_out", game["wind_out_flag"]),
                        "wind_out_flag": game["wind_out_flag"],
                        "lf_bearing": game.get("lf_bearing", np.nan),
                        "cf_bearing": game.get("cf_bearing", np.nan),
                        "rf_bearing": game.get("rf_bearing", np.nan),
                        "wind_to_lf_mph": game.get("wind_to_lf_mph", np.nan),
                        "wind_to_cf_mph": game.get("wind_to_cf_mph", np.nan),
                        "wind_to_rf_mph": game.get("wind_to_rf_mph", np.nan),
                        "unique_pitchers_faced": 1,
                        **hitter.drop(labels=["batter", "batter_name", "batter_name_norm", "batter_hand"]).to_dict(),
                        **hp_row.drop(labels=["pitcher", "pitcher_hand"]).to_dict(),
                    })

        if pd.notna(away_pitcher_id):
            away_pitcher_id = int(away_pitcher_id)
            ap = pitcher_snapshot[pitcher_snapshot["pitcher"] == away_pitcher_id]
            if not ap.empty:
                ap_row = ap.iloc[0]
                for _, hitter in home_hitters.iterrows():
                    rows.append({
                        "target_date": target_date,
                        "game_pk": game["game_pk"],
                        "commence_time": game["commence_time"],
                        "batter": hitter["batter"],
                        "batter_name": hitter["batter_name"],
                        "batter_name_norm": hitter["batter_name_norm"],
                        "batter_hand": hitter["batter_hand"],
                        "pitcher": away_pitcher_id,
                        "pitcher_name": game.get("away_pitcher_name"),
                        "starter_pitcher_hand": ap_row["pitcher_hand"],
                        "batting_team": home_team,
                        "fielding_team": away_team,
                        "venue_team": home_team,
                        "is_home_batter": 1,
                        "park_factor": game["park_factor"],
                        "temp_f": game["temp_f"],
                        "wind_speed_mph": game["wind_speed_mph"],
                        "wind_direction_deg": game["wind_direction_deg"],
                        "relative_humidity": game["relative_humidity"],
                        "is_roofed_no_wind": game.get("is_roofed_no_wind", 0),
                        "weather_blowing_out": game.get("weather_blowing_out", game["wind_out_flag"]),
                        "wind_out_flag": game["wind_out_flag"],
                        "lf_bearing": game.get("lf_bearing", np.nan),
                        "cf_bearing": game.get("cf_bearing", np.nan),
                        "rf_bearing": game.get("rf_bearing", np.nan),
                        "wind_to_lf_mph": game.get("wind_to_lf_mph", np.nan),
                        "wind_to_cf_mph": game.get("wind_to_cf_mph", np.nan),
                        "wind_to_rf_mph": game.get("wind_to_rf_mph", np.nan),
                        "unique_pitchers_faced": 1,
                        **hitter.drop(labels=["batter", "batter_name", "batter_name_norm", "batter_hand"]).to_dict(),
                        **ap_row.drop(labels=["pitcher", "pitcher_hand"]).to_dict(),
                    })

    board = pd.DataFrame(rows)
    if board.empty:
        return board

    board["platoon_advantage"] = (
        (board["batter_hand"].isin(["L", "R"]))
        & (board["starter_pitcher_hand"].isin(["L", "R"]))
        & (board["batter_hand"] != board["starter_pitcher_hand"])
    ).astype(int)
    board.loc[board["batter_hand"] == "B", "platoon_advantage"] = 1

    board = board.merge(
        batter_hand_snapshot,
        on=["batter", "starter_pitcher_hand"],
        how="left"
    )
    board = board.merge(
        pitcher_hand_snapshot,
        on=["pitcher", "batter_hand"],
        how="left"
    )
    board = compute_forward_pitch_fit(board, batter_pitch_snapshot, pitcher_pitch_snapshot)

    board = board.merge(
        matchup_snapshot,
        on=["batter", "pitcher"],
        how="left"
    )

    fill_zero_cols = [
        "batter_pa_vs_hand_prior", "batter_hr_rate_vs_hand_prior",
        "batter_barrel_rate_vs_hand_prior", "batter_hard_hit_rate_vs_hand_prior",
        "batter_avg_ev_vs_hand_prior",
        "pitcher_pa_vs_batter_hand_prior", "pitcher_hr_rate_vs_batter_hand_prior",
        "pitcher_barrel_rate_vs_batter_hand_prior", "pitcher_hard_hit_rate_vs_batter_hand_prior",
        "pitcher_avg_ev_vs_batter_hand_prior",
        "pitch_fit_score_prior", "pitch_fit_coverage_prior",
        "matchup_games_prior", "matchup_pa_prior", "matchup_hr_prior",
        "matchup_hr_rate_prior", "matchup_barrel_rate_prior", "matchup_hard_hit_rate_prior",
        "matchup_avg_ev_prior", "matchup_recent_hr_rate_3", "matchup_recent_hr_rate_5",
        "matchup_recent_barrel_rate_3", "matchup_recent_avg_ev_3", "matchup_recent_pa_3",
        "matchup_history_score_prior"
    ]
    for c in fill_zero_cols:
        if c in board.columns:
            board[c] = pd.to_numeric(board[c], errors="coerce").fillna(0)

    board = add_pull_wind_features(board)

    board["interaction_hr_rates"] = board["batter_hr_rate_prior"] * board["pitcher_hr_rate_allowed_prior"]
    board["interaction_barrel_rates"] = board["batter_barrel_rate_prior"] * board["pitcher_barrel_rate_allowed_prior"]
    board["interaction_power_damage"] = board["batter_power_score_prior"] * board["pitcher_damage_score_prior"]
    board["interaction_power_vs_k"] = board["batter_power_score_prior"] * (1 - board["pitcher_recent_k_rate_10"].clip(0, 0.5))
    board["interaction_power_vs_elevation"] = board["batter_power_score_prior"] * (
        board["pitcher_recent_fb_rate_allowed_10"].fillna(0) - board["pitcher_recent_gb_rate_allowed_10"].fillna(0)
    )
    board["interaction_matchup_pitchfit"] = board["matchup_history_score_prior"] * board["pitch_fit_score_prior"]
    board["interaction_matchup_power"] = board["matchup_history_score_prior"] * board["batter_power_score_prior"]

    for c in FEATURE_COLUMNS:
        if c not in board.columns:
            board[c] = 0
        board[c] = pd.to_numeric(board[c], errors="coerce").fillna(0)

    return board


# =========================================================
# 7) FORWARD BOARD HR CHECK
# =========================================================
def get_actual_home_runs_for_date(target_date: str, include_in_progress: bool = False) -> pd.DataFrame:
    print(f"\nPulling official MLB HR results for {target_date} ...")

    schedule_url = "https://statsapi.mlb.com/api/v1/schedule"
    schedule_params = {
        "sportId": 1,
        "date": target_date,
        "hydrate": "linescore",
    }
    schedule_payload = get_json(schedule_url, params=schedule_params)

    if not schedule_payload or not schedule_payload.get("dates"):
        return pd.DataFrame(columns=[
            "batter", "actual_hr_name", "actual_home_run", "actual_hr_count",
            "game_pk", "game_status"
        ])

    final_statuses = {
        "Final",
        "Game Over",
        "Completed Early",
        "Completed Early: Rain",
    }

    hr_rows = []
    for date_block in schedule_payload.get("dates", []):
        for game in date_block.get("games", []):
            game_pk = game.get("gamePk")
            status_text = (
                game.get("status", {}).get("detailedState")
                or game.get("status", {}).get("abstractGameState")
                or "Unknown"
            )

            if (not include_in_progress) and (status_text not in final_statuses):
                continue

            boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
            boxscore = get_json(boxscore_url)
            if not boxscore:
                continue

            teams = boxscore.get("teams", {})
            for side in ["home", "away"]:
                team_players = teams.get(side, {}).get("players", {})
                for _, player_blob in team_players.items():
                    person = player_blob.get("person", {})
                    stats = player_blob.get("stats", {}).get("batting", {})
                    hr_count = stats.get("homeRuns", 0)
                    try:
                        hr_count = int(hr_count)
                    except Exception:
                        hr_count = 0

                    if hr_count > 0 and person.get("id") is not None:
                        hr_rows.append({
                            "batter": int(person.get("id")),
                            "actual_hr_name": person.get("fullName"),
                            "actual_home_run": 1,
                            "actual_hr_count": hr_count,
                            "game_pk": game_pk,
                            "game_status": status_text,
                        })

    if not hr_rows:
        return pd.DataFrame(columns=[
            "batter", "actual_hr_name", "actual_home_run", "actual_hr_count",
            "game_pk", "game_status"
        ])

    actual_hr = pd.DataFrame(hr_rows)
    actual_hr = (
        actual_hr.groupby("batter", as_index=False)
        .agg(
            actual_hr_name=("actual_hr_name", "first"),
            actual_home_run=("actual_home_run", "max"),
            actual_hr_count=("actual_hr_count", "sum"),
        )
    )
    return actual_hr


def attach_actual_hr_results_to_board(board: pd.DataFrame, target_date: str) -> pd.DataFrame:
    if board.empty:
        return board

    out = board.copy()
    out["batter"] = pd.to_numeric(out["batter"], errors="coerce")
    out = out.dropna(subset=["batter"]).copy()
    out["batter"] = out["batter"].astype(int)

    actual_hr = get_actual_home_runs_for_date(target_date)
    if actual_hr.empty:
        out["actual_home_run"] = 0
        out["actual_hr_count"] = 0
        out["actual_hr_name"] = pd.NA
        return out

    out = out.merge(
        actual_hr[["batter", "actual_hr_name", "actual_home_run", "actual_hr_count"]],
        on="batter",
        how="left",
    )
    out["actual_home_run"] = out["actual_home_run"].fillna(0).astype(int)
    out["actual_hr_count"] = out["actual_hr_count"].fillna(0).astype(int)
    return out


def summarize_forward_board_hits(board: pd.DataFrame, top_n: int) -> dict:
    ranked = board.sort_values(_sort_col(board), ascending=False).drop_duplicates(subset=["batter"]).head(top_n).copy()
    players = len(ranked)
    homers = int(ranked["actual_home_run"].sum()) if players else 0
    total_hrs = int(ranked["actual_hr_count"].sum()) if players else 0
    return {
        "target_date": ranked["target_date"].iloc[0] if players and "target_date" in ranked.columns else None,
        "top_n": top_n,
        "players": players,
        "homers": homers,
        "total_hrs": total_hrs,
        "hit_rate": float(homers / players) if players else np.nan,
        "avg_model_prob": float(ranked["pred_hr_prob"].mean()) if players else np.nan,
    }


def run_forward_hr_check(board: pd.DataFrame, target_date: str) -> pd.DataFrame:
    if board.empty:
        print("\nNo forward board available to grade.")
        return board

    graded = attach_actual_hr_results_to_board(board, target_date)
    graded = graded.sort_values(_sort_col(graded), ascending=False).drop_duplicates(subset=["batter"]).copy()
    graded["board_rank"] = np.arange(1, len(graded) + 1)
    if "final_hr_probability" not in graded.columns:
        graded["final_hr_probability"] = (graded["pred_hr_prob"] * 100).round(1)
    if "calibrated_hr_prob" in graded.columns and "calibrated_hr_probability" not in graded.columns:
        graded["calibrated_hr_probability"] = (graded["calibrated_hr_prob"] * 100).round(1)
    if "game_matchup" not in graded.columns:
        graded["game_matchup"] = graded.apply(
            lambda x: " vs. ".join(sorted([str(x["batting_team"]), str(x["fielding_team"])])),
            axis=1,
        )

    hr_hitters = graded[graded["actual_home_run"] == 1].copy()

    print("\n=== PLAYERS ON YOUR FORWARD BOARD WHO HOMERED ===")
    if hr_hitters.empty:
        print("None of the players on your forward board were graded with a HR.")
    else:
        display_cols = [
            "board_rank", "batter_name_hand", "actual_hr_count", "game_matchup",
            "pitcher_name_hand", "final_hr_probability"
        ]
        display_cols = [c for c in display_cols if c in hr_hitters.columns]
        print(
            hr_hitters[display_cols]
            .sort_values(["board_rank", "actual_hr_count"], ascending=[True, False])
            .to_string(index=False)
        )

    top25_hr_hitters = hr_hitters[hr_hitters["board_rank"] <= TOP_N_OUTPUT].copy()
    print("\n=== TOP 25 PICKS WHO HOMERED ===")
    if top25_hr_hitters.empty:
        print("None of the top 25 picks homered.")
    else:
        print(
            top25_hr_hitters[[
                "board_rank", "batter_name_hand", "actual_hr_count", "game_matchup",
                "pitcher_name_hand", "final_hr_probability"
            ]]
            .sort_values("board_rank", ascending=True)
            .to_string(index=False)
        )

    summaries = [summarize_forward_board_hits(graded, n) for n in HR_CHECK_TOP_NS]
    summary_df = pd.DataFrame(summaries)

    print("\n=== FORWARD BOARD HR CHECK SUMMARY ===")
    print(summary_df.to_string(index=False))

    graded_path = f"trained_hr_model_board_graded_{target_date}.csv"
    summary_path = f"trained_hr_model_board_hr_check_summary_{target_date}.csv"
    graded.to_csv(graded_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(f" - {graded_path}")
    print(f" - {summary_path}")

    return graded


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n=== REVERTED ORIGINAL WEIGHTS VERSION (NO DK) ===")
    validate_team_context()

    pa_df = load_statcast_pa(
        FULL_DATA_START_DATE,
        FULL_DATA_END_DATE,
        use_cache=USE_CACHE,
        refresh_cache=REFRESH_CACHE,
    )

    model_df = build_model_dataset(pa_df)
    train_df, valid_df, test_df = split_model_dataset(model_df)

    if train_df.empty or valid_df.empty:
        print("\nNeed non-empty train and validation sets. Adjust split dates.")
        return

    global ACTIVE_FEATURE_COLUMNS

    # First fit uses the full engineered feature set so we can audit true validation value.
    ACTIVE_FEATURE_COLUMNS = FULL_FEATURE_COLUMNS.copy()
    model, calibrator = fit_calibrated_hgb(train_df, valid_df)

    if RUN_FEATURE_AUDIT:
        selected_features = run_feature_audit_and_select(
            model=model,
            train_df=train_df,
            valid_df=valid_df,
            feature_cols=FULL_FEATURE_COLUMNS,
        )
        if APPLY_FEATURE_PRUNING:
            ACTIVE_FEATURE_COLUMNS = selected_features
            print("\nRefitting final model with audited/pruned feature set...")
            model, calibrator = fit_calibrated_hgb(train_df, valid_df)

    print_feature_importance(model)

    if RUN_BACKTEST:
        run_backtest(model, calibrator, test_df)

    if RUN_FORWARD_BOARD:
        board = build_forward_board_input(model_df, pa_df, TARGET_DATE)
        if board.empty:
            print("\nNo forward board could be built.")
            return

        board["raw_hr_prob"] = predict_raw(model, board)
        board["calibrated_hr_prob"] = predict_calibrated(model, calibrator, board)
        # Rank and display final probability using raw probability to avoid isotonic bucket ties.
        board["pred_hr_prob"] = board["raw_hr_prob"]
        board = add_macro_board_columns(board)
        board = board.sort_values(_sort_col(board), ascending=False).copy()
        board["ranking"] = np.arange(1, len(board) + 1)

        print_forward_board_tables(board, TOP_N_OUTPUT)

        board.to_csv(f"trained_hr_model_board_{TARGET_DATE}.csv", index=False)
        print(f"\nSaved: trained_hr_model_board_{TARGET_DATE}.csv")

        if RUN_HR_CHECK:
            run_forward_hr_check(board, TARGET_DATE)


if __name__ == "__main__":
    main()
