from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
LOGS_DIR = DATA_DIR / "logs"

STATCAST_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_TRAINING_FILE = PROCESSED_DIR / "training_rows.csv"
DEFAULT_MODEL_FILE = MODELS_DIR / "hr_model.joblib"
DEFAULT_PREDICTIONS_FILE = PREDICTIONS_DIR / "latest_predictions.csv"
DEFAULT_ODDS_FILE = RAW_DIR / "manual_hr_odds.csv"
DEFAULT_STATCAST_FILE = RAW_DIR / "statcast_history.csv"
DEFAULT_BACKTEST_FILE = PROCESSED_DIR / "backtest_predictions.csv"
DEFAULT_REFRESH_LOG = LOGS_DIR / "refresh.log"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

FEATURE_COLUMNS = [
    "batter_pa_30",
    "batter_hr_rate_30",
    "batter_barrel_rate_30",
    "batter_hardhit_rate_30",
    "batter_avg_ev_30",
    "batter_avg_la_30",
    "pitcher_bf_30",
    "pitcher_hr_rate_allowed_30",
    "pitcher_barrel_rate_allowed_30",
    "pitcher_hardhit_rate_allowed_30",
    "park_hr_factor",
    "is_home",
    "temperature_2m",
    "wind_speed_10m",
]

FEATURE_GROUPS = {
    "baseline": FEATURE_COLUMNS,
    "power_contact": [
        "batter_estimated_woba_30",
        "batter_estimated_slg_30",
        "batter_woba_30",
        "batter_iso_30",
        "batter_max_ev_30",
        "batter_sweet_spot_rate_30",
    ],
    "plate_discipline": [
        "batter_k_rate_30",
        "batter_bb_rate_30",
        "batter_swinging_strike_rate_30",
        "batter_contact_rate_30",
    ],
    "pitcher_vulnerability": [
        "pitcher_k_rate_30",
        "pitcher_bb_rate_30",
        "pitcher_fip_proxy_30",
        "pitcher_flyball_rate_allowed_30",
        "pitcher_hr_per_flyball_allowed_30",
    ],
    "matchup_shape": [
        "batter_bats_right",
        "batter_bats_left",
        "pitcher_throws_right",
        "pitcher_throws_left",
        "platoon_advantage",
    ],
    "pitch_quality_proxy": [
        "pitcher_avg_release_speed_30",
        "pitcher_max_release_speed_30",
        "pitcher_avg_spin_rate_30",
        "pitcher_avg_extension_30",
        "pitcher_avg_abs_movement_30",
        "pitcher_fastball_rate_30",
    ],
}

FEATURE_SET_ALIASES = {
    "all": "all_free_statcast",
}

DEFAULT_FEATURE_SET = "baseline"
