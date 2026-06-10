from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_FEATURE_SET,
    DEFAULT_MODEL_FILE,
    DEFAULT_ODDS_FILE,
    DEFAULT_PREDICTIONS_FILE,
    DEFAULT_TRAINING_FILE,
    RAW_DIR,
)
from .data_sources import fetch_schedule, fetch_weather_for_game, read_manual_odds
from .features import latest_batter_profiles, latest_pitcher_profiles, resolve_feature_columns
from .model import load_model, predict_probabilities
from .parks import park_info
from .utils import american_to_implied_probability, confidence_tier, ensure_parent


def predict_for_date(
    game_date: str | None = None,
    training_path: Path = DEFAULT_TRAINING_FILE,
    model_path: Path = DEFAULT_MODEL_FILE,
    odds_path: Path = DEFAULT_ODDS_FILE,
    output_path: Path = DEFAULT_PREDICTIONS_FILE,
) -> Path:
    game_date = game_date or date.today().isoformat()
    training = pd.read_csv(training_path, parse_dates=["game_date"])
    model, feature_columns = load_model(model_path)
    schedule = _load_schedule_for_date(game_date)
    candidates = build_daily_candidates(training, schedule, feature_columns=feature_columns)
    if candidates.empty:
        return output_path

    candidates["hr_probability"] = predict_probabilities(model, candidates, feature_columns=feature_columns)
    candidates["feature_set"] = _feature_set_name(training, feature_columns)
    candidates["confidence_tier"] = candidates["hr_probability"].map(confidence_tier)
    candidates = attach_odds(candidates, game_date, odds_path)
    candidates["rank"] = candidates["hr_probability"].rank(ascending=False, method="first").astype(int)
    candidates = candidates.sort_values(["rank", "player_name"])
    candidates.to_csv(ensure_parent(output_path), index=False)
    latest = output_path.parent / "latest_predictions.csv"
    if latest != output_path:
        candidates.to_csv(ensure_parent(latest), index=False)
    return output_path


def _load_schedule_for_date(game_date: str) -> list[dict]:
    schedule_path = RAW_DIR / f"schedule_{game_date}.json"
    if schedule_path.exists():
        return json.loads(schedule_path.read_text(encoding="utf-8"))
    try:
        return fetch_schedule(game_date)
    except Exception:
        cached_schedules = sorted(RAW_DIR.glob("schedule_*.json"))
        if not cached_schedules:
            raise
        return json.loads(cached_schedules[-1].read_text(encoding="utf-8"))


def build_daily_candidates(
    training: pd.DataFrame, schedule: list[dict], feature_columns: list[str] | None = None
) -> pd.DataFrame:
    feature_columns = feature_columns or resolve_feature_columns(DEFAULT_FEATURE_SET)
    batter_profiles = latest_batter_profiles(training)
    pitcher_profiles = latest_pitcher_profiles(training)
    rows = []
    for game in schedule:
        weather = fetch_weather_for_game(game)
        venue_name = game.get("venue_name")
        park_factor = park_info(venue_name)["hr_factor"]
        matchups = [
            {
                "team": game.get("away_team"),
                "opponent": game.get("home_team"),
                "opposing_pitcher_id": game.get("home_probable_pitcher_id"),
                "opposing_pitcher": game.get("home_probable_pitcher"),
                "is_home": 0,
            },
            {
                "team": game.get("home_team"),
                "opponent": game.get("away_team"),
                "opposing_pitcher_id": game.get("away_probable_pitcher_id"),
                "opposing_pitcher": game.get("away_probable_pitcher"),
                "is_home": 1,
            },
        ]
        for matchup in matchups:
            team_hitters = batter_profiles[batter_profiles["team"] == matchup["team"]].copy()
            team_hitters = team_hitters.sort_values("batter_pa_30", ascending=False).head(12)
            pitcher = pitcher_profiles[pitcher_profiles["pitcher"] == matchup["opposing_pitcher_id"]]
            pitcher_features = _default_pitcher_features() if pitcher.empty else pitcher.iloc[0].to_dict()
            for _, hitter in team_hitters.iterrows():
                row = hitter.to_dict()
                row.update(
                    {
                        "game_date": game.get("game_date"),
                        "game_pk": game.get("game_pk"),
                        "venue_name": venue_name,
                        "opponent": matchup["opponent"],
                        "opposing_pitcher": matchup["opposing_pitcher"] or "TBD",
                        "opposing_pitcher_id": matchup["opposing_pitcher_id"],
                        "is_home": matchup["is_home"],
                        "park_hr_factor": park_factor,
                        "temperature_2m": weather["temperature_2m"],
                        "wind_speed_10m": weather["wind_speed_10m"],
                        **{k: pitcher_features.get(k, 0.0) for k in pitcher_features if k.startswith("pitcher_")},
                        "p_throws": pitcher_features.get("p_throws", "U"),
                    }
                )
                row.update(_matchup_feature_values(row))
                rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    for col in feature_columns:
        if col not in candidates.columns:
            candidates[col] = 0.0
    candidates[feature_columns] = candidates[feature_columns].fillna(0.0)
    candidates["matchup_note"] = candidates.apply(_matchup_note, axis=1)
    return candidates


def attach_odds(candidates: pd.DataFrame, game_date: str, odds_path: Path) -> pd.DataFrame:
    out = candidates.copy()
    odds = read_manual_odds(odds_path)
    odds = odds[odds["date"].astype(str) == game_date]
    if odds.empty:
        out["american_odds"] = pd.NA
        out["book"] = pd.NA
        out["implied_probability"] = pd.NA
        out["value_flag"] = False
        return out
    merged = out.merge(odds, on="player_name", how="left")
    merged["implied_probability"] = merged["american_odds"].map(
        lambda x: american_to_implied_probability(x) if pd.notna(x) else pd.NA
    )
    merged["value_flag"] = merged.apply(
        lambda row: bool(pd.notna(row["implied_probability"]) and row["hr_probability"] > row["implied_probability"]),
        axis=1,
    )
    return merged


def _default_pitcher_features() -> dict[str, float]:
    return {
        "pitcher_bf_30": 0.0,
        "pitcher_hr_rate_allowed_30": 0.03,
        "pitcher_barrel_rate_allowed_30": 0.08,
        "pitcher_hardhit_rate_allowed_30": 0.38,
        "pitcher_k_rate_30": 0.0,
        "pitcher_bb_rate_30": 0.0,
        "pitcher_fip_proxy_30": 0.0,
        "pitcher_flyball_rate_allowed_30": 0.0,
        "pitcher_hr_per_flyball_allowed_30": 0.0,
        "pitcher_avg_release_speed_30": 0.0,
        "pitcher_max_release_speed_30": 0.0,
        "pitcher_avg_spin_rate_30": 0.0,
        "pitcher_avg_extension_30": 0.0,
        "pitcher_avg_abs_movement_30": 0.0,
        "pitcher_fastball_rate_30": 0.0,
        "p_throws": "U",
    }


def _matchup_note(row: pd.Series) -> str:
    pieces = [
        f"{row['team']} vs {row['opponent']}",
        f"SP: {row.get('opposing_pitcher', 'TBD')}",
        f"barrel {row.get('batter_barrel_rate_30', 0):.1%}",
        f"park {row.get('park_hr_factor', 1):.2f}",
    ]
    if pd.notna(row.get("temperature_2m")):
        pieces.append(f"{row['temperature_2m']:.0f}F")
    return " | ".join(pieces)


def _matchup_feature_values(row: dict) -> dict[str, int]:
    stand = str(row.get("stand", "U"))
    throws = str(row.get("p_throws", "U"))
    return {
        "batter_bats_right": int(stand == "R"),
        "batter_bats_left": int(stand == "L"),
        "pitcher_throws_right": int(throws == "R"),
        "pitcher_throws_left": int(throws == "L"),
        "platoon_advantage": int((stand == "R" and throws == "L") or (stand == "L" and throws == "R")),
    }


def _feature_set_name(training: pd.DataFrame, feature_columns: list[str]) -> str:
    if "feature_set" in training.columns and training["feature_set"].notna().any():
        return str(training["feature_set"].dropna().iloc[0])
    for name in ["baseline", "baseline+power_contact", "baseline+pitcher_vulnerability", "all_free_statcast"]:
        if resolve_feature_columns(name) == feature_columns:
            return name
    return "custom"
