from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_PREDICTIONS_FILE


@dataclass(frozen=True)
class FavoritePick:
    rank: int
    player_name: str
    team: str
    opponent: str
    opposing_pitcher: str
    hr_probability: float
    pick_score: float
    confidence_tier: str
    analyst_case: str
    bettor_lens: str
    fan_read: str


def favorite_three_picks(predictions_path: Path = DEFAULT_PREDICTIONS_FILE) -> list[FavoritePick]:
    if not predictions_path.exists():
        raise FileNotFoundError("Missing predictions. Run Predict Today first.")
    predictions = pd.read_csv(predictions_path)
    if predictions.empty:
        raise ValueError("Predictions file is empty. Run Predict Today first.")
    return favorite_three_from_predictions(predictions)


def favorite_three_from_predictions(predictions: pd.DataFrame) -> list[FavoritePick]:
    df = predictions.copy()
    if len(df) < 3:
        raise ValueError("Need at least 3 prediction candidates to generate favorite picks.")
    df["pick_score"] = _pick_score(df)
    viable = df[df["hr_probability"].fillna(0.0) >= 0.05].copy()
    if len(viable) < 3:
        viable = df.copy()
    selected = _select_diverse(viable.sort_values("pick_score", ascending=False), count=3)
    return [_to_pick(row, idx + 1) for idx, (_, row) in enumerate(selected.iterrows())]


def _pick_score(df: pd.DataFrame) -> pd.Series:
    probability = _scale(df, "hr_probability", 0.04, 0.20)
    power = (
        0.35 * _scale(df, "batter_barrel_rate_30", 0.04, 0.14)
        + 0.25 * _scale(df, "batter_hardhit_rate_30", 0.25, 0.45)
        + 0.25 * _scale(df, "batter_hr_rate_30", 0.02, 0.09)
        + 0.15 * _scale(df, "batter_max_ev_30", 98.0, 108.0)
    )
    pitcher = (
        0.45 * _scale(df, "pitcher_hr_rate_allowed_30", 0.015, 0.055)
        + 0.30 * _scale(df, "pitcher_barrel_rate_allowed_30", 0.02, 0.07)
        + 0.15 * _scale(df, "pitcher_hardhit_rate_allowed_30", 0.22, 0.36)
        + 0.10 * _scale(df, "pitcher_flyball_rate_allowed_30", 0.08, 0.22)
    )
    context = (
        0.45 * _scale(df, "park_hr_factor", 0.90, 1.10)
        + 0.35 * _scale(df, "temperature_2m", 55.0, 88.0)
        + 0.20 * _scale(df, "wind_speed_10m", 0.0, 18.0)
    )
    platoon = df.get("platoon_advantage", pd.Series(0, index=df.index)).fillna(0.0).astype(float).clip(0, 1)
    rank = df.get("rank", pd.Series(999, index=df.index)).fillna(999).astype(float)
    contrarian = ((rank > 10) & (rank <= 30)).astype(float) * (power >= 0.55).astype(float) * 0.06
    return 100 * (0.46 * probability + 0.26 * power + 0.16 * pitcher + 0.08 * context + 0.04 * platoon + contrarian)


def _select_diverse(sorted_df: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    selected = []
    team_counts: dict[str, int] = {}
    game_counts: dict[str, int] = {}
    for _, row in sorted_df.iterrows():
        team = str(row.get("team", ""))
        game = str(row.get("game_pk", ""))
        if team_counts.get(team, 0) >= 2:
            continue
        if game_counts.get(game, 0) >= 2:
            continue
        selected.append(row)
        team_counts[team] = team_counts.get(team, 0) + 1
        game_counts[game] = game_counts.get(game, 0) + 1
        if len(selected) == count:
            return pd.DataFrame(selected)
    for _, row in sorted_df.iterrows():
        if len(selected) == count:
            break
        if any(str(row.get("player_name")) == str(existing.get("player_name")) for existing in selected):
            continue
        selected.append(row)
    return pd.DataFrame(selected)


def _to_pick(row: pd.Series, pick_rank: int) -> FavoritePick:
    return FavoritePick(
        rank=pick_rank,
        player_name=str(row.get("player_name", "Unknown")),
        team=str(row.get("team", "")),
        opponent=str(row.get("opponent", "")),
        opposing_pitcher=str(row.get("opposing_pitcher", "TBD")),
        hr_probability=float(row.get("hr_probability", 0.0)),
        pick_score=float(row.get("pick_score", 0.0)),
        confidence_tier=str(row.get("confidence_tier", "")),
        analyst_case=_analyst_case(row),
        bettor_lens=_bettor_lens(row),
        fan_read=_fan_read(row),
    )


def _analyst_case(row: pd.Series) -> str:
    return (
        f"{_name(row)} grades at {_pct(row.get('hr_probability', 0.0))} with a "
        f"{_pct(row.get('batter_barrel_rate_30', 0.0))} barrel rate and "
        f"{_pct(row.get('batter_hardhit_rate_30', 0.0))} hard-hit rate over the recent window. "
        f"The matchup is {row.get('team')} vs {row.get('opponent')} against {row.get('opposing_pitcher', 'TBD')}, "
        f"with park factor {float(row.get('park_hr_factor', 1.0)):.2f} and "
        f"{float(row.get('temperature_2m', 70.0)):.0f}F weather."
    )


def _bettor_lens(row: pd.Series) -> str:
    probability = float(row.get("hr_probability", 0.0))
    implied = row.get("implied_probability")
    odds = row.get("american_odds")
    if pd.notna(implied) and pd.notna(odds):
        value = "a value lean" if probability > float(implied) else "not a clear value at the listed price"
        return f"Listed odds {odds} imply {_pct(implied)}; my model/pick blend has him at {_pct(probability)}, so this is {value}."
    if probability >= 0.14:
        return "This is a shorter-list power profile rather than a pure longshot; I would expect the market to notice it."
    if probability >= 0.09:
        return "This sits in the playable middle tier: not automatic, but enough power indicators to keep on the card."
    return "This is more of a longshot swing, so the price would need to be generous."


def _fan_read(row: pd.Series) -> str:
    notes = []
    if float(row.get("batter_barrel_rate_30", 0.0)) >= 0.09:
        notes.append("the barrel quality is there")
    if float(row.get("batter_hr_rate_30", 0.0)) >= 0.06:
        notes.append("the recent homer pace is real")
    if int(row.get("platoon_advantage", 0) or 0) == 1:
        notes.append("the handedness matchup helps")
    if float(row.get("pitcher_hr_rate_allowed_30", 0.0)) >= 0.035:
        notes.append("the opposing arm has been giving up damage")
    if not notes:
        notes.append("the overall power/matchup blend is stronger than the rank alone suggests")
    return f"My fan read: {_name(row)} is the kind of bat I want exposure to today because " + ", ".join(notes[:3]) + "."


def _scale(df: pd.DataFrame, col: str, low: float, high: float) -> pd.Series:
    values = df.get(col, pd.Series(low, index=df.index)).fillna(low).astype(float)
    return ((values - low) / (high - low)).clip(0, 1)


def _pct(value) -> str:
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "n/a"


def _name(row: pd.Series) -> str:
    return str(row.get("player_name", "This hitter"))

