from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hr_predictor.favorites import favorite_three_from_predictions, favorite_three_picks


def test_favorite_three_returns_exactly_three():
    picks = favorite_three_from_predictions(_predictions())

    assert len(picks) == 3


def test_favorite_three_not_always_top_three_probability():
    picks = favorite_three_from_predictions(_predictions())
    picked_names = [pick.player_name for pick in picks]

    assert picked_names != ["Top One", "Top Two", "Top Three"]
    assert "Power Four" in picked_names


def test_favorite_explanations_include_required_sections():
    pick = favorite_three_from_predictions(_predictions())[0]

    assert pick.analyst_case
    assert pick.bettor_lens
    assert pick.fan_read
    assert pick.team in pick.analyst_case
    assert pick.opposing_pitcher in pick.analyst_case


def test_missing_predictions_has_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Run Predict Today first"):
        favorite_three_picks(tmp_path / "missing.csv")


def _predictions() -> pd.DataFrame:
    rows = []
    for idx in range(1, 9):
        rows.append(
            {
                "rank": idx,
                "player_name": [
                    "Top One",
                    "Top Two",
                    "Top Three",
                    "Power Four",
                    "Power Five",
                    "Game Six",
                    "Game Seven",
                    "Game Eight",
                ][idx - 1],
                "team": ["AAA", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"][idx - 1],
                "opponent": "OPP",
                "game_pk": [1, 1, 2, 3, 4, 5, 6, 7][idx - 1],
                "opposing_pitcher": f"Pitcher {idx}",
                "hr_probability": [0.20, 0.19, 0.18, 0.16, 0.15, 0.11, 0.10, 0.09][idx - 1],
                "confidence_tier": "A",
                "batter_barrel_rate_30": [0.04, 0.05, 0.05, 0.15, 0.13, 0.08, 0.07, 0.06][idx - 1],
                "batter_hardhit_rate_30": [0.26, 0.27, 0.28, 0.47, 0.44, 0.33, 0.31, 0.30][idx - 1],
                "batter_hr_rate_30": [0.03, 0.03, 0.03, 0.09, 0.08, 0.04, 0.04, 0.04][idx - 1],
                "batter_max_ev_30": [99, 99, 99, 108, 106, 101, 100, 100][idx - 1],
                "pitcher_hr_rate_allowed_30": [0.02, 0.02, 0.02, 0.055, 0.05, 0.03, 0.03, 0.03][idx - 1],
                "pitcher_barrel_rate_allowed_30": [0.02, 0.02, 0.02, 0.07, 0.06, 0.03, 0.03, 0.03][idx - 1],
                "pitcher_hardhit_rate_allowed_30": 0.3,
                "pitcher_flyball_rate_allowed_30": 0.15,
                "park_hr_factor": 1.02,
                "temperature_2m": 78,
                "wind_speed_10m": 7,
                "platoon_advantage": 1 if idx in [4, 5] else 0,
                "implied_probability": pd.NA,
                "american_odds": pd.NA,
            }
        )
    return pd.DataFrame(rows)

