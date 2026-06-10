from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hr_predictor.experiments import run_feature_set_experiments


def test_run_feature_set_experiments_writes_separate_summaries(tmp_path: Path):
    source = tmp_path / "statcast.csv"
    _synthetic_statcast().to_csv(source, index=False)

    results = run_feature_set_experiments(
        ["baseline", "baseline+power_contact"],
        statcast_path=source,
        root=tmp_path / "experiments",
        min_train_days=10,
    )

    assert [result.feature_set for result in results] == ["baseline", "baseline+power_contact"]
    for result in results:
        assert result.training_path.exists()
        assert result.model_path.exists()
        assert result.backtest_path.exists()
        assert result.summary_path.exists()
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert summary["feature_set"] == result.feature_set
        assert summary["rows"] > 0
        assert "brier" in summary
        assert "calibration_0_5_rows" in summary


def _synthetic_statcast() -> pd.DataFrame:
    rows = []
    for i, day in enumerate(pd.date_range("2025-04-01", periods=70)):
        for batter in [1, 2, 3, 4]:
            is_power = batter in [1, 2]
            is_hr = is_power and i % (9 + batter) == 0
            rows.append(
                {
                    "game_date": day.date().isoformat(),
                    "batter": batter,
                    "player_name": f"Hitter {batter}",
                    "pitcher": 10 + (i % 3),
                    "events": "home_run" if is_hr else ("strikeout" if batter == 4 and i % 3 == 0 else "field_out"),
                    "description": "swinging_strike" if batter == 4 else "hit_into_play",
                    "bb_type": "fly_ball" if is_power else "ground_ball",
                    "launch_speed": 101 if is_power else 84,
                    "launch_angle": 27 if is_power else 4,
                    "launch_speed_angle": 6 if is_power else 2,
                    "estimated_woba_using_speedangle": 0.7 if is_power else 0.2,
                    "estimated_slg_using_speedangle": 1.2 if is_power else 0.3,
                    "woba_value": 2.0 if is_hr else 0.0,
                    "iso_value": 3.0 if is_hr else 0.0,
                    "stand": "R" if batter % 2 else "L",
                    "p_throws": "L" if i % 2 else "R",
                    "pitch_type": "FF",
                    "release_speed": 93 + i % 5,
                    "release_spin_rate": 2200 + i,
                    "release_extension": 6.2,
                    "pfx_x": 0.4,
                    "pfx_z": 1.0,
                    "inning_topbot": "Top",
                    "away_team": "NYY",
                    "home_team": "BOS",
                    "venue_name": "Fenway Park",
                    "game_pk": int(day.strftime("%Y%m%d")),
                    "at_bat_number": batter,
                    "pitch_number": 1,
                }
            )
    return pd.DataFrame(rows)
