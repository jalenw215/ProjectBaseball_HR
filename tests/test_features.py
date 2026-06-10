from pathlib import Path

import pandas as pd

from hr_predictor.features import build_training_rows, resolve_feature_columns


def test_build_training_rows(tmp_path: Path):
    rows = []
    for day in pd.date_range("2025-04-01", periods=35):
        for batter in [1, 2]:
            rows.append(
                {
                    "game_date": day.date().isoformat(),
                    "batter": batter,
                    "player_name": f"Hitter {batter}",
                    "pitcher": 10,
                    "events": "home_run" if batter == 1 and day.day % 11 == 0 else "field_out",
                    "launch_speed": 100 if batter == 1 else 86,
                    "launch_angle": 28 if batter == 1 else 5,
                    "launch_speed_angle": 6 if batter == 1 else 2,
                    "estimated_woba_using_speedangle": 0.8 if batter == 1 else 0.2,
                    "estimated_slg_using_speedangle": 1.5 if batter == 1 else 0.3,
                    "woba_value": 2.0 if batter == 1 else 0.0,
                    "iso_value": 3.0 if batter == 1 else 0.0,
                    "description": "swinging_strike" if batter == 2 else "hit_into_play",
                    "bb_type": "fly_ball" if batter == 1 else "ground_ball",
                    "stand": "R" if batter == 1 else "L",
                    "p_throws": "L",
                    "pitch_type": "FF",
                    "release_speed": 94,
                    "release_spin_rate": 2300,
                    "release_extension": 6.5,
                    "pfx_x": 0.4,
                    "pfx_z": 1.1,
                    "inning_topbot": "Top",
                    "away_team": "NYY",
                    "home_team": "BOS",
                    "venue_name": "Fenway Park",
                    "game_pk": int(day.strftime("%Y%m%d")),
                    "at_bat_number": 1,
                    "pitch_number": 1,
                }
            )
    source = tmp_path / "statcast.csv"
    output = tmp_path / "training.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    path = build_training_rows(source, output)
    training = pd.read_csv(path)

    assert not training.empty
    assert "target_hr" in training.columns
    assert "batter_hr_rate_30" in training.columns
    assert training["batter_pa_30"].max() > 0


def test_feature_group_columns_and_no_same_day_leakage(tmp_path: Path):
    rows = []
    for i, day in enumerate(pd.date_range("2025-04-01", periods=3)):
        rows.append(
            {
                "game_date": day.date().isoformat(),
                "batter": 1,
                "player_name": "Power Hitter",
                "pitcher": 10,
                "events": "home_run" if i == 2 else "field_out",
                "description": "hit_into_play",
                "bb_type": "fly_ball",
                "launch_speed": 110 if i == 2 else 90,
                "launch_angle": 28,
                "launch_speed_angle": 6 if i == 2 else 4,
                "estimated_woba_using_speedangle": 2.0 if i == 2 else 0.2,
                "estimated_slg_using_speedangle": 4.0 if i == 2 else 0.4,
                "woba_value": 2.0 if i == 2 else 0.0,
                "iso_value": 3.0 if i == 2 else 0.0,
                "stand": "R",
                "p_throws": "L",
                "pitch_type": "FF",
                "release_speed": 95 + i,
                "release_spin_rate": 2200 + i,
                "release_extension": 6.0,
                "pfx_x": 0.3,
                "pfx_z": 1.2,
                "inning_topbot": "Top",
                "away_team": "NYY",
                "home_team": "BOS",
                "venue_name": "Fenway Park",
                "game_pk": int(day.strftime("%Y%m%d")),
                "at_bat_number": 1,
                "pitch_number": 1,
            }
        )
    source = tmp_path / "statcast.csv"
    output = tmp_path / "training.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    build_training_rows(source, output, feature_set="all_free_statcast")
    training = pd.read_csv(output).sort_values("game_date")

    for col in resolve_feature_columns("all_free_statcast"):
        assert col in training.columns
    first_day = training.iloc[0]
    homer_day = training.iloc[-1]
    assert first_day["batter_hr_rate_30"] == 0
    assert homer_day["target_hr"] == 1
    assert homer_day["batter_hr_rate_30"] == 0
    assert homer_day["batter_estimated_woba_30"] < 2.0
    assert homer_day["platoon_advantage"] == 1
