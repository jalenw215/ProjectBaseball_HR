from pathlib import Path

import pandas as pd

from hr_predictor.config import FEATURE_COLUMNS
from hr_predictor.model import load_model, predict_probabilities, train_model


def test_train_model_round_trip(tmp_path: Path):
    rows = []
    for i in range(80):
        target = 1 if i % 9 == 0 else 0
        row = {
            "game_date": f"2025-05-{(i % 28) + 1:02d}",
            "target_hr": target,
            "player_name": f"Hitter {i % 12}",
        }
        for feature in FEATURE_COLUMNS:
            row[feature] = 0.1 + (i % 7) * 0.01
        row["batter_pa_30"] = 80 + i % 20
        row["batter_hr_rate_30"] = 0.08 if target else 0.02
        row["park_hr_factor"] = 1.05
        rows.append(row)
    training = tmp_path / "training.csv"
    model_path = tmp_path / "model.joblib"
    pd.DataFrame(rows).to_csv(training, index=False)

    train_model(training, model_path)
    model, _ = load_model(model_path)
    predictions = predict_probabilities(model, pd.read_csv(training))

    assert len(predictions) == 80
    assert predictions.min() >= 0
    assert predictions.max() <= 1

