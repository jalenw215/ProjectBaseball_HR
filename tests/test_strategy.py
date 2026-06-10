from __future__ import annotations

import json
from pathlib import Path

import pytest

from hr_predictor.strategy import ALL_FEATURES, BASELINE, BEST_PROBABILITY, BEST_TOP10, resolve_model_strategy


def test_best_top10_strategy_uses_hit_rate_then_hits_then_brier(tmp_path: Path):
    _experiment(tmp_path, "baseline", top10_hit_rate=0.18, top10_hits=20, brier=0.09)
    _experiment(tmp_path, "baseline+power_contact", top10_hit_rate=0.19, top10_hits=18, brier=0.1)
    _experiment(tmp_path, "all_free_statcast", top10_hit_rate=0.19, top10_hits=25, brier=0.2)

    selected = resolve_model_strategy(BEST_TOP10, experiments_dir=tmp_path)

    assert selected.feature_set == "all_free_statcast"


def test_best_probability_strategy_uses_brier_then_log_loss(tmp_path: Path):
    _experiment(tmp_path, "baseline", brier=0.08, log_loss=0.32)
    _experiment(tmp_path, "baseline+power_contact", brier=0.08, log_loss=0.3)
    _experiment(tmp_path, "all_free_statcast", brier=0.09, log_loss=0.29)

    selected = resolve_model_strategy(BEST_PROBABILITY, experiments_dir=tmp_path)

    assert selected.feature_set == "baseline+power_contact"


def test_missing_all_features_model_raises_clear_error(tmp_path: Path):
    exp_dir = tmp_path / "all_free_statcast"
    exp_dir.mkdir(parents=True)
    (exp_dir / "summary.json").write_text(json.dumps({"feature_set": "all_free_statcast"}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Run Feature Experiments first"):
        resolve_model_strategy(ALL_FEATURES, experiments_dir=tmp_path)


def test_baseline_falls_back_to_default_model(tmp_path: Path):
    default_model = tmp_path / "default.joblib"
    default_model.write_text("model", encoding="utf-8")

    selected = resolve_model_strategy(BASELINE, experiments_dir=tmp_path, default_model_path=default_model)

    assert selected.model_path == default_model
    assert selected.feature_set == "baseline"


def _experiment(
    root: Path,
    feature_set: str,
    top10_hit_rate: float = 0.0,
    top10_hits: float = 0.0,
    brier: float = 0.1,
    log_loss: float = 0.3,
) -> None:
    slug = feature_set.replace("+", "__")
    exp_dir = root / slug
    exp_dir.mkdir(parents=True)
    (exp_dir / "hr_model.joblib").write_text("model", encoding="utf-8")
    (exp_dir / "summary.json").write_text(
        json.dumps(
            {
                "feature_set": feature_set,
                "top10_hit_rate": top10_hit_rate,
                "top10_hits": top10_hits,
                "brier": brier,
                "log_loss": log_loss,
            }
        ),
        encoding="utf-8",
    )

