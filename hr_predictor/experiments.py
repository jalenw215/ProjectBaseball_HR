from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_STATCAST_FILE, EXPERIMENTS_DIR
from .features import build_training_rows, feature_set_slug, resolve_feature_columns
from .model import summarize_backtest, train_model, walk_forward_backtest
from .utils import ensure_parent

DEFAULT_EXPERIMENT_FEATURE_SETS = [
    "baseline",
    "baseline+power_contact",
    "baseline+pitcher_vulnerability",
    "all_free_statcast",
]


@dataclass(frozen=True)
class ExperimentResult:
    feature_set: str
    training_path: Path
    model_path: Path
    backtest_path: Path
    summary_path: Path
    summary: dict[str, float | str | list[str]]


def experiment_dir(feature_set: str, root: Path = EXPERIMENTS_DIR) -> Path:
    return root / feature_set_slug(feature_set)


def run_feature_set_experiment(
    feature_set: str,
    statcast_path: Path = DEFAULT_STATCAST_FILE,
    root: Path = EXPERIMENTS_DIR,
    min_train_days: int = 45,
) -> ExperimentResult:
    resolve_feature_columns(feature_set)
    out_dir = experiment_dir(feature_set, root)
    training_path = out_dir / "training_rows.csv"
    model_path = out_dir / "hr_model.joblib"
    backtest_path = out_dir / "backtest_predictions.csv"
    summary_path = out_dir / "summary.json"

    build_training_rows(statcast_path, training_path, feature_set=feature_set)
    train_model(training_path, model_path, feature_set=feature_set)
    predictions = walk_forward_backtest(training_path, min_train_days=min_train_days, feature_set=feature_set)
    predictions.to_csv(ensure_parent(backtest_path), index=False)
    summary = summarize_backtest(predictions)
    summary.update({"feature_set": feature_set, "features": resolve_feature_columns(feature_set)})
    ensure_parent(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ExperimentResult(feature_set, training_path, model_path, backtest_path, summary_path, summary)


def run_feature_set_experiments(
    feature_sets: list[str] | None = None,
    statcast_path: Path = DEFAULT_STATCAST_FILE,
    root: Path = EXPERIMENTS_DIR,
    min_train_days: int = 45,
) -> list[ExperimentResult]:
    feature_sets = feature_sets or DEFAULT_EXPERIMENT_FEATURE_SETS
    return [
        run_feature_set_experiment(feature_set, statcast_path=statcast_path, root=root, min_train_days=min_train_days)
        for feature_set in feature_sets
    ]
