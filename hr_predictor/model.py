from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_FEATURE_SET, DEFAULT_MODEL_FILE
from .features import resolve_feature_columns
from .utils import ensure_parent


def train_model(training_path: Path, model_path: Path | None = None, feature_set: str = DEFAULT_FEATURE_SET) -> Path:
    model_path = model_path or DEFAULT_MODEL_FILE
    feature_columns = resolve_feature_columns(feature_set)
    rows = pd.read_csv(training_path, parse_dates=["game_date"])
    rows = rows.dropna(subset=["target_hr"])
    _ensure_feature_columns(rows, feature_columns)
    X = rows[feature_columns]
    y = rows["target_hr"].astype(int)
    base = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    cv = min(5, max(2, int(y.sum())))
    model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    model.fit(X, y)
    joblib.dump({"model": model, "features": feature_columns, "feature_set": feature_set}, ensure_parent(model_path))
    return model_path


def load_model(model_path: Path = DEFAULT_MODEL_FILE):
    payload = joblib.load(model_path)
    return payload["model"], payload["features"]


def predict_probabilities(model, rows: pd.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
    feature_columns = feature_columns or resolve_feature_columns(DEFAULT_FEATURE_SET)
    _ensure_feature_columns(rows, feature_columns)
    return model.predict_proba(rows[feature_columns])[:, 1]


def walk_forward_backtest(
    training_path: Path, min_train_days: int = 45, feature_set: str = DEFAULT_FEATURE_SET
) -> pd.DataFrame:
    feature_columns = resolve_feature_columns(feature_set)
    rows = pd.read_csv(training_path, parse_dates=["game_date"])
    rows = rows.sort_values("game_date").dropna(subset=["target_hr"])
    _ensure_feature_columns(rows, feature_columns)
    dates = sorted(rows["game_date"].dt.date.unique())
    outputs = []
    for current_date in dates:
        train = rows[rows["game_date"].dt.date < current_date]
        test = rows[rows["game_date"].dt.date == current_date]
        if train["game_date"].dt.date.nunique() < min_train_days or train["target_hr"].sum() < 5 or test.empty:
            continue
        base = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(train[feature_columns], train["target_hr"].astype(int))
        day = test.copy()
        day["hr_probability"] = model.predict_proba(day[feature_columns])[:, 1]
        day["rank"] = day["hr_probability"].rank(ascending=False, method="first").astype(int)
        day["feature_set"] = feature_set
        outputs.append(day)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def summarize_backtest(predictions: pd.DataFrame) -> dict[str, float]:
    if predictions.empty:
        return {}
    y = predictions["target_hr"].astype(int)
    p = predictions["hr_probability"].clip(0.001, 0.999)
    top10 = predictions[predictions["rank"] <= 10]
    summary = {
        "rows": float(len(predictions)),
        "hr_events": float(y.sum()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "top10_hit_rate": float(top10["target_hr"].mean()) if not top10.empty else 0.0,
        "top10_hits": float(top10["target_hr"].sum()) if not top10.empty else 0.0,
    }
    for label, low, high in [
        ("calibration_0_5", 0.0, 0.05),
        ("calibration_5_10", 0.05, 0.10),
        ("calibration_10_15", 0.10, 0.15),
        ("calibration_15_plus", 0.15, 1.01),
    ]:
        bucket = predictions[(p >= low) & (p < high)]
        summary[f"{label}_rows"] = float(len(bucket))
        summary[f"{label}_actual_rate"] = float(bucket["target_hr"].mean()) if not bucket.empty else 0.0
        summary[f"{label}_avg_probability"] = float(bucket["hr_probability"].mean()) if not bucket.empty else 0.0
    return summary


def _ensure_feature_columns(rows: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [col for col in feature_columns if col not in rows.columns]
    if missing:
        raise ValueError(f"Training rows are missing feature columns: {missing}")
