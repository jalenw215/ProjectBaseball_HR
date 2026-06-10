from __future__ import annotations

from datetime import date
from pathlib import Path

from hr_predictor.runner import PipelineRunner, RunnerConfig, month_chunks, read_recent_log


def test_month_chunks():
    chunks = list(month_chunks(date(2024, 1, 15), date(2024, 3, 2)))

    assert chunks == [
        (date(2024, 1, 15), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 2)),
    ]


def test_morning_refresh_step_order(monkeypatch, tmp_path: Path):
    runner = PipelineRunner(config=_config(tmp_path))
    calls = []

    monkeypatch.setattr(runner, "fetch_historical_data", lambda: calls.append("fetch") or _ok("fetch"))
    monkeypatch.setattr(runner, "build_training_set", lambda: calls.append("build") or _ok("build"))
    monkeypatch.setattr(runner, "train_model", lambda: calls.append("train") or _ok("train"))
    monkeypatch.setattr(runner, "run_backtest", lambda: calls.append("backtest") or _ok("backtest"))
    monkeypatch.setattr(
        runner,
        "predict_today",
        lambda prediction_date=None, model_path=None: calls.append("predict") or _ok("predict"),
    )

    results = runner.morning_refresh("2026-06-08")

    assert calls == ["fetch", "build", "train", "backtest", "predict"]
    assert all(result.status == "ok" for result in results)


def test_lineup_refresh_skips_training_when_files_exist(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    config.training_file.write_text("training", encoding="utf-8")
    config.model_file.write_text("model", encoding="utf-8")
    runner = PipelineRunner(config=config)
    calls = []

    monkeypatch.setattr(runner, "build_training_set", lambda: calls.append("build") or _ok("build"))
    monkeypatch.setattr(runner, "train_model", lambda: calls.append("train") or _ok("train"))
    monkeypatch.setattr(runner, "predict_today", lambda prediction_date=None: calls.append("predict") or _ok("predict"))

    runner.lineup_refresh("2026-06-08")

    assert calls == ["predict"]


def test_morning_refresh_stops_after_failure(monkeypatch, tmp_path: Path):
    runner = PipelineRunner(config=_config(tmp_path))
    calls = []

    monkeypatch.setattr(runner, "fetch_historical_data", lambda: calls.append("fetch") or _ok("fetch"))
    monkeypatch.setattr(runner, "build_training_set", lambda: calls.append("build") or _error("build"))
    monkeypatch.setattr(runner, "train_model", lambda: calls.append("train") or _ok("train"))

    results = runner.morning_refresh("2026-06-08")

    assert calls == ["fetch", "build"]
    assert results[-1].status == "error"


def test_morning_refresh_uses_cached_statcast_when_fetch_fails(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    config.statcast_file.write_text("cached statcast", encoding="utf-8")
    runner = PipelineRunner(config=config)
    calls = []

    def fail_fetch():
        calls.append("fetch")
        raise RuntimeError("dns unavailable")

    monkeypatch.setattr(runner, "fetch_historical_data", fail_fetch)
    monkeypatch.setattr(runner, "build_training_set", lambda: calls.append("build") or _ok("build"))
    monkeypatch.setattr(runner, "train_model", lambda: calls.append("train") or _ok("train"))
    monkeypatch.setattr(runner, "run_backtest", lambda: calls.append("backtest") or _ok("backtest"))
    monkeypatch.setattr(
        runner,
        "predict_today",
        lambda prediction_date=None, model_path=None: calls.append("predict") or _ok("predict"),
    )

    results = runner.morning_refresh("2026-06-08")

    assert calls == ["fetch", "build", "train", "backtest", "predict"]
    assert all(result.status == "ok" for result in results)


def test_failed_step_logs_without_deleting_previous_predictions(tmp_path: Path):
    config = _config(tmp_path)
    config.predictions_file.write_text("previous predictions", encoding="utf-8")
    runner = PipelineRunner(config=config)

    result = runner._run_step("Explode", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert result.status == "error"
    assert config.predictions_file.read_text(encoding="utf-8") == "previous predictions"
    assert "boom" in read_recent_log(config.log_file)


def test_predict_today_uses_selected_model_path(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    config.training_file.write_text("training", encoding="utf-8")
    config.model_file.write_text("default", encoding="utf-8")
    selected_model = tmp_path / "selected.joblib"
    selected_model.write_text("selected", encoding="utf-8")
    calls = {}

    def fake_predict_for_date(prediction_date, training_path, model_path, output_path):
        calls["model_path"] = model_path
        output_path.write_text("predictions", encoding="utf-8")
        return output_path

    monkeypatch.setattr("hr_predictor.runner.predict_for_date", fake_predict_for_date)
    monkeypatch.setattr("hr_predictor.runner.fetch_schedule", lambda prediction_date: [])

    result = PipelineRunner(config=config).predict_today("2026-06-08", model_path=selected_model)

    assert result.status == "ok"
    assert calls["model_path"] == selected_model


def test_predict_today_ignores_schedule_fetch_failures(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    config.training_file.write_text("training", encoding="utf-8")
    config.model_file.write_text("default", encoding="utf-8")
    monkeypatch.setattr("hr_predictor.runner.fetch_schedule", lambda prediction_date: (_ for _ in ()).throw(RuntimeError("dns")))
    monkeypatch.setattr("hr_predictor.runner.predict_for_date", lambda *args, **kwargs: config.predictions_file.write_text("predictions", encoding="utf-8") or config.predictions_file)

    result = PipelineRunner(config=config).predict_today("2026-06-08")

    assert result.status == "ok"
    assert config.predictions_file.read_text(encoding="utf-8") == "predictions"


def _config(tmp_path: Path) -> RunnerConfig:
    return RunnerConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        statcast_file=tmp_path / "statcast.csv",
        training_file=tmp_path / "training.csv",
        model_file=tmp_path / "model.joblib",
        predictions_file=tmp_path / "predictions.csv",
        backtest_file=tmp_path / "backtest.csv",
        log_file=tmp_path / "refresh.log",
    )


def _ok(name):
    from hr_predictor.runner import StepResult

    return StepResult(name=name, status="ok", detail="ok")


def _error(name):
    from hr_predictor.runner import StepResult

    return StepResult(name=name, status="error", detail="bad")
