from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hr_predictor.config import (
    DEFAULT_BACKTEST_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_PREDICTIONS_FILE,
    DEFAULT_REFRESH_LOG,
    DEFAULT_STATCAST_FILE,
    DEFAULT_TRAINING_FILE,
    EXPERIMENTS_DIR,
)
from hr_predictor.runner import PipelineRunner, read_recent_log
from hr_predictor.strategy import BEST_TOP10, MODEL_STRATEGIES, resolve_model_strategy
from hr_predictor.favorites import favorite_three_from_predictions

st.set_page_config(page_title="MLB HR Predictor", layout="wide")
st.title("MLB Daily Home Run Predictor")
st.caption("Research probabilities only. Use backtests and calibration before treating anything as betting signal.")


def run_job(label: str, action):
    status = st.status(label, expanded=True)

    def progress(message: str) -> None:
        status.write(message)

    runner = PipelineRunner(progress=progress)
    results = action(runner)
    errors = [result for result in results if result.status != "ok"]
    if errors:
        status.update(label=f"{label} finished with errors", state="error")
        for result in errors:
            st.error(f"{result.name}: {result.detail}")
    else:
        status.update(label=f"{label} complete", state="complete")
        st.success("Done. Refreshing dashboard data.")
        st.rerun()


st.subheader("No-Terminal Controls")
st.write("Use these buttons instead of Terminal. Long data pulls can take a while, especially the first historical fetch.")

strategy = st.selectbox("Model Strategy", MODEL_STRATEGIES, index=MODEL_STRATEGIES.index(BEST_TOP10))
strategy_selection = None
try:
    strategy_selection = resolve_model_strategy(strategy)
    st.caption(
        f"Using `{strategy_selection.feature_set}` from `{strategy_selection.model_path}`. "
        f"{strategy_selection.explanation}"
    )
except Exception as exc:
    st.warning(f"{strategy}: {exc}")

control_cols = st.columns(3)
with control_cols[0]:
    if st.button("Fetch Historical Data", use_container_width=True):
        run_job("Fetching two seasons of Statcast data", lambda r: [r.fetch_historical_data()])
    if st.button("Build Training Set", use_container_width=True):
        run_job("Building training rows", lambda r: [r.build_training_set()])
with control_cols[1]:
    if st.button("Train Model", use_container_width=True):
        run_job("Training HR model", lambda r: [r.train_model()])
    if st.button("Run Backtest", use_container_width=True):
        run_job("Running walk-forward backtest", lambda r: [r.run_backtest()])
with control_cols[2]:
    if st.button("Predict Today", use_container_width=True):
        if strategy_selection is None:
            st.error("No model is available for the selected strategy. Run Feature Experiments first or choose Baseline.")
        else:
            run_job("Generating today's predictions", lambda r: [r.predict_today(model_path=strategy_selection.model_path)])
    if st.button("Full Refresh", type="primary", use_container_width=True):
        if strategy_selection is None:
            st.error("No model is available for the selected strategy. Run Feature Experiments first or choose Baseline.")
        else:
            run_job("Running full refresh", lambda r: r.full_refresh(model_path=strategy_selection.model_path))

if st.button("Run Feature Experiments", use_container_width=True):
    run_job("Running feature group experiments", lambda r: [r.run_experiments()])

status_cols = st.columns(4)
status_cols[0].metric("Historical Data", "Ready" if DEFAULT_STATCAST_FILE.exists() else "Missing")
status_cols[1].metric("Training Set", "Ready" if DEFAULT_TRAINING_FILE.exists() else "Missing")
status_cols[2].metric("Model", "Ready" if DEFAULT_MODEL_FILE.exists() else "Missing")
status_cols[3].metric("Predictions", "Ready" if DEFAULT_PREDICTIONS_FILE.exists() else "Missing")

with st.expander("Refresh Log", expanded=not DEFAULT_PREDICTIONS_FILE.exists()):
    st.code(read_recent_log(DEFAULT_REFRESH_LOG), language="text")

if not DEFAULT_PREDICTIONS_FILE.exists():
    st.info("No predictions yet. Click Full Refresh to fetch data, train the model, backtest, and generate today's rankings.")
    st.stop()

df = pd.read_csv(DEFAULT_PREDICTIONS_FILE)
if df.empty:
    st.warning("The prediction file exists but is empty. Run Predict Today or Full Refresh.")
    st.stop()

df = df.sort_values("rank")

st.subheader("Favorite 3 HR Picks")
st.caption("A data-driven analyst card: model probability plus power form, matchup, pitcher vulnerability, and game context.")
if st.button("Give Me Favorite 3 Picks", type="primary", use_container_width=True):
    try:
        st.session_state["favorite_picks"] = favorite_three_from_predictions(df)
    except Exception as exc:
        st.error(f"Could not generate favorite picks: {exc}")

favorite_picks = st.session_state.get("favorite_picks", [])
if favorite_picks:
    pick_cols = st.columns(3)
    for col, pick in zip(pick_cols, favorite_picks):
        with col:
            st.markdown(f"### {pick.rank}. {pick.player_name}")
            st.metric("HR Probability", f"{pick.hr_probability:.1%}", help=f"Confidence tier: {pick.confidence_tier}")
            st.metric("Pick Score", f"{pick.pick_score:.1f}")
            st.write(f"{pick.team} vs {pick.opponent} | SP: {pick.opposing_pitcher}")
            st.markdown(f"**Analyst Case:** {pick.analyst_case}")
            st.markdown(f"**Bettor Lens:** {pick.bettor_lens}")
            st.markdown(f"**Fan Read:** {pick.fan_read}")

st.subheader("Today's HR Rankings")
top_n = st.slider("Show top hitters", 5, 50, 20)
tiers = sorted(df["confidence_tier"].dropna().unique())
selected_tiers = st.multiselect("Confidence tiers", tiers, default=tiers)
filtered = df[df["confidence_tier"].isin(selected_tiers)].head(top_n)

metric_cols = st.columns(4)
metric_cols[0].metric("Candidates", f"{len(df):,}")
metric_cols[1].metric("Top Probability", f"{df['hr_probability'].max():.1%}")
metric_cols[2].metric("Value Flags", f"{int(df.get('value_flag', pd.Series(dtype=bool)).fillna(False).sum())}")
metric_cols[3].metric("Games", f"{df['game_pk'].nunique():,}")

feature_set = df.get("feature_set", pd.Series(["baseline"])).dropna()
st.caption(f"Active feature set: {feature_set.iloc[0] if not feature_set.empty else 'baseline'}")

display_cols = [
    "rank",
    "player_name",
    "team",
    "opponent",
    "hr_probability",
    "confidence_tier",
    "feature_set",
    "opposing_pitcher",
    "venue_name",
    "matchup_note",
]
optional_cols = ["american_odds", "implied_probability", "book", "value_flag"]
display_cols.extend([c for c in optional_cols if c in df.columns])

st.dataframe(
    filtered[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "hr_probability": st.column_config.ProgressColumn("HR probability", format="%.1f%%", min_value=0, max_value=1),
        "implied_probability": st.column_config.ProgressColumn("Implied probability", format="%.1f%%", min_value=0, max_value=1),
    },
)

st.subheader("Model Signals")
signal_cols = [
    "player_name",
    "batter_pa_30",
    "batter_hr_rate_30",
    "batter_barrel_rate_30",
    "batter_hardhit_rate_30",
    "batter_avg_ev_30",
    "park_hr_factor",
    "temperature_2m",
    "wind_speed_10m",
]
st.dataframe(filtered[[c for c in signal_cols if c in filtered.columns]], use_container_width=True, hide_index=True)

if DEFAULT_BACKTEST_FILE.exists():
    st.subheader("Backtest Snapshot")
    bt = pd.read_csv(DEFAULT_BACKTEST_FILE)
    if not bt.empty:
        bt["game_date"] = pd.to_datetime(bt["game_date"])
        daily = bt.groupby(bt["game_date"].dt.date).agg(
            candidates=("player_name", "count"),
            actual_hr=("target_hr", "sum"),
            top10_hr=("target_hr", lambda s: s[bt.loc[s.index, "rank"] <= 10].sum()),
        )
        st.line_chart(daily[["actual_hr", "top10_hr"]])

experiment_rows = []
if EXPERIMENTS_DIR.exists():
    for summary_path in sorted(EXPERIMENTS_DIR.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        experiment_rows.append(
            {
                "feature_set": summary.get("feature_set", summary_path.parent.name),
                "rows": summary.get("rows", 0),
                "hr_events": summary.get("hr_events", 0),
                "brier": summary.get("brier"),
                "log_loss": summary.get("log_loss"),
                "top10_hit_rate": summary.get("top10_hit_rate"),
                "top10_hits": summary.get("top10_hits"),
            }
        )
if experiment_rows:
    st.subheader("Feature Experiment Results")
    experiments = pd.DataFrame(experiment_rows).sort_values(["brier", "log_loss"], na_position="last")
    st.dataframe(experiments, use_container_width=True, hide_index=True)
