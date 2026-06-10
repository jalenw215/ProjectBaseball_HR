# MLB Daily Home Run Predictor

This project builds a free-data-first MLB home run prediction workflow:

- fetch historical Statcast data from Baseball Savant;
- build leakage-safe hitter-game training rows;
- train a calibrated logistic model;
- backtest daily rankings;
- generate today's HR probability rankings;
- view the results in a Streamlit dashboard;
- optionally post the daily report to Discord.

The model produces probabilities and research rankings, not guaranteed picks.

## Quick Start

Use Python 3.10 or newer. On this machine, the system `python3` is 3.9, so the verified local setup uses the bundled Python 3.12 runtime:

```bash
/Users/jalenai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,discord]"
```

## No-Terminal Workflow

The dashboard is the main interface. Once it is running, use the buttons at the top:

- `Fetch Historical Data`
- `Build Training Set`
- `Train Model`
- `Run Backtest`
- `Predict Today`
- `Full Refresh`

`Full Refresh` does the whole setup flow for the current 2-season window: fetches Statcast data from 2024 through today, builds training rows, trains the model, runs a backtest, and generates today's rankings.

The dashboard also shows:

- whether each required file exists;
- recent refresh logs from `data/logs/refresh.log`;
- prediction rankings when available;
- backtest charts when available.

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

## Codex/Automation Jobs

These scripts are designed for Codex or scheduled automations to run for you:

```bash
python scripts/morning_refresh.py
python scripts/lineup_refresh.py
```

Morning refresh updates data, training rows, model, backtest, and predictions. Lineup refresh only regenerates predictions unless the model/training files are missing.

## Optional CLI Workflow

The same workflow is available through CLI commands, but you do not need to use these manually.

Fetch a small historical sample:

```bash
hr-predictor fetch-statcast --start-date 2025-04-01 --end-date 2025-04-14
```

Build training rows:

```bash
hr-predictor build-training --statcast data/raw/statcast_2025-04-01_2025-04-14.csv
```

Train and backtest:

```bash
hr-predictor train
hr-predictor backtest
```

Generate today's predictions:

```bash
hr-predictor predict-today
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

## Optional Odds Input

Reliable player home run prop odds are usually paid or limited. For a free workflow, add odds manually to:

```text
data/raw/manual_hr_odds.csv
```

Expected columns:

```csv
date,player_name,american_odds,book
2026-06-08,Aaron Judge,+240,DraftKings
```

When odds are present, predictions include implied probability and a value flag.

## Discord

Create a `.env` file:

```text
DISCORD_BOT_TOKEN=your_token_here
DISCORD_CHANNEL_ID=123456789012345678
```

After generating predictions:

```bash
python scripts/post_discord.py --predictions data/predictions/latest_predictions.csv
```

By default, Discord posts show the top 5 picks.

## Important Notes

- Confirmed lineups are not guaranteed by free public sources early in the day. The v1 predictor uses recent team hitters as likely candidates.
- For best results, rerun predictions after lineups are posted.
- The backtest is more important than a single day's picks. Track calibration, Brier score, log loss, top-N hit rate, and optional ROI.
