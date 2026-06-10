from __future__ import annotations

from pathlib import Path

import pandas as pd


def format_daily_report(predictions_path: Path, limit: int = 15) -> str:
    df = pd.read_csv(predictions_path)
    if df.empty:
        return "No HR predictions available."
    top = df.sort_values("rank").head(limit)
    lines = ["# MLB HR Predictor", "", "Daily probabilities are research estimates, not guarantees.", ""]
    for _, row in top.iterrows():
        odds = ""
        if "american_odds" in row and pd.notna(row["american_odds"]):
            odds = f" | odds {row['american_odds']}"
        value = " | VALUE" if bool(row.get("value_flag", False)) else ""
        lines.append(
            f"{int(row['rank'])}. {row['player_name']} ({row['team']}) "
            f"{row['hr_probability']:.1%} [{row['confidence_tier']}]{odds}{value}"
        )
        lines.append(f"   {row['matchup_note']}")
    return "\n".join(lines)

