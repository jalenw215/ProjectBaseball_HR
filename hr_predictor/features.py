from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_FEATURE_SET, FEATURE_COLUMNS, FEATURE_GROUPS, FEATURE_SET_ALIASES, PROCESSED_DIR
from .parks import park_info
from .utils import ensure_parent


def load_statcast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "game_date" not in df.columns:
        raise ValueError("Statcast file must contain game_date")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    events = _series(out, "events")
    descriptions = _series(out, "description")
    bb_type = _series(out, "bb_type")
    pitch_type = _series(out, "pitch_type")
    out["is_hr"] = (events == "home_run").astype(int)
    out["is_bbe"] = _series(out, "launch_speed").notna().astype(int)
    out["is_hardhit"] = (pd.to_numeric(_series(out, "launch_speed"), errors="coerce") >= 95).astype(int)
    out["is_barrel"] = (pd.to_numeric(_series(out, "launch_speed_angle"), errors="coerce") == 6).astype(int)
    out["is_sweet_spot"] = pd.to_numeric(_series(out, "launch_angle"), errors="coerce").between(8, 32).astype(int)
    out["is_strikeout"] = (events == "strikeout").astype(int)
    out["is_walk"] = events.isin(["walk", "intent_walk"]).astype(int)
    out["is_swinging_strike"] = descriptions.isin(
        ["swinging_strike", "swinging_strike_blocked", "foul_tip"]
    ).astype(int)
    out["is_contact_pitch"] = descriptions.isin(
        ["foul", "foul_bunt", "foul_tip", "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"]
    ).astype(int)
    out["is_swing"] = (out["is_swinging_strike"] | out["is_contact_pitch"]).astype(int)
    out["is_flyball"] = bb_type.isin(["fly_ball", "popup"]).astype(int)
    out["is_fastball"] = pitch_type.isin(["FF", "SI", "FC", "FA"]).astype(int)
    inning_topbot = _series(out, "inning_topbot")
    out["batter_team"] = np.where(inning_topbot == "Top", _series(out, "away_team"), _series(out, "home_team"))
    out["pitcher_team"] = np.where(inning_topbot == "Top", _series(out, "home_team"), _series(out, "away_team"))
    out["is_home"] = (out["batter_team"] == _series(out, "home_team")).astype(int)
    if "venue_name" in out.columns:
        out["park_hr_factor"] = out["venue_name"].map(lambda name: park_info(name).get("hr_factor", 1.0))
    else:
        out["park_hr_factor"] = 1.0
    out["launch_speed"] = pd.to_numeric(_series(out, "launch_speed"), errors="coerce")
    out["launch_angle"] = pd.to_numeric(_series(out, "launch_angle"), errors="coerce")
    for col in [
        "estimated_woba_using_speedangle",
        "estimated_slg_using_speedangle",
        "woba_value",
        "iso_value",
        "release_speed",
        "release_spin_rate",
        "release_extension",
        "pfx_x",
        "pfx_z",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan
    out["abs_movement"] = out[["pfx_x", "pfx_z"]].abs().sum(axis=1, min_count=1)
    return out


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.0)


def resolve_feature_columns(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    feature_set = FEATURE_SET_ALIASES.get(feature_set, feature_set)
    if feature_set == "all_free_statcast":
        groups = list(FEATURE_GROUPS)
    else:
        groups = feature_set.split("+")
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}")
    columns: list[str] = []
    for group in groups:
        for col in FEATURE_GROUPS[group]:
            if col not in columns:
                columns.append(col)
    return columns


def feature_set_slug(feature_set: str = DEFAULT_FEATURE_SET) -> str:
    return FEATURE_SET_ALIASES.get(feature_set, feature_set).replace("+", "__")


def build_training_rows(statcast_path: Path, output_path: Path | None = None, feature_set: str = DEFAULT_FEATURE_SET) -> Path:
    output_path = output_path or PROCESSED_DIR / "training_rows.csv"
    df = add_derived_columns(load_statcast(statcast_path))
    feature_columns = resolve_feature_columns(feature_set)

    batter_day = (
        df.groupby(["game_date", "batter", "player_name", "batter_team"], dropna=False)
        .agg(
            pa=("events", lambda s: s.notna().sum()),
            hr=("is_hr", "max"),
            hr_count=("is_hr", "sum"),
            bbe=("is_bbe", "sum"),
            barrels=("is_barrel", "sum"),
            hardhits=("is_hardhit", "sum"),
            sweet_spots=("is_sweet_spot", "sum"),
            strikeouts=("is_strikeout", "sum"),
            walks=("is_walk", "sum"),
            swings=("is_swing", "sum"),
            swinging_strikes=("is_swinging_strike", "sum"),
            contact_pitches=("is_contact_pitch", "sum"),
            avg_ev=("launch_speed", "mean"),
            avg_la=("launch_angle", "mean"),
            max_ev=("launch_speed", "max"),
            estimated_woba=("estimated_woba_using_speedangle", "mean"),
            estimated_slg=("estimated_slg_using_speedangle", "mean"),
            woba=("woba_value", "mean"),
            iso=("iso_value", "mean"),
            stand=("stand", _mode_or_unknown),
            is_home=("is_home", "max"),
            park_hr_factor=("park_hr_factor", "mean"),
        )
        .reset_index()
        .sort_values(["batter", "game_date"])
    )

    pitcher_day = (
        df.groupby(["game_date", "pitcher"], dropna=False)
        .agg(
            batters_faced=("events", lambda s: s.notna().sum()),
            hr_allowed=("is_hr", "sum"),
            bbe_allowed=("is_bbe", "sum"),
            barrels_allowed=("is_barrel", "sum"),
            hardhits_allowed=("is_hardhit", "sum"),
            strikeouts=("is_strikeout", "sum"),
            walks=("is_walk", "sum"),
            flyballs_allowed=("is_flyball", "sum"),
            avg_release_speed=("release_speed", "mean"),
            max_release_speed=("release_speed", "max"),
            avg_spin_rate=("release_spin_rate", "mean"),
            avg_extension=("release_extension", "mean"),
            avg_abs_movement=("abs_movement", "mean"),
            fastballs=("is_fastball", "sum"),
            pitches=("pitch_type", lambda s: s.notna().sum()),
            p_throws=("p_throws", _mode_or_unknown),
        )
        .reset_index()
        .sort_values(["pitcher", "game_date"])
    )

    batter_roll = _rolling_batter_features(batter_day)
    pitcher_roll = _rolling_pitcher_features(pitcher_day)

    first_pitcher = (
        df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
        .groupby(["game_date", "batter"], dropna=False)
        .agg(pitcher=("pitcher", "first"))
        .reset_index()
    )

    rows = batter_day.merge(first_pitcher, on=["game_date", "batter"], how="left")
    rows = rows.merge(batter_roll, on=["game_date", "batter"], how="left")
    rows = rows.merge(pitcher_roll, on=["game_date", "pitcher"], how="left")
    rows = _normalize_hand_columns(rows)
    rows = _add_matchup_features(rows)
    rows["temperature_2m"] = 70.0
    rows["wind_speed_10m"] = 5.0
    rows = rows.rename(columns={"hr": "target_hr"})
    rows = rows.dropna(subset=["batter_pa_30"])
    for col in feature_columns:
        if col not in rows.columns:
            rows[col] = 0.0
    rows[feature_columns] = rows[feature_columns].fillna(0.0)
    rows["park_hr_factor"] = rows["park_hr_factor"].replace(0, 1.0).fillna(1.0)
    rows["player_name"] = rows["player_name"].fillna("Unknown")
    rows["feature_set"] = feature_set
    rows.to_csv(ensure_parent(output_path), index=False)
    return output_path


def _rolling_batter_features(batter_day: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in batter_day.groupby("batter", dropna=False):
        group = group.sort_values("game_date").copy()
        shifted = group[
            [
                "pa",
                "hr_count",
                "bbe",
                "barrels",
                "hardhits",
                "sweet_spots",
                "strikeouts",
                "walks",
                "swings",
                "swinging_strikes",
                "contact_pitches",
                "avg_ev",
                "avg_la",
                "max_ev",
                "estimated_woba",
                "estimated_slg",
                "woba",
                "iso",
            ]
        ].shift(1)
        roll = shifted.rolling(30, min_periods=1).sum()
        avg_roll = group[
            ["avg_ev", "avg_la", "max_ev", "estimated_woba", "estimated_slg", "woba", "iso"]
        ].shift(1).rolling(30, min_periods=1).mean()
        stand = group["stand"].shift(1).ffill().fillna("U")
        features = pd.DataFrame(
            {
                "game_date": group["game_date"].values,
                "batter": group["batter"].values,
                "batter_pa_30": roll["pa"].values,
                "batter_hr_rate_30": _safe_rate(roll["hr_count"], roll["pa"]).values,
                "batter_barrel_rate_30": _safe_rate(roll["barrels"], roll["bbe"]).values,
                "batter_hardhit_rate_30": _safe_rate(roll["hardhits"], roll["bbe"]).values,
                "batter_avg_ev_30": avg_roll["avg_ev"].fillna(88.0).values,
                "batter_avg_la_30": avg_roll["avg_la"].fillna(12.0).values,
                "batter_estimated_woba_30": avg_roll["estimated_woba"].fillna(0.0).values,
                "batter_estimated_slg_30": avg_roll["estimated_slg"].fillna(0.0).values,
                "batter_woba_30": avg_roll["woba"].fillna(0.0).values,
                "batter_iso_30": avg_roll["iso"].fillna(0.0).values,
                "batter_max_ev_30": avg_roll["max_ev"].fillna(88.0).values,
                "batter_sweet_spot_rate_30": _safe_rate(roll["sweet_spots"], roll["bbe"]).values,
                "batter_k_rate_30": _safe_rate(roll["strikeouts"], roll["pa"]).values,
                "batter_bb_rate_30": _safe_rate(roll["walks"], roll["pa"]).values,
                "batter_swinging_strike_rate_30": _safe_rate(roll["swinging_strikes"], roll["swings"]).values,
                "batter_contact_rate_30": _safe_rate(roll["contact_pitches"], roll["swings"]).values,
                "stand": stand.values,
            }
        )
        parts.append(features)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _rolling_pitcher_features(pitcher_day: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in pitcher_day.groupby("pitcher", dropna=False):
        group = group.sort_values("game_date").copy()
        shifted = group[
            [
                "batters_faced",
                "hr_allowed",
                "bbe_allowed",
                "barrels_allowed",
                "hardhits_allowed",
                "strikeouts",
                "walks",
                "flyballs_allowed",
                "avg_release_speed",
                "max_release_speed",
                "avg_spin_rate",
                "avg_extension",
                "avg_abs_movement",
                "fastballs",
                "pitches",
            ]
        ].shift(1)
        roll = shifted.rolling(30, min_periods=1).sum()
        avg_roll = group[
            ["avg_release_speed", "max_release_speed", "avg_spin_rate", "avg_extension", "avg_abs_movement"]
        ].shift(1).rolling(30, min_periods=1).mean()
        p_throws = group["p_throws"].shift(1).ffill().fillna("U")
        fip_proxy = ((13 * roll["hr_allowed"]) + (3 * roll["walks"]) - (2 * roll["strikeouts"])).div(
            roll["batters_faced"].replace(0, np.nan)
        ).fillna(0.0)
        features = pd.DataFrame(
            {
                "game_date": group["game_date"].values,
                "pitcher": group["pitcher"].values,
                "pitcher_bf_30": roll["batters_faced"].fillna(0.0).values,
                "pitcher_hr_rate_allowed_30": _safe_rate(roll["hr_allowed"], roll["batters_faced"]).values,
                "pitcher_barrel_rate_allowed_30": _safe_rate(roll["barrels_allowed"], roll["bbe_allowed"]).values,
                "pitcher_hardhit_rate_allowed_30": _safe_rate(roll["hardhits_allowed"], roll["bbe_allowed"]).values,
                "pitcher_k_rate_30": _safe_rate(roll["strikeouts"], roll["batters_faced"]).values,
                "pitcher_bb_rate_30": _safe_rate(roll["walks"], roll["batters_faced"]).values,
                "pitcher_fip_proxy_30": fip_proxy.values,
                "pitcher_flyball_rate_allowed_30": _safe_rate(roll["flyballs_allowed"], roll["bbe_allowed"]).values,
                "pitcher_hr_per_flyball_allowed_30": _safe_rate(roll["hr_allowed"], roll["flyballs_allowed"]).values,
                "pitcher_avg_release_speed_30": avg_roll["avg_release_speed"].fillna(0.0).values,
                "pitcher_max_release_speed_30": avg_roll["max_release_speed"].fillna(0.0).values,
                "pitcher_avg_spin_rate_30": avg_roll["avg_spin_rate"].fillna(0.0).values,
                "pitcher_avg_extension_30": avg_roll["avg_extension"].fillna(0.0).values,
                "pitcher_avg_abs_movement_30": avg_roll["avg_abs_movement"].fillna(0.0).values,
                "pitcher_fastball_rate_30": _safe_rate(roll["fastballs"], roll["pitches"]).values,
                "p_throws": p_throws.values,
            }
        )
        parts.append(features)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def latest_batter_profiles(training_rows: pd.DataFrame) -> pd.DataFrame:
    rows = training_rows.sort_values("game_date").copy()
    idx = rows.groupby("batter")["game_date"].idxmax()
    excluded = {"target_hr", "hr_count", "pa", "bbe", "barrels", "hardhits", "avg_ev", "avg_la", "pitcher"}
    profile_cols = [
        col
        for col in rows.columns
        if col not in excluded and (col.startswith("batter_") or col in {"batter", "player_name", "batter_team", "stand"})
    ]
    profiles = rows.loc[idx, profile_cols].copy()
    return profiles.rename(columns={"batter_team": "team"})


def latest_pitcher_profiles(training_rows: pd.DataFrame) -> pd.DataFrame:
    rows = training_rows.sort_values("game_date").copy()
    pitcher_cols = [
        "pitcher",
        "pitcher_bf_30",
        "pitcher_hr_rate_allowed_30",
        "pitcher_barrel_rate_allowed_30",
        "pitcher_hardhit_rate_allowed_30",
        "pitcher_k_rate_30",
        "pitcher_bb_rate_30",
        "pitcher_fip_proxy_30",
        "pitcher_flyball_rate_allowed_30",
        "pitcher_hr_per_flyball_allowed_30",
        "pitcher_avg_release_speed_30",
        "pitcher_max_release_speed_30",
        "pitcher_avg_spin_rate_30",
        "pitcher_avg_extension_30",
        "pitcher_avg_abs_movement_30",
        "pitcher_fastball_rate_30",
        "p_throws",
    ]
    rows = rows.dropna(subset=["pitcher"])
    idx = rows.groupby("pitcher")["game_date"].idxmax()
    return rows.loc[idx, [col for col in pitcher_cols if col in rows.columns]].copy()


def _mode_or_unknown(series: pd.Series) -> str:
    modes = series.dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else "U"


def _series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _add_matchup_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    stand = out.get("stand", pd.Series("U", index=out.index)).fillna("U").astype(str)
    throws = out.get("p_throws", pd.Series("U", index=out.index)).fillna("U").astype(str)
    out["batter_bats_right"] = (stand == "R").astype(int)
    out["batter_bats_left"] = (stand == "L").astype(int)
    out["pitcher_throws_right"] = (throws == "R").astype(int)
    out["pitcher_throws_left"] = (throws == "L").astype(int)
    out["platoon_advantage"] = (((stand == "R") & (throws == "L")) | ((stand == "L") & (throws == "R"))).astype(int)
    return out


def _normalize_hand_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "stand" not in out.columns:
        stand_cols = [col for col in ["stand_y", "stand_x"] if col in out.columns]
        out["stand"] = out[stand_cols[0]] if stand_cols else "U"
    if "p_throws" not in out.columns:
        throw_cols = [col for col in ["p_throws_y", "p_throws_x"] if col in out.columns]
        out["p_throws"] = out[throw_cols[0]] if throw_cols else "U"
    return out
