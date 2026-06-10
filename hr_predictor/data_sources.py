from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import MLB_SCHEDULE_URL, OPEN_METEO_URL, RAW_DIR, STATCAST_URL
from .parks import park_info
from .utils import ensure_parent

TEAM_ID_TO_ABBREVIATION = {
    108: "LAA",
    109: "AZ",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KC",
    119: "LAD",
    120: "WSH",
    121: "NYM",
    133: "ATH",
    134: "PIT",
    135: "SD",
    136: "SEA",
    137: "SF",
    138: "STL",
    139: "TB",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CWS",
    146: "MIA",
    147: "NYY",
    158: "MIL",
}

MLB_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

_MLB_SESSION: requests.Session | None = None


def _mlb_session() -> requests.Session:
    global _MLB_SESSION
    if _MLB_SESSION is None:
        session = requests.Session()
        session.headers.update(MLB_BROWSER_HEADERS)
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _MLB_SESSION = session
    return _MLB_SESSION


def _request_text(url: str, *, params: dict[str, Any], timeout: int) -> str:
    response = _mlb_session().get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.text


def _request_json(url: str, *, params: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = _mlb_session().get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_statcast_csv(start_date: str, end_date: str, output_path: Path | None = None) -> Path:
    output_path = output_path or RAW_DIR / f"statcast_{start_date}_{end_date}.csv"
    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|PO|S|",
        "hfPR": "",
        "hfZ": "",
        "stadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": "",
        "hfSit": "",
        "player_type": "batter",
        "hfOuts": "",
        "opponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "hfInfield": "",
        "team": "",
        "position": "",
        "hfOutfield": "",
        "hfRO": "",
        "home_road": "",
        "hfFlag": "",
        "metric_1": "",
        "hfInn": "",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc",
        "min_abs": "0",
        "type": "details",
    }
    ensure_parent(output_path).write_text(_request_text(STATCAST_URL, params=params, timeout=90), encoding="utf-8")
    return output_path


def fetch_schedule(game_date: str) -> list[dict[str, Any]]:
    payload = _request_json(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher,venue",
        },
        timeout=30,
    )
    games: list[dict[str, Any]] = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            teams = game.get("teams", {})
            venue = game.get("venue", {})
            away_team = teams.get("away", {}).get("team", {})
            home_team = teams.get("home", {}).get("team", {})
            games.append(
                {
                    "game_pk": game.get("gamePk"),
                    "game_date": game_date,
                    "game_time_utc": game.get("gameDate"),
                    "venue_name": venue.get("name"),
                    "away_team": team_abbreviation(away_team),
                    "home_team": team_abbreviation(home_team),
                    "away_probable_pitcher_id": teams.get("away", {}).get("probablePitcher", {}).get("id"),
                    "away_probable_pitcher": teams.get("away", {}).get("probablePitcher", {}).get("fullName"),
                    "home_probable_pitcher_id": teams.get("home", {}).get("probablePitcher", {}).get("id"),
                    "home_probable_pitcher": teams.get("home", {}).get("probablePitcher", {}).get("fullName"),
                    "status": game.get("status", {}).get("detailedState"),
                }
            )
    return games


def team_abbreviation(team: dict[str, Any]) -> str | None:
    if team.get("abbreviation"):
        return team["abbreviation"]
    team_id = team.get("id")
    if team_id in TEAM_ID_TO_ABBREVIATION:
        return TEAM_ID_TO_ABBREVIATION[team_id]
    return team.get("name")


def fetch_weather_for_game(game: dict[str, Any]) -> dict[str, float]:
    info = park_info(game.get("venue_name"))
    game_time = game.get("game_time_utc")
    params = {
        "latitude": info["lat"],
        "longitude": info["lon"],
        "hourly": "temperature_2m,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
        "forecast_days": 2,
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=20)
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
        if not game_time or not hourly.get("time"):
            raise ValueError("missing game time or hourly weather")
        target = datetime.fromisoformat(game_time.replace("Z", "+00:00")).replace(minute=0, second=0, microsecond=0)
        times = [datetime.fromisoformat(t).replace(tzinfo=target.tzinfo) for t in hourly["time"]]
        index = min(range(len(times)), key=lambda i: abs(times[i] - target))
        return {
            "temperature_2m": float(hourly["temperature_2m"][index]),
            "wind_speed_10m": float(hourly["wind_speed_10m"][index]),
        }
    except Exception:
        return {"temperature_2m": 70.0, "wind_speed_10m": 5.0}


def read_manual_odds(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "player_name", "american_odds", "book"])
    odds = pd.read_csv(path)
    required = {"date", "player_name", "american_odds"}
    missing = required.difference(odds.columns)
    if missing:
        raise ValueError(f"Manual odds file is missing columns: {sorted(missing)}")
    if "book" not in odds.columns:
        odds["book"] = "manual"
    return odds
