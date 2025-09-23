"""Utilities for retrieving the most liquid FORTS futures from MOEX ISS."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

BASE_URL = "https://iss.moex.com"
HIST_ENDPOINT = "/iss/history/engines/futures/markets/forts/securities.json"
DEFAULT_LIMIT = 1000
MAX_RETRIES = 5
RETRY_DELAY = 1.0


def _iss_get(url_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform GET request to MOEX ISS with simple retry logic."""

    url = BASE_URL + url_path
    headers = {
        "User-Agent": "tvr-portfolio-agent",
        "Accept": "application/json",
    }
    last_exc: Optional[Exception] = None
    for _ in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network paths are hard to unit test
            last_exc = exc
            time.sleep(RETRY_DELAY)
    raise RuntimeError(f"MOEX ISS request failed after retries: {url} params={params}\n{last_exc}")


def _json_table_to_df(section: Dict[str, Any]) -> pd.DataFrame:
    """Convert ISS table section to pandas DataFrame."""

    if not section or not section.get("data"):
        return pd.DataFrame()
    columns = section.get("columns", [])
    rows = section.get("data", [])
    return pd.DataFrame(rows, columns=columns)


def fetch_forts_history(
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """Load raw FORTS history rows for the provided date or interval."""

    if date and (start_date or end_date):
        raise ValueError("Provide either date or date interval, not both.")

    params_base: Dict[str, Any] = {"limit": limit}
    frames: List[pd.DataFrame] = []

    def _fetch_for_one_day(date_str: str) -> None:
        start = 0
        while True:
            params = dict(params_base)
            params["start"] = start
            params["date"] = date_str
            payload = _iss_get(HIST_ENDPOINT, params)
            section = payload.get("history", {})
            frame = _json_table_to_df(section)
            if frame.empty:
                break
            frames.append(frame)
            if len(frame) < limit:
                break
            start += limit

    if date:
        if date == "today":
            from datetime import date as _date

            date = _date.today().isoformat()
        _fetch_for_one_day(date)
    else:
        from datetime import date as _date, timedelta

        if not start_date or not end_date:
            today = _date.today().isoformat()
            _fetch_for_one_day(today)
        else:
            d0 = _date.fromisoformat(start_date)
            d1 = _date.fromisoformat(end_date)
            if d1 < d0:
                raise ValueError("end_date must be greater than or equal to start_date")
            current = d0
            while current <= d1:
                _fetch_for_one_day(current.isoformat())
                current += timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    for column in ("VALUE", "VOLUME", "NUMTRADES"):
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0)
    if "TRADEDATE" in combined.columns:
        combined["TRADEDATE"] = pd.to_datetime(
            combined["TRADEDATE"], errors="coerce"
        ).dt.date
    return combined


def rank_top_futures(
    df_raw: pd.DataFrame,
    sort_by: str = "VALUE",
    top_n: int = 20,
    allowed_secid: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Aggregate by SECID and select the most liquid futures."""

    if df_raw.empty:
        return pd.DataFrame()
    if sort_by not in {"VALUE", "VOLUME", "NUMTRADES"}:
        raise ValueError("sort_by must be one of VALUE, VOLUME, NUMTRADES")
    required = {"SECID", "VALUE", "VOLUME", "NUMTRADES"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Raw dataframe is missing columns: {missing}")
    filtered = df_raw
    if allowed_secid is not None:
        allowed_set = {sec.upper() for sec in allowed_secid}
        filtered = filtered[filtered["SECID"].astype(str).str.upper().isin(allowed_set)]
    aggregated = (
        filtered.groupby("SECID", as_index=False)[["VALUE", "VOLUME", "NUMTRADES"]]
        .sum()
        .sort_values(by=sort_by, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return aggregated


def get_top_futures(
    *,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: str = "VALUE",
    top_n: int = 20,
    limit: int = DEFAULT_LIMIT,
    allowed_secid: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Convenience helper to load and rank FORTS futures in a single call."""

    raw = fetch_forts_history(
        date=date,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    if raw.empty:
        return raw
    return rank_top_futures(raw, sort_by=sort_by, top_n=top_n, allowed_secid=allowed_secid)


__all__ = [
    "fetch_forts_history",
    "get_top_futures",
    "rank_top_futures",
]
