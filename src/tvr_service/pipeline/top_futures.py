"""Utilities for retrieving the most liquid FORTS futures from MOEX ISS."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
import requests

from .portfolio import load_securities, load_whitelist

BASE_URL = "https://iss.moex.com"
HIST_ENDPOINT = "/iss/history/engines/futures/markets/forts/securities.json"
SHARES_ENDPOINT = "/iss/engines/stock/markets/shares/securities.json"
DESCRIPTION_ENDPOINT_TEMPLATE = "/iss/securities/{secid}.json"
DEFAULT_LIMIT = 1000
MAX_RETRIES = 5
RETRY_DELAY = 1.0
SHARES_PAGE_LIMIT = 1000
SHARES_SLEEP = 0.15

ENV_SHARE_UNIVERSE_FILE = "TVR_SHARE_UNIVERSE_FILE"
DEFAULT_SHARE_UNIVERSE_PATH = Path(__file__).with_name("data") / "share_universe.csv"

ALLOWED_SHARE_INSTRUMENT_IDS = {"EQIN"}
ALLOWED_SHARE_SECURITY_TYPES = {"1", "2"}

PathLike = Union[str, Path]

TOP_FUTURES_COLUMNS = (
    "base_code",
    "SECID",
    "VALUE",
    "VOLUME",
    "NUMTRADES",
    "last_price",
    "last_price_avg",
    "guarantee_deposit",
    "guarantee_deposit_avg",
)


def _empty_top_futures_frame() -> pd.DataFrame:
    """Return consistent schema for empty top futures results."""

    return pd.DataFrame(columns=TOP_FUTURES_COLUMNS)


def _iss_get(url_path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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


def _fetch_shortnames(secids: Iterable[str]) -> Dict[str, str]:
    """Return SHORTNAME values for given SECIDs via ISS description endpoint."""

    result: Dict[str, str] = {}
    unique_secids = {str(secid).strip() for secid in secids if str(secid).strip()}
    if not unique_secids:
        return result
    for secid in unique_secids:
        try:
            payload = _iss_get(
                DESCRIPTION_ENDPOINT_TEMPLATE.format(secid=secid),
                params={"iss.only": "description"},
            )
        except RuntimeError:
            continue
        description_section = payload.get("description", {})
        description = _json_table_to_df(description_section)
        if description.empty or "name" not in description.columns:
            continue
        match = description.loc[description["name"] == "SHORTNAME", "value"]
        if not match.empty:
            result[secid] = str(match.iloc[0])
    return result


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
    top_n: Optional[int] = None,
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
        .reset_index(drop=True)
    )
    if "CLOSE" in filtered.columns and "TRADEDATE" in filtered.columns:
        last_close = (
            filtered.sort_values(by=["SECID", "TRADEDATE"])
            .groupby("SECID")["CLOSE"]
            .last()
            .rename("last_price")
        )
        aggregated = aggregated.merge(last_close, on="SECID", how="left")
    else:
        aggregated["last_price"] = pd.NA
    if top_n is not None:
        aggregated = aggregated.head(top_n)
    return aggregated


def _attach_metadata(
    futures_df: pd.DataFrame,
    sec_reference: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Join futures with reference table to expose SHORTNAME and base_code."""

    if futures_df.empty:
        return futures_df
    reference = sec_reference
    if reference is None:
        reference = load_securities()
    available_columns = [
        column
        for column in ("SECID", "SHORTNAME", "base_code")
        if column in reference.columns
    ]
    mapping = reference[available_columns].drop_duplicates(subset=["SECID"])
    enriched = futures_df.merge(mapping, on="SECID", how="left")

    missing_shortname = enriched["SHORTNAME"].isna()
    if missing_shortname.any():
        fetched = _fetch_shortnames(enriched.loc[missing_shortname, "SECID"].unique())
        if fetched:
            enriched.loc[missing_shortname, "SHORTNAME"] = (
                enriched.loc[missing_shortname, "SECID"].map(fetched)
            )

    if "base_code" not in enriched.columns:
        enriched["base_code"] = pd.NA
    needs_base_code = enriched["base_code"].isna() | (enriched["base_code"].astype(str) == "")
    if needs_base_code.any():
        enriched.loc[needs_base_code, "base_code"] = (
            enriched.loc[needs_base_code, "SHORTNAME"]
            .astype(str)
            .str.split("-", n=1)
            .str[0]
            .str.strip()
        )
    enriched["base_code"] = enriched["base_code"].fillna(enriched["SECID"])
    enriched["base_code"] = enriched["base_code"].astype(str).str.upper()
    return enriched


def _pick_sector_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("SECTOR", "SECTORID", "INDUSTRY", "GROUP"):
        if candidate in df.columns:
            return candidate
    return None


def fetch_shares_securities_full(
    is_trading: Optional[int] = 1,
    lang: str = "ru",
    page_limit: int = SHARES_PAGE_LIMIT,
    sleep_sec: float = SHARES_SLEEP,
) -> pd.DataFrame:
    """Fetch full MOEX shares securities table with pagination."""

    frames: List[pd.DataFrame] = []
    start = 0
    while True:
        params: Dict[str, Any] = {
            "iss.only": "securities",
            "start": start,
            "limit": page_limit,
            "lang": lang,
        }
        if is_trading is not None:
            params["is_trading"] = is_trading
        payload = _iss_get(SHARES_ENDPOINT, params)
        section = payload.get("securities", {})
        frame = _json_table_to_df(section)
        if frame.empty:
            break
        frame.columns = [str(col).upper() for col in frame.columns]
        frames.append(frame)
        if len(frame) < page_limit:
            break
        start += page_limit
        time.sleep(sleep_sec)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _resolve_share_universe_path(path: Optional[PathLike]) -> Path:
    if path is not None:
        return Path(path).expanduser()
    env_value = os.getenv(ENV_SHARE_UNIVERSE_FILE)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_SHARE_UNIVERSE_PATH


def _read_share_universe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "SECID" not in df.columns:
        raise ValueError(f"Share universe file is missing SECID column: {path}")
    for column in ("SECID", "SECTOR", "INSTRID", "SECTYPE"):
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
    df = df[df["SECID"] != ""].drop_duplicates(subset=["SECID"]).reset_index(drop=True)
    return df


def _filter_share_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """Limit the share universe to equity instruments only."""

    if df.empty:
        return df
    if "INSTRID" in df.columns:
        mask = df["INSTRID"].astype(str).str.upper().isin(ALLOWED_SHARE_INSTRUMENT_IDS)
        filtered = df[mask]
        if not filtered.empty:
            return filtered
    if "SECTYPE" in df.columns:
        mask = df["SECTYPE"].astype(str).str.upper().isin(ALLOWED_SHARE_SECURITY_TYPES)
        filtered = df[mask]
        if not filtered.empty:
            return filtered
    return df


def _prepare_share_universe(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize and filter the share universe response from MOEX ISS."""

    if raw.empty:
        return pd.DataFrame(columns=["SECID", "SECTOR"])
    filtered = _filter_share_instruments(raw)
    sector_column = _pick_sector_column(filtered)
    columns = ["SECID"]
    if sector_column:
        columns.append(sector_column)
    metadata_columns = [column for column in ("INSTRID", "SECTYPE") if column in filtered.columns]
    columns.extend(metadata_columns)
    result = filtered.loc[:, columns].copy()
    if sector_column and sector_column != "SECTOR":
        result = result.rename(columns={sector_column: "SECTOR"})
    if "SECTOR" not in result.columns:
        result["SECTOR"] = pd.NA
    for column in ("SECID", "SECTOR", "INSTRID", "SECTYPE"):
        if column in result.columns:
            result[column] = result[column].astype(str).str.strip()
    result = result[result["SECID"] != ""].drop_duplicates(subset=["SECID"]).reset_index(drop=True)
    ordered = ["SECID"]
    if "SECTOR" in result.columns:
        ordered.append("SECTOR")
    for column in ("INSTRID", "SECTYPE"):
        if column in result.columns:
            ordered.append(column)
    return result.loc[:, ordered]


def load_share_universe(
    is_trading: Optional[int] = 1,
    lang: str = "ru",
    share_universe_path: Optional[PathLike] = None,
    use_cache: bool = True,
    refresh: bool = False,
    save_to_cache: bool = True,
) -> pd.DataFrame:
    """Return DataFrame with SECID of MOEX shares (optionally cached to disk)."""

    cache_path = _resolve_share_universe_path(share_universe_path)
    cached_df: Optional[pd.DataFrame] = None
    if use_cache and not refresh and cache_path.exists():
        try:
            cached_df = _read_share_universe(cache_path)
        except Exception:
            cached_df = None

    prepared: Optional[pd.DataFrame] = None
    if cached_df is not None and any(col in cached_df.columns for col in ("INSTRID", "SECTYPE")):
        prepared = _prepare_share_universe(cached_df)

    if prepared is None:
        raw = fetch_shares_securities_full(is_trading=is_trading, lang=lang)
        if raw.empty:
            if cached_df is not None:
                prepared = _prepare_share_universe(cached_df)
            else:
                return pd.DataFrame(columns=["SECID", "SECTOR"])
        else:
            if "SECID" not in raw.columns:
                raise RuntimeError("Share universe response does not contain SECID column.")
            prepared = _prepare_share_universe(raw)
            if save_to_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                prepared.to_csv(cache_path, index=False)

    columns_to_return = ["SECID"]
    if "SECTOR" in prepared.columns:
        columns_to_return.append("SECTOR")
    return prepared.loc[:, columns_to_return].reset_index(drop=True)


def _filter_equity_futures(
    futures_df: pd.DataFrame,
    share_universe: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Keep only futures whose base_code exists in the share universe."""

    if futures_df.empty or share_universe is None or share_universe.empty:
        return futures_df
    share_set = {
        str(code).upper()
        for code in share_universe["SECID"].dropna()
        if str(code).strip()
    }
    mask = futures_df["base_code"].astype(str).str.upper().isin(share_set)
    return futures_df[mask].reset_index(drop=True)


def _aggregate_by_base_code(
    futures_df: pd.DataFrame,
    sort_by: str,
) -> pd.DataFrame:
    """Group futures-level metrics by base_code and attach derived statistics."""

    if futures_df.empty:
        return futures_df
    aggregated = (
        futures_df.groupby("base_code", as_index=False)[["VALUE", "VOLUME", "NUMTRADES"]]
        .sum()
    )
    representative = (
        futures_df.reset_index()
        .sort_values(by=sort_by, ascending=False)
        .drop_duplicates(subset=["base_code"], keep="first")
        .set_index("base_code")
    )

    if "SECID" in representative.columns:
        aggregated["SECID"] = aggregated["base_code"].map(representative["SECID"])
    else:
        aggregated["SECID"] = pd.NA

    if "last_price" in futures_df.columns:
        if "last_price" in representative.columns:
            aggregated["last_price"] = aggregated["base_code"].map(representative["last_price"])
        else:
            aggregated["last_price"] = pd.NA
        avg_last_price = (
            futures_df.groupby("base_code")["last_price"]
            .mean()
            .rename("last_price_avg")
        )
        aggregated["last_price_avg"] = aggregated["base_code"].map(avg_last_price)
    else:
        aggregated["last_price"] = pd.NA
        aggregated["last_price_avg"] = pd.NA

    if "SELLDEPO" in futures_df.columns:
        if "SELLDEPO" in representative.columns:
            aggregated["guarantee_deposit"] = aggregated["base_code"].map(representative["SELLDEPO"])
        else:
            aggregated["guarantee_deposit"] = pd.NA
        avg_margin = (
            futures_df.groupby("base_code")["SELLDEPO"]
            .mean()
            .rename("guarantee_deposit_avg")
        )
        aggregated["guarantee_deposit_avg"] = aggregated["base_code"].map(avg_margin)
    else:
        aggregated["guarantee_deposit"] = pd.NA
        aggregated["guarantee_deposit_avg"] = pd.NA
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
    only_equities: bool = False,
    share_universe: Optional[pd.DataFrame] = None,
    shares_is_trading: Optional[int] = 1,
    shares_lang: str = "ru",
    share_universe_path: Optional[PathLike] = None,
    shares_use_cache: bool = True,
    shares_refresh: bool = False,
    sec_reference: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load, rank, and optionally filter FORTS futures."""

    raw = fetch_forts_history(
        date=date,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    if raw.empty:
        return _empty_top_futures_frame()

    ranking_limit: Optional[int] = None if only_equities else top_n
    ranked = rank_top_futures(
        raw,
        sort_by=sort_by,
        top_n=ranking_limit,
        allowed_secid=allowed_secid,
    )
    if ranked.empty:
        return _empty_top_futures_frame()

    enriched = _attach_metadata(ranked, sec_reference)

    if only_equities:
        universe = share_universe
        if universe is None:
            universe = load_share_universe(
                is_trading=shares_is_trading,
                lang=shares_lang,
                share_universe_path=share_universe_path,
                use_cache=shares_use_cache,
                refresh=shares_refresh,
                save_to_cache=True,
            )
        enriched = _filter_equity_futures(enriched, universe)
        if enriched.empty:
            return _empty_top_futures_frame()

    aggregated_by_base = _aggregate_by_base_code(enriched, sort_by)
    if aggregated_by_base.empty:
        return _empty_top_futures_frame()
    aggregated_by_base = aggregated_by_base.sort_values(by=sort_by, ascending=False)
    if top_n is not None:
        aggregated_by_base = aggregated_by_base.head(top_n)

    for column in TOP_FUTURES_COLUMNS:
        if column not in aggregated_by_base.columns:
            aggregated_by_base[column] = pd.NA
    ordered = aggregated_by_base.loc[:, TOP_FUTURES_COLUMNS]
    return ordered.reset_index(drop=True)


def get_portfolio_whitelist_futures(
    contract_suffix: Optional[str],
    *,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    whitelist: Optional[Iterable[str]] = None,
    whitelist_path: Optional[PathLike] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return futures stats for whitelist entries with optional SECID suffix override."""

    columns_order = [
        "base_code",
        "SECID_original",
        "SECID",
        "last_price_avg",
        "guarantee_deposit_avg",
        "last_price",
        "guarantee_deposit",
        "VALUE",
        "VOLUME",
        "NUMTRADES",
    ]

    whitelist_values: Iterable[str]
    if whitelist is None:
        whitelist_values = load_whitelist(whitelist_path)
    else:
        whitelist_values = whitelist

    allowed_base_codes = {
        str(code).strip().upper()
        for code in whitelist_values
        if str(code).strip()
    }
    if not allowed_base_codes:
        return pd.DataFrame(columns=columns_order)

    inner_kwargs = dict(kwargs)
    inner_kwargs["top_n"] = None
    futures = get_top_futures(
        date=date,
        start_date=start_date,
        end_date=end_date,
        **inner_kwargs,
    )

    if futures.empty:
        return pd.DataFrame(columns=columns_order)

    mask = futures["base_code"].astype(str).str.upper().isin(allowed_base_codes)
    filtered = futures.loc[mask].copy()
    if filtered.empty:
        return pd.DataFrame(columns=columns_order)

    filtered["SECID_original"] = filtered["SECID"]

    normalized_suffix = ""
    if contract_suffix is not None:
        normalized_suffix = str(contract_suffix).strip().upper()

    def _override_suffix(secid: Any) -> Any:
        if not isinstance(secid, str):
            return secid
        cleaned = secid.strip()
        if not cleaned:
            return cleaned
        if not normalized_suffix:
            if len(cleaned) >= 2 and cleaned[-1].isdigit():
                return cleaned[:-2]
            return cleaned
        if len(cleaned) >= 2 and cleaned[-1].isdigit():
            return cleaned[:-2] + normalized_suffix
        return cleaned

    filtered["SECID"] = filtered["SECID"].apply(_override_suffix)

    for column in columns_order:
        if column not in filtered.columns:
            filtered[column] = pd.NA

    return filtered.loc[:, columns_order].reset_index(drop=True)


__all__ = [
    "fetch_forts_history",
    "fetch_shares_securities_full",
    "get_portfolio_whitelist_futures",
    "get_top_futures",
    "load_share_universe",
    "rank_top_futures",
]
