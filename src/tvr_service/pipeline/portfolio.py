"""Utilities for building a basic securities portfolio allocation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests

PathLike = Union[str, Path]

DEFAULT_MOEX_URL = (
    "https://iss.moex.com/iss/engines/futures/markets/forts/boards/RFUD/securities.json"
)
ENV_SEC_DATA_FILE = "TVR_SEC_DATA_FILE"
DEFAULT_SEC_DATA_CANDIDATES: Sequence[Path] = (
    Path("sec_tvr.csv"),
    Path(__file__).resolve().parents[3] / "sec_tvr.csv",
    Path.home() / "MyDrive" / "work_data" / "TVR" / "sec_tvr.csv",
    Path("G:/MyDrive/work_data/TVR/sec_tvr.csv"),
)
DEFAULT_WHITELIST_PATH = Path(__file__).with_name("data") / "portfolio_whitelist.txt"

REFERENCE_COLUMNS = (
    "SECID",
    "MINSTEP",
    "STEPPRICE",
    "PREVSETTLEPRICE",
    "INITIALMARGIN",
    "BUYSELLFEE",
    "SCALPERFEE",
)


@dataclass(frozen=True)
class PortfolioEntry:
    """Represents a planned allocation for a single security."""

    secid: str
    allocation: float
    estimated_lots: int
    full_price: float


def fetch_remote_securities(url: str = DEFAULT_MOEX_URL) -> pd.DataFrame:
    """Download the latest securities table from the MOEX ISS API."""

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    columns = payload["securities"]["columns"]
    data = payload["securities"]["data"]
    return pd.DataFrame(data, columns=columns)


def load_local_securities(path: PathLike) -> pd.DataFrame:
    """Load securities data from a CSV file exported earlier."""

    return pd.read_csv(Path(path), sep=",", index_col=False)


def _resolve_local_sec_file(sec_data_file: Optional[PathLike]) -> Optional[Path]:
    if sec_data_file is not None:
        candidate = Path(sec_data_file).expanduser()
        if candidate.exists():
            return candidate
    env_value = os.getenv(ENV_SEC_DATA_FILE)
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists():
            return candidate
    for candidate in DEFAULT_SEC_DATA_CANDIDATES:
        resolved = candidate.expanduser()
        if resolved.exists():
            return resolved
    return None


def load_securities(
    sec_data_file: Optional[PathLike] = None,
    prefer_remote: bool = True,
) -> pd.DataFrame:
    """Return a securities table ready for filtering and allocation."""

    errors: List[str] = []
    reference: Optional[pd.DataFrame] = None
    if prefer_remote:
        try:
            reference = fetch_remote_securities()
        except requests.RequestException as exc:
            errors.append(f"remote fetch failed: {exc}")
    if reference is None:
        local_path = _resolve_local_sec_file(sec_data_file)
        if local_path is None:
            joined = "; ".join(errors) or "no attempts"
            raise RuntimeError(
                "Unable to load securities data: remote fetch unavailable and no local file found ("
                + joined
                + ")."
            )
        reference = load_local_securities(local_path)
    return _prepare_reference(reference)



def _prepare_reference(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REFERENCE_COLUMNS if column not in raw.columns]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Input data is missing required columns: {missing_list}")
    reference = raw[list(REFERENCE_COLUMNS)].copy()
    if "SHORTNAME" in raw.columns:
        reference["SHORTNAME"] = raw["SHORTNAME"]
    reference["full_price"] = (
        reference["PREVSETTLEPRICE"] / reference["MINSTEP"] * reference["STEPPRICE"]
    )
    reference["CODE"] = reference["SECID"].astype(str)
    reference["PRICE"] = pd.to_numeric(reference["PREVSETTLEPRICE"], errors="coerce").fillna(0.0)
    reference["SELLDEPO"] = pd.to_numeric(reference["INITIALMARGIN"], errors="coerce").fillna(0.0)
    if "SHORTNAME" in reference.columns:
        reference["base_code"] = (
            reference["SHORTNAME"].astype(str).str.split("-", n=1).str[0].str.strip()
        )
    else:
        reference["base_code"] = reference["CODE"]
    tb_rows = reference[reference["SECID"].astype(str).str.startswith("TB")].copy()
    if not tb_rows.empty:
        ti_rows = tb_rows.copy()
        ti_rows["SECID"] = ti_rows["SECID"].str.replace("^TB", "TI", regex=True)
        ti_rows["CODE"] = ti_rows["SECID"]
        if "SHORTNAME" in ti_rows.columns:
            ti_rows["SHORTNAME"] = ti_rows["SHORTNAME"].str.replace("^TB", "TI", regex=True)
            ti_rows["base_code"] = (
                ti_rows["SHORTNAME"].astype(str).str.split("-", n=1).str[0].str.strip()
            )
        else:
            ti_rows["base_code"] = ti_rows["CODE"]
        reference = pd.concat([reference, ti_rows], ignore_index=True)
    reference = reference.drop_duplicates(subset=["SECID"]).reset_index(drop=True)
    return reference



def load_whitelist(path: Optional[PathLike] = None) -> List[str]:
    """Read the preferred securities universe from a text file."""

    resolved_path = Path(path) if path else DEFAULT_WHITELIST_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Whitelist file not found: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    return entries


def filter_by_suffix(data: pd.DataFrame, suffix: Optional[str]) -> pd.DataFrame:
    """Filter securities whose SECID ends with the provided suffix."""

    if suffix is None:
        return data
    mask = data["SECID"].astype(str).str.endswith(suffix)
    return data[mask].reset_index(drop=True)



def filter_by_whitelist(
    data: pd.DataFrame,
    whitelist: Optional[Iterable[str]],
) -> pd.DataFrame:
    """Restrict securities to those contained in the whitelist."""

    if whitelist is None:
        return data
    whitelist_set = {item.strip().upper() for item in whitelist if item}
    if not whitelist_set:
        return data
    mask = pd.Series(False, index=data.index)
    candidate_columns = [
        column
        for column in ("base_code", "SHORTNAME", "CODE", "SECID")
        if column in data.columns
    ]
    if not candidate_columns:
        return data
    for column in candidate_columns:
        mask = mask | data[column].astype(str).str.upper().isin(whitelist_set)
    return data[mask].reset_index(drop=True)



def build_portfolio(
    capital: float,
    suffix: Optional[str] = None,
    whitelist: Optional[Sequence[str]] = None,
    whitelist_path: Optional[PathLike] = None,
    sec_data_file: Optional[PathLike] = None,
    prefer_remote: bool = True,
) -> List[PortfolioEntry]:
    """Construct a proportional portfolio allocation for the selected securities."""

    if capital <= 0:
        raise ValueError("Capital must be positive.")
    reference = load_securities(sec_data_file=sec_data_file, prefer_remote=prefer_remote)
    filtered = filter_by_suffix(reference, suffix)
    if whitelist is not None:
        filtered = filter_by_whitelist(filtered, whitelist)
    else:
        try:
            loaded_whitelist = load_whitelist(whitelist_path)
        except FileNotFoundError:
            loaded_whitelist = None
        if loaded_whitelist:
            candidate = filter_by_whitelist(filtered, loaded_whitelist)
            if not candidate.empty:
                filtered = candidate
    if filtered.empty:
        raise ValueError("No securities available after applying filters.")
    allocation_per_security = float(capital) / len(filtered)
    allocations: List[PortfolioEntry] = []
    for _, row in filtered.iterrows():
        full_price = float(row.get("full_price", 0) or 0)
        if full_price > 0:
            estimated_lots = int(np.floor(allocation_per_security / full_price))
        else:
            estimated_lots = 0
        allocations.append(
            PortfolioEntry(
                secid=str(row["SECID"]),
                allocation=allocation_per_security,
                estimated_lots=estimated_lots,
                full_price=full_price,
            )
        )
    return allocations


def allocations_to_frame(entries: Sequence[PortfolioEntry]) -> pd.DataFrame:
    """Convert portfolio entries into a pandas DataFrame for further processing."""

    frame = pd.DataFrame(
        {
            "SECID": [entry.secid for entry in entries],
            "allocation": [entry.allocation for entry in entries],
            "estimated_lots": [entry.estimated_lots for entry in entries],
            "full_price": [entry.full_price for entry in entries],
        }
    )
    frame["used_capital"] = frame["estimated_lots"] * frame["full_price"]
    frame["unused_capital"] = frame["allocation"] - frame["used_capital"]
    return frame


def build_portfolio_advanced(
    data: pd.DataFrame,
    capital: float,
    security_column: str = "Sec_0",
    price_column: str = "sec_0_price",
    param_num_column: str = "param_num",
    ensure_min_contract: bool = True,
) -> pd.DataFrame:
    """
    Построение портфеля с распределением капитала по группам инструментов и параметрам.
    
    Args:
        data: DataFrame с данными об инструментах и параметрах
        capital: Общий капитал для распределения
        security_column: Название столбца с инструментами для группировки
        price_column: Название столбца с ценами для расчета лотов
        param_num_column: Название столбца с номерами параметров в группе
        ensure_min_contract: Если True, гарантирует минимум 1 контракт каждому инструменту
    
    Returns:
        DataFrame с добавленными столбцами: allocation, estimated_lots, used_capital, unused_capital
    """
    if capital <= 0:
        raise ValueError("Капитал должен быть положительным числом")
    
    if data.empty:
        raise ValueError("DataFrame не может быть пустым")
    
    # Проверяем наличие необходимых столбцов
    required_columns = [security_column, price_column, param_num_column]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют столбцы: {missing_columns}")
    
    # Создаем копию данных для работы
    result_df = data.copy()
    
    # Инициализируем новые столбцы
    result_df["allocation"] = 0.0
    result_df["estimated_lots"] = 0
    result_df["used_capital"] = 0.0
    result_df["unused_capital"] = 0.0
    
    # Группируем по инструментам
    grouped = result_df.groupby(security_column)
    
    # ЭТАП 1: Обычное распределение капитала (как в оригинальной логике)
    unique_securities = result_df[security_column].nunique()
    capital_per_security = capital / unique_securities
    
    for security_name, group in grouped:
        # Сортируем группу по номеру параметра
        group_sorted = group.sort_values(param_num_column)
        
        # Получаем цену инструмента (берем из первой строки)
        instrument_price = float(group_sorted[price_column].iloc[0])
        
        if instrument_price <= 0:
            # Если цена 0 или отрицательная, пропускаем инструмент
            continue
        
        # Количество строк в группе
        param_count = len(group_sorted)

        if param_count == 0:
            continue

        # ЖАДНЫЙ АЛГОРИТМ: выдаем лоты, работая с общим бюджетом инструмента
        remaining_capital = capital_per_security
        lots_per_row = [0] * param_count  # Количество лотов для каждой строки

        while remaining_capital >= instrument_price:
            min_lots_idx = lots_per_row.index(min(lots_per_row))
            lots_per_row[min_lots_idx] += 1
            remaining_capital -= instrument_price

        used_per_row = [lots * instrument_price for lots in lots_per_row]
        total_used = sum(used_per_row)
        leftover_capital = max(capital_per_security - total_used, 0.0)

        for i, idx in enumerate(group_sorted.index):
            used_capital = used_per_row[i]
            result_df.at[idx, "allocation"] = used_capital
            result_df.at[idx, "estimated_lots"] = lots_per_row[i]
            result_df.at[idx, "used_capital"] = used_capital
            result_df.at[idx, "unused_capital"] = 0.0

        if leftover_capital > 0:
            leftover_idx = lots_per_row.index(min(lots_per_row))
            target_idx = group_sorted.index[leftover_idx]
            result_df.at[target_idx, "allocation"] += leftover_capital
            result_df.at[target_idx, "unused_capital"] = leftover_capital
    
    # ЭТАП 2: Если включен режим минимального контракта, добавляем по 1 контракту тем, кто остался без контрактов
    if ensure_min_contract:
        # Находим инструменты без контрактов
        instruments_with_contracts = result_df.groupby(security_column)['estimated_lots'].sum()
        instruments_without_contracts = instruments_with_contracts[instruments_with_contracts == 0].index.tolist()
        
        if instruments_without_contracts:
            for security_name in instruments_without_contracts:
                group = result_df[result_df[security_column] == security_name]
                group_sorted = group.sort_values(param_num_column)
                
                # Получаем цену инструмента
                instrument_price = float(group_sorted[price_column].iloc[0])
                
                if instrument_price > 0:
                    # Выделяем 1 контракт первому параметру этого инструмента
                    first_idx = group_sorted.index[0]
                    result_df.at[first_idx, "estimated_lots"] = 1
                    result_df.at[first_idx, "used_capital"] = instrument_price
                    result_df.at[first_idx, "allocation"] = instrument_price
                    result_df.at[first_idx, "unused_capital"] = 0.0
    
    return result_df

