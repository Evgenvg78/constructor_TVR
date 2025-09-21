"""Набор утилит для чтения и преобразования таблиц TVR."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["TVR_transform", "TVR_transform_local", "TVR_asis"]

PathLike = Union[str, Path]

ENV_SEC_DATA_FILE = "TVR_SEC_DATA_FILE"
DEFAULT_SEC_DATA_CANDIDATES: Sequence[Path] = (
    Path(__file__).with_name("sec_tvr.csv"),
    Path.home() / "MyDrive" / "work_data" / "TVR" / "sec_tvr.csv",
    Path("G:/MyDrive/work_data/TVR/sec_tvr.csv"),
)

FILLER_ROW_INDEX = 9999
FILLER_COLUMN_COUNT = 56
V_PREFIX = "V"
W_PREFIX = "W"


def _range_string_to_list(range_expression: str) -> List[int]:
    """Преобразует строку вида '3-5,7,9' в список целых чисел."""
    result: List[int] = []
    for chunk in range_expression.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start, end = int(start_str), int(end_str)
            if start > end:
                raise ValueError(f"Неверный диапазон '{chunk}': начало больше конца.")
            result.extend(range(start, end + 1))
        else:
            result.append(int(chunk))
    return result


def _parse_variant_types(path: PathLike) -> pd.DataFrame:
    """Читает файл конфигурации вариантов и раскрывает диапазоны строк."""
    variant_frame = pd.read_csv(path, sep="=", header=0)
    if "variant" not in variant_frame.columns:
        raise ValueError("В файле вариантов отсутствует колонка 'variant'.")
    variant_frame = variant_frame.copy()
    variant_frame["variant"] = (
        variant_frame["variant"].astype(str).apply(_range_string_to_list)
    )
    variant_frame = variant_frame.explode("variant").dropna(subset=["variant"])
    variant_frame["variant"] = variant_frame["variant"].astype(int)
    return variant_frame


def _resolve_sec_data_path(explicit_path: Optional[PathLike]) -> Path:
    """Возвращает путь к справочнику инструментов, проверяя несколько вариантов."""
    candidates: List[Optional[Path]] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    env_value = os.getenv(ENV_SEC_DATA_FILE)
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(DEFAULT_SEC_DATA_CANDIDATES)
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser()
        if resolved.exists():
            return resolved
    joined_candidates = ", ".join(str(c) for c in candidates if c is not None)
    raise FileNotFoundError(
        "Не удалось найти файл со справочными данными SEC. "
        f"Проверьте путь: {joined_candidates or 'путь не указан'}."
    )


def _load_sec_reference(sec_data_file: Optional[PathLike]) -> pd.DataFrame:
    """Читает данные по инструментам и добавляет столбец 'full_price'."""
    path = _resolve_sec_data_path(sec_data_file)
    sec_data = pd.read_csv(path, sep=",", index_col=False)
    required_columns = {
        "SECID",
        "MINSTEP",
        "STEPPRICE",
        "PREVSETTLEPRICE",
        "INITIALMARGIN",
        "BUYSELLFEE",
        "SCALPERFEE",
    }
    missing = required_columns - set(sec_data.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"В файле '{path}' отсутствуют колонки: {missing_list}."
        )
    sec_data = sec_data.copy()
    sec_data["full_price"] = (
        sec_data["PREVSETTLEPRICE"] / sec_data["MINSTEP"] * sec_data["STEPPRICE"]
    )
    return sec_data[
        [
            "SECID",
            "MINSTEP",
            "STEPPRICE",
            "PREVSETTLEPRICE",
            "INITIALMARGIN",
            "BUYSELLFEE",
            "SCALPERFEE",
            "full_price",
        ]
    ]


def _load_layout_table(tvr_path: PathLike) -> pd.DataFrame:
    """Читает layout-файл TVR и возвращает базовую таблицу."""
    path = Path(tvr_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл layout '{path}' не найден.")
    with path.open(encoding="utf8") as fh:
        raw_lines = fh.readlines()
    if not raw_lines:
        raise ValueError(f"Файл '{path}' пуст.")
    first_line = raw_lines[0].strip()
    if not first_line:
        raise ValueError(f"Первая строка файла '{path}' пуста.")
    separator = first_line[0]
    header_segments = first_line.split(separator)
    column_names = [segment for segment in header_segments[2:] if segment]
    if not column_names:
        column_names = [segment for segment in header_segments if segment]
    data: List[List[str]] = []
    reader = csv.reader(raw_lines[1:], delimiter=" ")
    for row in reader:
        if not row:
            continue
        if len(row) > 2:
            row = [row[0], row[1], " ".join(row[2:])]
        if len(row) == 3:
            data.append(row)
    if not data:
        raise ValueError(f"Файл '{path}' не содержит строк с данными.")
    df = pd.DataFrame(data, columns=["stroka", "stolbec", "data"])
    df["stroka"] = df["stroka"].astype(int)
    df["stolbec"] = df["stolbec"].astype(int)
    filler = pd.DataFrame(
        {
            "stroka": [FILLER_ROW_INDEX] * FILLER_COLUMN_COUNT,
            "stolbec": list(range(1, FILLER_COLUMN_COUNT + 1)),
            "data": ["**"] * FILLER_COLUMN_COUNT,
        }
    )
    df = pd.concat([df, filler], ignore_index=True)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    pivot = df.pivot(index="stroka", columns="stolbec", values="data")
    if column_names:
        if len(column_names) != len(pivot.columns):
            raise ValueError(
                "Число столбцов в данных не совпадает с заголовком. "
                f"Данные: {len(pivot.columns)}, заголовок: {len(column_names)}."
            )
        pivot.columns = column_names
    pivot.reset_index(inplace=True)
    return pivot


def _build_transformed_table(
    base_table: pd.DataFrame,
    variants: pd.DataFrame,
    sec_reference: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Преобразует базовую таблицу, добавляя справочную информацию и расчёты."""
    tvr = base_table.copy()
    tvr["stroka"] = tvr["stroka"].astype(int)
    required_columns = {"Start", "Kill all", "Mode"}
    missing_layout = required_columns - set(tvr.columns)
    if missing_layout:
        missing_list = ", ".join(sorted(missing_layout))
        raise KeyError(
            f"В layout-файле отсутствуют обязательные колонки: {missing_list}."
        )
    tvr = tvr[(tvr["Start"] == "True") & (tvr["Kill all"] != "True")].copy()
    if tvr.empty:
        return tvr
    if "V 0" not in tvr.columns:
        raise KeyError("В таблице нет ожидаемой колонки 'V 0'.")
    v0_position = tvr.columns.get_loc("V 0")
    tvr.insert(v0_position + 1, "W 0", np.nan)
    col_names = tvr.columns.tolist()
    reshaped = tvr.reset_index()
    col_numbers = list(range(1, len(reshaped.columns)))
    col_numbers.insert(0, "stroka")
    reshaped.columns = col_numbers
    melted = pd.melt(reshaped, id_vars=["stroka"], var_name="stolbec", value_name="data")
    melted = melted.sort_values(by=["stroka", "stolbec"])
    merged = melted.merge(sec_reference, how="left", left_on="data", right_on="SECID")
    if value_column not in merged.columns:
        raise KeyError(f"В справочнике нет колонки '{value_column}'.")
    merged[value_column] = merged[value_column].shift(2)
    merged["data"] = np.where(
        merged[value_column] > 0, merged[value_column], merged["data"]
    )
    pivot_source = merged[["stroka", "stolbec", "data"]]
    pivot = pivot_source.pivot(index="stroka", columns="stolbec", values="data")
    pivot.columns = col_names
    pivot.reset_index(drop=True, inplace=True)
    pivot = pivot.sort_values(by=["Mode", "stroka"])
    pivot["__order"] = pivot.groupby("Mode").cumcount()
    pivot = pivot.sort_values(by=["__order", "stroka"]).drop(columns="__order")
    pivot["variant_start"] = np.where(
        pivot["Mode"] == "1", pivot["stroka"].astype(str), np.nan
    )
    pivot["variant_end"] = np.where(
        pivot["Mode"] == "-1", pivot["stroka"].astype(str), np.nan
    )
    pivot["variant_start"] = pivot["variant_start"].ffill()
    pivot["variant_end"] = pivot["variant_end"].bfill()
    pivot["variant_range"] = (
        pivot["variant_start"].astype(str) + "-" + pivot["variant_end"].astype(str)
    )
    pivot.drop(columns=["variant_start", "variant_end"], inplace=True)
    v_columns = [col for col in pivot.columns if col.startswith(V_PREFIX)]
    w_columns = [col for col in pivot.columns if col.startswith(W_PREFIX)]
    if len(v_columns) != len(w_columns):
        raise ValueError(
            "Количество столбцов с префиксом 'V' и 'W' не совпадает."
        )
    for column in v_columns + w_columns:
        pivot[column] = pivot[column].fillna(0).astype(int)
    pivot["total_GO"] = sum(
        abs(pivot[v_col]) * pivot[w_col] for v_col, w_col in zip(v_columns, w_columns)
    )
    result = pivot.merge(variants, how="left", left_on="stroka", right_on="variant")
    if "variant" in result.columns:
        result = result.rename(columns={"variant": "variant_id"})
    required_params = ["C", "N", "P", "E", "FrId", "MoveN"]
    missing_params = [column for column in required_params if column not in result.columns]
    if missing_params:
        missing_list = ", ".join(sorted(missing_params))
        raise KeyError(
            f"Не хватает колонок для формирования PARAMS: {missing_list}."
        )
    result["PARAMS"] = (
        result["C"].astype(str)
        + "_"
        + result["N"].astype(str)
        + "_"
        + result["P"].astype(str)
        + "_"
        + result["E"].astype(str)
        + "_"
        + result["FrId"].astype(str)
        + "_"
        + result["MoveN"].astype(str)
    )
    return result


def TVR_transform(
    TVR: PathLike,
    varyant_types: PathLike,
    type: str = "Go",
    sec_data_file: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Возвращает таблицу TVR с расчётами GO/стоимости и раскрытыми вариантами."""
    value_column = "INITIALMARGIN" if type == "Go" else "full_price"
    base_table = _load_layout_table(TVR)
    variants = _parse_variant_types(varyant_types)
    sec_reference = _load_sec_reference(sec_data_file)
    return _build_transformed_table(base_table, variants, sec_reference, value_column)


def TVR_transform_local(
    TVR: PathLike,
    varyant_types: PathLike,
    type: str = "Go",
    sec_data_file: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Обёртка для обратной совместимости: использует локальный путь по умолчанию."""
    default_local_path = sec_data_file or Path("G:/MyDrive/work_data/TVR/sec_tvr.csv")
    return TVR_transform(
        TVR=TVR,
        varyant_types=varyant_types,
        type=type,
        sec_data_file=default_local_path,
    )


def TVR_asis(TVR: PathLike) -> pd.DataFrame:
    """Возвращает базовую таблицу без дополнительных расчётов."""
    return _load_layout_table(TVR)
