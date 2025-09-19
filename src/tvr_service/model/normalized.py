"""Normalized TVR representations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from ..io.tvr_io import TVRFile, read_tvr2

REFERENCE_COLUMNS: tuple[str, ...] = (
    "InL1",
    "InL2",
    "OutL1",
    "OutL2",
    "secIn",
    "secOut",
)


@dataclass(slots=True)
class TVRRow:
    """Normalized view of a single TVR row."""

    stroka: int
    values: Dict[str, Any] = field(default_factory=dict)
    references: Dict[str, tuple[Any, ...]] = field(default_factory=dict)

    def to_dict(self, *, include_empty: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {"stroka": self.stroka}
        for key, value in self.values.items():
            if value is None and not include_empty:
                continue
            data[key] = value
        if self.references:
            data["__references__"] = {
                key: list(values) for key, values in self.references.items()
            }
        return data

    def linked_strokes(self) -> Dict[str, tuple[int, ...]]:
        result: Dict[str, tuple[int, ...]] = {}
        for key, values in self.references.items():
            numeric = tuple(v for v in values if isinstance(v, int))
            if numeric:
                result[key] = numeric
        return result


@dataclass(slots=True)
class TVRNormalized:
    """Normalized TVR graph keyed by `stroka`."""

    rows: Dict[int, TVRRow]
    separator: str
    triple_separator: str
    columns: Sequence[str]

    def get(self, stroka: int) -> TVRRow | None:
        return self.rows.get(stroka)

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for stroka, row in sorted(self.rows.items()):
            record = {"stroka": stroka}
            record.update(row.values)
            data.append(record)
        # Ensure all expected columns exist
        df = pd.DataFrame(data)
        missing = [col for col in self.columns if col not in df.columns]
        for col in missing:
            df[col] = pd.NA
        return df[["stroka", *self.columns]]

    def export_excel(self, path: str | Path) -> None:
        df = self.to_dataframe()
        df.to_excel(path, index=False)

    def to_dict(self) -> Dict[int, Dict[str, Any]]:
        return {stroka: row.to_dict() for stroka, row in self.rows.items()}


def normalize_tvr(
    tvr: TVRFile,
    *,
    reference_columns: Iterable[str] = REFERENCE_COLUMNS,
) -> TVRNormalized:
    reference_set = tuple(reference_columns)

    columns = [col for col in tvr.dataframe.columns if col != "stroka"]
    rows: Dict[int, TVRRow] = {}

    for _, raw_row in tvr.dataframe.iterrows():
        stroka = int(raw_row["stroka"])
        values: Dict[str, Any] = {}
        references: Dict[str, tuple[Any, ...]] = {}

        for column in columns:
            raw_value = raw_row[column]
            value = _clean_value(raw_value)
            values[column] = value

            if column in reference_set:
                parsed = _parse_references(value)
                if parsed:
                    references[column] = parsed

        rows[stroka] = TVRRow(stroka=stroka, values=values, references=references)

    return TVRNormalized(
        rows=rows,
        separator=tvr.separator,
        triple_separator=tvr.triple_separator,
        columns=columns,
    )


def load_and_normalize(
    path: str | Path,
    *,
    reference_columns: Iterable[str] = REFERENCE_COLUMNS,
    export_excel: str | Path | None = None,
) -> TVRNormalized:
    tvr = read_tvr2(path)
    normalized = normalize_tvr(tvr, reference_columns=reference_columns)
    if export_excel is not None:
        normalized.export_excel(export_excel)
    return normalized


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        numeric = _try_parse_number(stripped)
        return numeric if numeric is not None else stripped
    return value


def _try_parse_number(text: str) -> Any:
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        try:
            return int(text)
        except ValueError:
            return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value


def _parse_references(value: Any) -> tuple[Any, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return tuple()
        if float(value).is_integer():
            return (int(value),)
        return (float(value),)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        parsed: list[Any] = []
        for part in parts:
            parsed_value = _clean_value(part)
            parsed.append(parsed_value if parsed_value is not None else part)
        return tuple(parsed)
    return (value,)
