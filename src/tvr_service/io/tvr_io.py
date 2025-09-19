"""IO utilities for working with .tvr2 files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_HEADER_SEPARATOR = "\u266a"
DEFAULT_TRIPLE_SEPARATOR = " "


@dataclass
class TVRFile:
    """Container for TVR2 data and its formatting metadata."""

    dataframe: pd.DataFrame
    separator: str = DEFAULT_HEADER_SEPARATOR
    triple_separator: str = DEFAULT_TRIPLE_SEPARATOR


class TVRIOError(RuntimeError):
    """Raised when a TVR2 file cannot be read or written."""


def read_tvr2(path: str | Path) -> TVRFile:
    """Load a TVR2 file and return its dataframe plus formatting metadata."""

    path = Path(path)
    separator, column_names = _read_header(path)
    triplets = list(iter_tvr2_triplets(path))

    if not triplets:
        df = pd.DataFrame(columns=["stroka", *column_names])
        return TVRFile(df, separator=separator)

    trip_df = pd.DataFrame(triplets, columns=["stroka", "stolbec", "data"])
    trip_df["stroka"] = pd.to_numeric(trip_df["stroka"], errors="raise", downcast="integer")
    trip_df["stolbec"] = pd.to_numeric(trip_df["stolbec"], errors="raise", downcast="integer")

    pivot = trip_df.pivot(index="stroka", columns="stolbec", values="data")
    full_cols = list(range(1, len(column_names) + 1))
    if full_cols:
        pivot = pivot.reindex(columns=full_cols)
        rename_map = {idx: name for idx, name in enumerate(column_names, start=1)}
        pivot = pivot.rename(columns=rename_map)
    pivot = pivot.reset_index().rename(columns={"index": "stroka"})
    pivot = pivot.replace(r"^\s*$", pd.NA, regex=True)

    return TVRFile(pivot.sort_values("stroka"), separator=separator)


def iter_tvr2_triplets(path: str | Path) -> Iterator[Tuple[int, int, str]]:
    """Yield raw (stroka, stolbec, value) triplets from a TVR2 file."""

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline()
            if not header:
                return
            for raw_line in handle:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split(None, 2)
                if len(parts) < 3:
                    continue
                row_token, col_token, value = parts
                try:
                    row = int(row_token)
                    col = int(col_token)
                except ValueError as exc:
                    raise TVRIOError(f"Malformed triplet line: {raw_line!r}") from exc
                yield row, col, value
    except OSError as exc:
        raise TVRIOError(f"Cannot open TVR file: {path}") from exc


def write_tvr2(
    data: TVRFile | pd.DataFrame,
    path: str | Path,
    *,
    separator: str | None = None,
    triple_separator: str | None = None,
    skip_empty: bool = True,
) -> None:
    """Persist dataframe back into TVR2 format."""

    if isinstance(data, TVRFile):
        df = data.dataframe.copy()
        sep = separator or data.separator
        triple = triple_separator or data.triple_separator
    else:
        df = data.copy()
        sep = separator or DEFAULT_HEADER_SEPARATOR
        triple = triple_separator or DEFAULT_TRIPLE_SEPARATOR

    if "stroka" not in df.columns:
        df = df.reset_index().rename(columns={"index": "stroka"})

    try:
        df["stroka"] = pd.to_numeric(df["stroka"], errors="raise", downcast="integer")
    except (TypeError, ValueError) as exc:
        raise TVRIOError("Column 'stroka' must contain integers for TVR export.") from exc

    columns = [col for col in df.columns if col != "stroka"]
    header_line = sep + sep + sep.join(columns) + "\n"

    lines: list[str] = [header_line]
    working = df.sort_values("stroka")

    for _, row in working.iterrows():
        stroka = int(row["stroka"])
        for idx, col in enumerate(columns, start=1):
            value = row[col]
            value_str = _format_value_for_tvr(value)
            if skip_empty and value_str == "":
                continue
            lines.append(f"{stroka}{triple}{idx}{triple}{value_str}\n")

    try:
        Path(path).write_text("".join(lines), encoding="utf-8")
    except OSError as exc:
        raise TVRIOError(f"Cannot write TVR file: {path}") from exc


def _read_header(path: Path) -> Tuple[str, Sequence[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            header_line = handle.readline()
    except OSError as exc:
        raise TVRIOError(f"Cannot open TVR file: {path}") from exc

    if not header_line:
        raise TVRIOError(f"TVR file appears to be empty: {path}")

    separator = header_line[0] if header_line else DEFAULT_HEADER_SEPARATOR
    parts = header_line.rstrip("\r\n").split(separator)
    column_names = parts[2:] if len(parts) > 2 else []
    return separator, column_names


def _format_value_for_tvr(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return ""
        formatted = np.format_float_positional(float(value), trim="-")
        return "0" if formatted == "-0" else formatted
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if pd.isna(value):
        return ""
    return str(value).strip()
