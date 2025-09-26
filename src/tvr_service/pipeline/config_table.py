"""Helpers for building configuration tables before rendering TVR layouts."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd

from .portfolio import load_whitelist

__all__ = ["build_config_table", "enumerate_parameter_rows"]


class ConfigTableError(ValueError):
    """Raised when configuration table inputs are inconsistent."""


def enumerate_parameter_rows(options: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Return a list of dictionaries for every combination of parameters."""

    if not options:
        return [{}]
    names: list[str] = []
    value_groups: list[list[Any]] = []
    for name, raw_values in options.items():
        normalized = _normalize_values(raw_values)
        if not normalized:
            raise ConfigTableError(f"Parameter '{name}' has no values.")
        names.append(str(name))
        value_groups.append(normalized)
    combinations = product(*value_groups)
    return [dict(zip(names, combo)) for combo in combinations]


def build_config_table(
    *,
    parameter_options: Mapping[str, Sequence[Any]],
    futures: Sequence[str] | None = None,
    whitelist_path: str | Path | None = None,
    static_columns: Mapping[str, Any] | None = None,
    sec_column: str = "sec_0",
    strategy_column: str = "strategy_id",
    strategy_template: str | None = None,
) -> pd.DataFrame:
    """Construct a configuration DataFrame by multiplying futures and parameters."""

    universe = _resolve_universe(futures, whitelist_path)
    if not universe:
        raise ConfigTableError("Futures universe is empty.")
    parameter_rows = enumerate_parameter_rows(parameter_options)
    static = dict(static_columns or {})
    records: list[MutableMapping[str, Any]] = []
    for index, sec in enumerate(universe):
        for params in parameter_rows:
            row: MutableMapping[str, Any] = dict(params)
            row[sec_column] = sec
            if strategy_template is not None:
                template_context = dict(params)
                template_context.update({"sec": sec, "index": index})
                try:
                    row[strategy_column] = strategy_template.format(**template_context)
                except KeyError as error:
                    missing = error.args[0]
                    raise ConfigTableError(
                        f"Strategy template is missing placeholder '{missing}'."
                    ) from error
            elif strategy_column in static:
                row[strategy_column] = static[strategy_column]
            for column, value in static.items():
                if column == strategy_column and column in row:
                    continue
                row.setdefault(column, value)
            records.append(row)
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    ordered_columns = _preferred_order(frame.columns, strategy_column, sec_column)
    return frame.reindex(columns=ordered_columns)


def _normalize_values(values: Sequence[Any]) -> list[Any]:
    if isinstance(values, (str, bytes)):
        return [values]
    return [item for item in values]


def _resolve_universe(
    futures: Sequence[str] | None,
    whitelist_path: str | Path | None,
) -> list[str]:
    if futures is not None:
        return [
            entry for entry in (_normalize_future(item) for item in futures) if entry
        ]
    loaded = load_whitelist(whitelist_path)
    return [entry for entry in (_normalize_future(item) for item in loaded) if entry]


def _normalize_future(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _preferred_order(columns: Sequence[str], strategy_column: str, sec_column: str) -> list[str]:
    ordered: list[str] = []
    for name in (strategy_column, sec_column):
        if name in columns and name not in ordered:
            ordered.append(name)
    for name in columns:
        if name not in ordered:
            ordered.append(name)
    return ordered
