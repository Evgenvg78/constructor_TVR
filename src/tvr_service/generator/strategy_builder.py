"""High-level routines for building TVR tables from strategy templates."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from ..templates.naming import build_cell_mapping, sanitize_token
from ..templates.strategy_template import StrategyTemplate


class StrategyGenerator:
    """Generate TVR-ready dataframes from templates and config tables."""

    def __init__(
        self,
        template: StrategyTemplate,
        *,
        strategy_column: str = "strategy_id",
        start_column: str = "start",
        sec_column: str | None = "sec_0",
    ) -> None:
        self.template = template
        self.strategy_column = strategy_column
        self.start_column = start_column
        self.sec_column = sec_column

    def generate(
        self,
        table: pd.DataFrame,
        *,
        blank_rows_between: int = 1,
        sec_separator_rows: int = 1,
        include_strategy_column: bool = True,
    ) -> pd.DataFrame:
        """Render the entire configuration table into a TVR dataframe."""

        overrides_mapping = _prepare_override_mapping(
            table.columns,
            template=self.template,
            strategy_column=self.strategy_column,
            start_column=self.start_column,
            sec_column=self.sec_column,
        )

        previous_sec_value: Any | None = None
        output_records: List[Dict[str, Any]] = []
        base_columns = _base_output_columns(self.template.columns, include_strategy_column)

        for _, row in table.iterrows():
            start_value = row[self.start_column]
            if pd.isna(start_value):
                raise ValueError(
                    f"Start column '{self.start_column}' contains empty value for row {row.name}"
                )
            start_int = int(start_value)
            strategy_id = (
                row[self.strategy_column]
                if self.strategy_column in row
                else row.name
            )
            sec_value = None
            if self.sec_column and self.sec_column in row:
                sec_candidate = row[self.sec_column]
                sec_value = None if pd.isna(sec_candidate) else sec_candidate

            overrides = _collect_overrides(row, overrides_mapping)
            context = _build_context(row)
            context.setdefault("strategy_id", strategy_id)
            if sec_value is not None:
                context.setdefault("sec", sec_value)

            block_records = self.template.instantiate(
                start=start_int,
                overrides=overrides,
                context=context,
            )
            for record in block_records:
                # Исключаем добавление strategy_id в финальный результат
                # if include_strategy_column:
                #     record.setdefault("strategy_id", strategy_id)
                if (
                    sec_value is not None
                    and "Sec 0" in record
                    and pd.isna(record["Sec 0"])
                ):
                    record["Sec 0"] = sec_value

            if self.sec_column and sec_value != previous_sec_value and previous_sec_value is not None:
                output_records.extend(
                    _make_separator_rows(
                        base_columns,
                        count=sec_separator_rows,
                        label="sec_separator",
                        sec_value=sec_value,
                    )
                )

            output_records.extend(block_records)

            if blank_rows_between > 0:
                output_records.extend(
                    _make_separator_rows(
                        base_columns,
                        count=blank_rows_between,
                        label="separator",
                    )
                )

            previous_sec_value = sec_value

        frame = pd.DataFrame(output_records)
        ordered_columns = _order_columns(base_columns)
        frame = frame.reindex(columns=ordered_columns)
        frame = frame.reset_index(drop=True)
        return frame


def _prepare_override_mapping(
    columns: Iterable[str],
    *,
    template: StrategyTemplate,
    strategy_column: str,
    start_column: str,
    sec_column: str | None,
) -> Dict[str, tuple[str, str]]:
    skip_aliases = {
        sanitize_token(strategy_column).lower(),
        sanitize_token(start_column).lower(),
    }
    if sec_column is not None:
        skip_aliases.add(sanitize_token(sec_column).lower())
    mapping: Dict[str, tuple[str, str]] = {}

    precomputed = build_cell_mapping(columns, template.row_aliases)
    for column, coordinates in precomputed.items():
        alias = sanitize_token(column).lower()
        if alias in skip_aliases:
            continue
        mapping[column] = coordinates
    return mapping


def _collect_overrides(
    row: pd.Series, mapping: Mapping[str, tuple[str, str]]
) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {}
    for column, (row_alias, column_alias) in mapping.items():
        if column not in row:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        row_bucket = overrides.setdefault(row_alias, {})
        row_bucket[column_alias] = value
    return overrides


def _build_context(row: pd.Series) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for column, value in row.items():
        normalized = sanitize_token(column)
        if pd.isna(value):
            continue
        context.setdefault(normalized, value)
    return context


def _make_separator_rows(
    columns: Sequence[str],
    *,
    count: int,
    label: str,
    sec_value: Any | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        record = {column: pd.NA for column in columns}
        record["stroka"] = pd.NA
        record["row_alias"] = label
        if sec_value is not None and index == 0:
            if "Sec 0" in record:
                record["Sec 0"] = sec_value
        rows.append(record)
    return rows


def _base_output_columns(columns: Sequence[str], include_strategy: bool) -> List[str]:
    ordered: List[str] = []
    # Исключаем служебные столбцы strategy_id и row_alias из финального результата
    # Они используются только для внутренней обработки
    ordered.extend(["stroka"])  # stroka остается как обязательный столбец
    for column in columns:
        if column not in ordered:
            ordered.append(column)
    return ordered


def _order_columns(base_columns: Sequence[str]) -> List[str]:
    return list(base_columns)
