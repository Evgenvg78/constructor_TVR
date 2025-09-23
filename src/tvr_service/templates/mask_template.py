"""Utilities for building strategy templates from mask tables."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from .naming import ColumnAliasMapper, sanitize_token
from .strategy_template import StrategyTemplate, TemplateRow

PathLike = str | Path

_DEFAULT_METADATA_COLUMNS = {"row_alias", "stroka", "offset"}
_DEFAULT_PLACEHOLDERS: Mapping[str, Any] = {
    "Start": "{{start}}",
    "Sec 0": "{{sec}}",
}


@dataclass(slots=True)
class MaskTemplateResult:
    """Output of mask-driven template builder."""

    template: StrategyTemplate
    mapper: ColumnAliasMapper
    override_columns: List[str]


def build_template_from_mask_df(
    mask_df: pd.DataFrame,
    *,
    name: str,
    marker: Any = 1,
    metadata_columns: Iterable[str] | None = None,
    placeholders: Mapping[str, Any] | None = None,
    row_alias_column: str = "row_alias",
    offset_column: str = "offset",
    stroka_column: str = "stroka",
    value_columns: Sequence[str] | None = None,
) -> MaskTemplateResult:
    """Convert a mask table with markers into a :class:`StrategyTemplate`."""

    if mask_df.empty:
        raise ValueError("Mask dataframe must not be empty")

    placeholders = dict(_DEFAULT_PLACEHOLDERS | dict(placeholders or {}))

    metadata = set(_DEFAULT_METADATA_COLUMNS)
    if metadata_columns is not None:
        metadata.update(metadata_columns)

    if value_columns is None:
        candidate_columns = [
            column for column in mask_df.columns if column not in metadata
        ]
        if not candidate_columns:
            raise ValueError("No value columns detected in mask dataframe")
        value_columns = candidate_columns
    else:
        missing = [column for column in value_columns if column not in mask_df.columns]
        if missing:
            missing_list = ", ".join(missing)
            raise KeyError(f"Value columns not found in mask dataframe: {missing_list}")

    mapper = ColumnAliasMapper.from_columns(value_columns)
    template_rows: List[TemplateRow] = []
    override_columns: List[str] = []

    offsets = _resolve_offsets(
        mask_df,
        offset_column=offset_column,
        stroka_column=stroka_column,
    )

    for index, row in mask_df.iterrows():
        raw_alias = row.get(row_alias_column)
        if pd.isna(raw_alias) or str(raw_alias).strip() == "":
            raise ValueError(
                f"Row {index} is missing value in column '{row_alias_column}'"
            )
        row_alias = sanitize_token(str(raw_alias))
        offset = offsets[index]
        defaults: Dict[str, Any] = {}

        for column in value_columns:
            cell_value = row[column]
            if pd.isna(cell_value):
                continue

            column_alias = mapper.alias_for(column)
            if _is_marked(cell_value, marker):
                override_name = f"{row_alias}_{column_alias}"
                override_columns.append(override_name)
                default_value = placeholders.get(column, pd.NA)
                defaults[column_alias] = default_value
            else:
                defaults[column_alias] = cell_value

        template_rows.append(
            TemplateRow(
                alias=row_alias,
                offset=offset,
                defaults=defaults,
            )
        )

    template = StrategyTemplate(
        name=name,
        rows=template_rows,
        column_mapper=mapper,
        base_columns=value_columns,
    )

    unique_overrides = sorted(dict.fromkeys(override_columns))

    return MaskTemplateResult(
        template=template,
        mapper=mapper,
        override_columns=unique_overrides,
    )


def build_template_from_mask_file(
    path: PathLike,
    *,
    name: str | None = None,
    marker: Any = 1,
    metadata_columns: Iterable[str] | None = None,
    placeholders: Mapping[str, Any] | None = None,
    row_alias_column: str = "row_alias",
    offset_column: str = "offset",
    stroka_column: str = "stroka",
    value_columns: Sequence[str] | None = None,
    sheet_name: int | str = 0,
) -> MaskTemplateResult:
    """Load mask table from Excel/CSV file and build a template."""

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Mask file does not exist: {path_obj}")

    if path_obj.suffix.lower() == ".csv":
        mask_df = pd.read_csv(path_obj)
    else:
        mask_df = pd.read_excel(path_obj, sheet_name=sheet_name)

    template_name = name or path_obj.stem

    return build_template_from_mask_df(
        mask_df,
        name=template_name,
        marker=marker,
        metadata_columns=metadata_columns,
        placeholders=placeholders,
        row_alias_column=row_alias_column,
        offset_column=offset_column,
        stroka_column=stroka_column,
        value_columns=value_columns,
    )


def _resolve_offsets(
    mask_df: pd.DataFrame,
    *,
    offset_column: str,
    stroka_column: str,
) -> Dict[int, int]:
    if offset_column in mask_df.columns:
        offsets_series = mask_df[offset_column]
        if offsets_series.isna().any():
            raise ValueError("Offset column contains empty values")
        return offsets_series.astype(int).to_dict()

    if stroka_column in mask_df.columns:
        stroka_series = mask_df[stroka_column]
        try:
            numeric = stroka_series.astype(int)
        except (ValueError, TypeError):
            base_value = 0
        else:
            if numeric.isna().any():
                raise ValueError("Stroka column contains empty values")
            base_value = int(numeric.min())
            return (numeric - base_value).to_dict()

    # fallback: use row index as offset order
    return {index: pos for pos, index in enumerate(mask_df.index)}


def _is_marked(value: Any, marker: Any) -> bool:
    if pd.isna(value):
        return False
    if marker is None:
        return bool(value)
    if isinstance(marker, (set, list, tuple, frozenset)):
        return value in marker
    if isinstance(value, str) and isinstance(marker, str):
        return value.strip().lower() == marker.strip().lower()
    try:
        return float(value) == float(marker)
    except (TypeError, ValueError):
        return value == marker
