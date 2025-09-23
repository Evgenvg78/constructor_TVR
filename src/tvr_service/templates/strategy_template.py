"""Strategy template abstractions for TVR generation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import pandas as pd

from .naming import ColumnAliasMapper, sanitize_token

_PLACEHOLDER_PATTERN = re.compile(r"^\{\{([A-Za-z0-9_]+)\}\}$")


@dataclass(slots=True)
class TemplateRow:
    """Definition of a single row inside a strategy template."""

    alias: str
    offset: int
    defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.alias = sanitize_token(self.alias)
        cleaned: Dict[str, Any] = {}
        for key, value in self.defaults.items():
            cleaned[sanitize_token(key)] = value
        object.__setattr__(self, "defaults", cleaned)


@dataclass(slots=True)
class StrategyTemplate:
    """Full template including row ordering and column aliases."""

    name: str
    rows: Sequence[TemplateRow]
    column_mapper: ColumnAliasMapper
    base_columns: Sequence[str] | None = None

    def __post_init__(self) -> None:
        aliases = [row.alias for row in self.rows]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Row aliases must be unique within a template")
        valid_column_aliases = set(self.column_mapper.aliases)
        for row in self.rows:
            unknown = set(row.defaults) - valid_column_aliases
            if unknown:
                unknown_list = ", ".join(sorted(unknown))
                raise ValueError(
                    f"Row '{row.alias}' references unknown columns: {unknown_list}"
                )

    @property
    def row_aliases(self) -> Sequence[str]:
        return [row.alias for row in self.rows]

    @property
    def columns(self) -> Sequence[str]:
        if self.base_columns is not None:
            return self.base_columns
        return tuple(self.column_mapper.columns)

    def instantiate(
        self,
        *,
        start: int,
        overrides: Mapping[str, Mapping[str, Any]] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Render template rows into concrete dataframe records."""

        overrides = overrides or {}
        base_context = dict(context or {})
        base_context.setdefault("start", start)

        records: List[Dict[str, Any]] = []
        columns = list(self.columns)
        alias_overrides: Dict[str, Mapping[str, Any]] = {
            sanitize_token(row_alias): {
                sanitize_token(column_alias): value
                for column_alias, value in row_override.items()
            }
            for row_alias, row_override in overrides.items()
        }

        for row in self.rows:
            row_context = dict(base_context)
            row_context.setdefault("row_alias", row.alias)
            record: Dict[str, Any] = {column: pd.NA for column in columns}
            record["stroka"] = start + row.offset
            record["row_alias"] = row.alias

            defaults = row.defaults
            merged: MutableMapping[str, Any] = dict(defaults)
            merged.update(alias_overrides.get(row.alias, {}))

            for column_alias, raw_value in merged.items():
                column_name = self.column_mapper.column_for(column_alias)
                value = _resolve_value(raw_value, row_context)
                record[column_name] = value

            records.append(record)

        return records


def _resolve_value(raw_value: Any, context: Mapping[str, Any]) -> Any:
    if callable(raw_value):
        return raw_value(context)
    if isinstance(raw_value, str):
        match = _PLACEHOLDER_PATTERN.match(raw_value)
        if match:
            key = match.group(1)
            return context.get(key)
    return raw_value
