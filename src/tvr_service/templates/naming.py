"""Normalization helpers for TVR column and cell aliases."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-z_]+")
_LEADING_PATTERN = re.compile(r"^[^A-Za-z_]+")


def sanitize_token(token: str) -> str:
    """Convert arbitrary column/row titles into snake-style aliases."""
    normalized = _SANITIZE_PATTERN.sub("_", token.strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("Alias cannot be empty after sanitization")
    normalized = _LEADING_PATTERN.sub("", normalized)
    if not normalized:
        normalized = "alias"
    return normalized


@dataclass(slots=True)
class ColumnAliasMapper:
    """Bidirectional mapping between TVR column names and safe aliases."""

    alias_to_name: Dict[str, str]
    name_to_alias: Dict[str, str]

    @classmethod
    def from_columns(cls, columns: Sequence[str]) -> "ColumnAliasMapper":
        alias_to_name: Dict[str, str] = {}
        name_to_alias: Dict[str, str] = {}

        for column in columns:
            base_alias = sanitize_token(column)
            alias = base_alias
            suffix = 1
            while alias in alias_to_name:
                alias = f"{base_alias}_{suffix}"
                suffix += 1
            alias_to_name[alias] = column
            name_to_alias[column] = alias

        return cls(alias_to_name=alias_to_name, name_to_alias=name_to_alias)

    def alias_for(self, column_name: str) -> str:
        try:
            return self.name_to_alias[column_name]
        except KeyError as exc:
            raise KeyError(f"Unknown column name '{column_name}'") from exc

    def column_for(self, alias: str) -> str:
        try:
            return self.alias_to_name[alias]
        except KeyError as exc:
            raise KeyError(f"Unknown column alias '{alias}'") from exc

    @property
    def aliases(self) -> Sequence[str]:
        return tuple(self.alias_to_name.keys())

    @property
    def columns(self) -> Sequence[str]:
        return tuple(self.name_to_alias.keys())


def build_cell_mapping(
    columns: Iterable[str],
    row_aliases: Iterable[str],
) -> Dict[str, tuple[str, str]]:
    """Precompute mapping from flattened cell names to (row, column) pairs."""
    row_alias_list = sorted(row_aliases, key=len, reverse=True)
    mapping: Dict[str, tuple[str, str]] = {}
    for column in columns:
        sanitized = sanitize_token(column)
        lowered = sanitized.lower()
        for row_alias in row_alias_list:
            prefix = f"{row_alias}_"
            if lowered.startswith(prefix.lower()):
                column_alias = sanitized[len(prefix) :]
                mapping[column] = (row_alias, column_alias)
                break
    return mapping
