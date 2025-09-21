"""Utilities for building new TVR snippets from parsed robots."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd

from ..io.tvr_io import TVRFile, write_tvr2
from ..model import REFERENCE_COLUMNS, ParsedRobot, load_and_normalize, parse_robot
from ..model.normalized import TVRNormalized
from ..model.parser import ParsedNode
from converters import TVRParams, tvr2_to_excel


@dataclass(slots=True)
class GenerationResult:
    dataframe: pd.DataFrame
    tvr_file: Path
    excel_file: Path | None
    stroka_mapping: Dict[int, int]


def generate_robot_segment(
    source_tvr_path: str | Path,
    base_strokas: Sequence[int],
    base_assignment: Mapping[int, int],
    *,
    sec0_value: str,
    output_tvr_path: str | Path,
    output_excel_path: str | Path | None = None,
    duplicate_shared_filters: bool = False,
    stroka_overrides: Mapping[int, int] | None = None,
) -> GenerationResult:
    """Build TVR fragment from selected base rows.

    :param base_assignment: mapping of original base stroka -> new base stroka.
    :param stroka_overrides: optional mapping that pins any node (base or filter)
        to a specific stroka number in the generated fragment.
    """

    normalized = load_and_normalize(source_tvr_path)
    robot = parse_robot(
        normalized,
        base_strokas,
        duplicate_shared_filters=duplicate_shared_filters,
    )

    dataframe, mapping = build_dataframe_from_robot(
        normalized,
        robot,
        base_assignment=base_assignment,
        sec0_value=sec0_value,
        stroka_overrides=stroka_overrides,
    )

    write_tvr2(
        TVRFile(dataframe=dataframe, separator=normalized.separator, triple_separator=normalized.triple_separator),
        output_tvr_path,
    )

    excel_path: Path | None = None
    if output_excel_path is not None:
        params = TVRParams(header_sep=normalized.separator, triple_sep=normalized.triple_separator)
        tvr2_to_excel(str(output_tvr_path), str(output_excel_path), params=params)
        excel_path = Path(output_excel_path)

    return GenerationResult(
        dataframe=dataframe,
        tvr_file=Path(output_tvr_path),
        excel_file=excel_path,
        stroka_mapping=mapping,
    )


def build_dataframe_from_robot(
    normalized: TVRNormalized,
    robot: ParsedRobot,
    *,
    base_assignment: Mapping[int, int],
    sec0_value: str,
    stroka_overrides: Mapping[int, int] | None = None,
) -> tuple[pd.DataFrame, Dict[int, int]]:
    """Convert parsed robot into dataframe with new stroka numbering."""

    mapping = _build_stroka_mapping(robot, base_assignment, stroka_overrides)
    records = []

    for stroka, new_stroka in sorted(mapping.items(), key=lambda kv: kv[1]):
        node = robot.nodes[stroka]
        record: Dict[str, object] = {"stroka": new_stroka}

        for column in normalized.columns:
            if column == "stroka":
                continue
            if column == "Sec 0" and column in node.values:
                record[column] = sec0_value
                continue

            value = node.values.get(column)

            if column in REFERENCE_COLUMNS:
                record[column] = _render_reference_column(node, column, mapping, value)
                continue

            record[column] = value

        records.append(record)

    df = pd.DataFrame(records)
    ordered_columns = ["stroka", *normalized.columns]
    df = df.reindex(columns=ordered_columns, fill_value=pd.NA)
    df = df.sort_values("stroka").reset_index(drop=True)

    return df, mapping


def _build_stroka_mapping(
    robot: ParsedRobot,
    base_assignment: Mapping[int, int],
    overrides: Mapping[int, int] | None,
) -> Dict[int, int]:
    if not base_assignment:
        raise ValueError("base_assignment must contain at least one mapping for base stroka")

    overrides_copy = dict(overrides or {})
    mapping: Dict[int, int] = {}
    used_targets: Dict[int, int] = {}

    for stroka, node in robot.nodes.items():
        candidate_values: list[int] = []
        for base_stroka, new_base in base_assignment.items():
            relative = node.relative_for(base_stroka)
            if relative is None:
                continue
            offset = _offset_from_label(relative)
            candidate_values.append(new_base + offset)

        if not candidate_values:
            raise ValueError(
                f"No base stroka from assignment covers node {stroka}. "
                "Ensure base_assignment includes all root stroka numbers or provide overrides."
            )

        if stroka in overrides_copy:
            target = overrides_copy.pop(stroka)
        else:
            unique_values = set(candidate_values)
            if len(unique_values) != 1:
                raise ValueError(
                    "Inconsistent mapping for node "
                    f"{stroka}: computed values {sorted(unique_values)}. "
                    "Provide explicit override for this node to resolve the conflict."
                )
            target = unique_values.pop()

        _validate_target_value(target, stroka)
        collision = used_targets.get(target)
        if collision is not None and collision != stroka:
            raise ValueError(
                f"Target stroka {target} already assigned to node {collision}. "
                "Adjust assignments or overrides to avoid collisions."
            )

        mapping[stroka] = target
        used_targets[target] = stroka

    if overrides_copy:
        unknown = ", ".join(map(str, overrides_copy.keys()))
        raise KeyError(f"Overrides specified for unknown stroka numbers: {unknown}")

    return mapping


def _validate_target_value(value: int, source: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"Computed stroka for node {source} must be int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"Computed stroka for node {source} must be positive, got {value}")


def _offset_from_label(label: str) -> int:
    if label == "begin":
        return 0
    if label.startswith("begin+"):
        return int(label.split("+")[1])
    if label.startswith("begin-"):
        return -int(label.split("-")[1])
    if label.startswith("begin") and len(label) > 5:
        return int(label[5:])
    return 0


def _render_reference_column(
    node: ParsedNode,
    column: str,
    mapping: Dict[int, int],
    raw_value: object,
) -> object:
    refs = node.reference_values.get(column)
    if not refs:
        return raw_value

    rendered: list[str] = []
    for item in refs:
        if isinstance(item, int):
            if item not in mapping:
                raise KeyError(
                    f"Reference from node {node.stroka} via {column} points to unknown stroka {item}."
                )
            rendered.append(str(mapping[item]))
        else:
            rendered.append(str(item))

    return ", ".join(rendered)
