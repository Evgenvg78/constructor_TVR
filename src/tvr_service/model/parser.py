"""Parser for TVR normalized structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Sequence

import numpy as np

from .normalized import REFERENCE_COLUMNS, TVRNormalized, load_and_normalize

MODE_LABELS = {
    1: "long",
    -1: "short",
}


def _resolve_mode_label(value: object) -> str | None:
    if isinstance(value, (int, np.integer)):
        return MODE_LABELS.get(int(value))
    if value in MODE_LABELS:
        return MODE_LABELS[value]  # type: ignore[index]
    return None


@dataclass(slots=True)
class ParsedEdge:
    """Reference link from a parent node to a child node."""

    key: str
    node: "ParsedNode"


@dataclass(slots=True)
class ParsedNode:
    """Parsed representation of a TVR row."""

    stroka: int
    values: Dict[str, object]
    mode_label: str
    relative_positions: Dict[int, str]
    reference_values: Dict[str, tuple[Any, ...]]
    edges: List[ParsedEdge] = field(default_factory=list)
    modes_present: set[str] = field(default_factory=set)
    shared: bool = False

    def add_edge(self, key: str, node: "ParsedNode") -> None:
        self.edges.append(ParsedEdge(key=key, node=node))

    def relative_for(self, base_stroka: int) -> str | None:
        return self.relative_positions.get(base_stroka)


@dataclass(slots=True)
class ParsedRoot:
    base_stroka: int
    label: str
    mode_label: str
    node: ParsedNode


@dataclass(slots=True)
class ParsedRobot:
    roots: List[ParsedRoot]
    nodes: Dict[int, ParsedNode]
    duplicate_shared_filters: bool

    def iter_nodes(self) -> Iterator[ParsedNode]:
        seen: set[int] = set()
        for root in self.roots:
            yield from _walk_nodes(root.node, seen)

    def outline(self) -> List[str]:
        lines: List[str] = []
        for root in self.roots:
            counters: Dict[int, int] = {}
            _render_outline(root.node, root, counters, [], lines)
        return lines


def parse_robot(
    normalized: TVRNormalized,
    base_strokas: Sequence[int],
    *,
    duplicate_shared_filters: bool = False,
    reference_columns: Iterable[str] = REFERENCE_COLUMNS,
) -> ParsedRobot:
    if not base_strokas:
        raise ValueError("base_strokas must contain at least one stroka number")

    base_infos: List[tuple[int, str, str]] = []
    for index, stroka in enumerate(base_strokas):
        row = normalized.get(stroka)
        if row is None:
            raise KeyError(f"Row {stroka} is missing in normalized data")
        mode_label = _resolve_mode_label(row.values.get("Mode")) or "unknown"
        label = f"base_{mode_label if mode_label != 'unknown' else index + 1}"
        base_infos.append((stroka, label, mode_label))

    global_nodes: Dict[int, ParsedNode] = {}
    roots: List[ParsedRoot] = []
    ref_columns = tuple(reference_columns)

    for base_stroka, label, mode_label in base_infos:
        visited_local: set[int] = set()
        node = _build_subtree(
            normalized,
            base_stroka,
            base_stroka,
            mode_label,
            ref_columns,
            global_nodes,
            visited_local,
            duplicate_shared_filters,
        )
        roots.append(ParsedRoot(base_stroka=base_stroka, label=label, mode_label=mode_label, node=node))

    for node in global_nodes.values():
        node.shared = len(node.relative_positions) > 1

    return ParsedRobot(roots=roots, nodes=global_nodes, duplicate_shared_filters=duplicate_shared_filters)


def parse_robot_from_file(
    path: str | Path,
    base_strokas: Sequence[int],
    *,
    duplicate_shared_filters: bool = False,
    reference_columns: Iterable[str] = REFERENCE_COLUMNS,
) -> ParsedRobot:
    normalized = load_and_normalize(path)
    return parse_robot(
        normalized,
        base_strokas,
        duplicate_shared_filters=duplicate_shared_filters,
        reference_columns=reference_columns,
    )


def _build_subtree(
    normalized: TVRNormalized,
    stroka: int,
    base_stroka: int,
    base_mode_label: str,
    reference_columns: Sequence[str],
    global_nodes: MutableMapping[int, ParsedNode],
    visited_local: set[int],
    duplicate_shared_filters: bool,
) -> ParsedNode:
    row = normalized.get(stroka)
    if row is None:
        raise KeyError(f"Row {stroka} is missing in normalized data")

    if not duplicate_shared_filters and stroka in global_nodes:
        node = global_nodes[stroka]
        _propagate_relative(node, base_stroka, base_mode_label, set())
        return node

    if stroka in visited_local:
        raise ValueError(f"Cycle detected while parsing TVR structure at stroka {stroka}")
    visited_local.add(stroka)

    values = dict(row.values)
    mode_label = _resolve_mode_label(values.get("Mode")) or "unknown"

    node = ParsedNode(
        stroka=stroka,
        values=values,
        mode_label=mode_label,
        relative_positions={base_stroka: _relative_label(base_stroka, stroka)},
        reference_values={key: tuple(val) for key, val in row.references.items()},
        modes_present={base_mode_label},
    )

    if not duplicate_shared_filters:
        global_nodes[stroka] = node

    for key in reference_columns:
        refs = row.references.get(key, tuple())
        for ref in refs:
            if isinstance(ref, int):
                child = _build_subtree(
                    normalized,
                    ref,
                    base_stroka,
                    base_mode_label,
                    reference_columns,
                    global_nodes,
                    visited_local,
                    duplicate_shared_filters,
                )
                node.add_edge(key, child)

    visited_local.remove(stroka)
    return node


def _propagate_relative(
    node: ParsedNode,
    base_stroka: int,
    base_mode_label: str,
    seen: set[int],
) -> None:
    if node.stroka in seen:
        return
    seen.add(node.stroka)
    node.relative_positions[base_stroka] = _relative_label(base_stroka, node.stroka)
    node.modes_present.add(base_mode_label)
    for edge in node.edges:
        _propagate_relative(edge.node, base_stroka, base_mode_label, seen)


def _relative_label(base: int, current: int) -> str:
    diff = current - base
    if diff == 0:
        return "begin"
    if diff > 0:
        return f"begin+{diff}"
    return f"begin{diff}"


def _walk_nodes(node: ParsedNode, seen: set[int]) -> Iterator[ParsedNode]:
    if node.stroka in seen:
        return
    seen.add(node.stroka)
    yield node
    for edge in node.edges:
        yield from _walk_nodes(edge.node, seen)


def _render_outline(
    node: ParsedNode,
    root: ParsedRoot,
    counters: Dict[int, int],
    path: List[str],
    lines: List[str],
) -> None:
    depth = len(path)
    counters.setdefault(depth, 0)
    counters[depth] += 1
    mode_suffix = "_&_".join(sorted(node.modes_present)) if node.modes_present else node.mode_label
    label = root.label if depth == 0 else f"filter_{depth}_{mode_suffix}"
    rel = node.relative_for(root.base_stroka) or ""
    lines.append(f"{'#' * (depth + 1)}{label} [{rel}]")

    next_path = path + [label]
    for edge in node.edges:
        _render_outline(edge.node, root, counters, next_path, lines)
