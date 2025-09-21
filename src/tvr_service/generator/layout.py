"""Helpers for inspecting and reshaping parsed TVR structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from ..model import ParsedRobot, ParsedRoot, ParsedNode, load_and_normalize, parse_robot


@dataclass(slots=True)
class StructureEntry:
    """Single node of a parsed robot, represented relative to the primary base."""

    order: int
    label: str
    display: str
    original_stroka: int
    relative_offset: int
    depth: int
    is_root: bool


@dataclass(slots=True)
class StructureLayout:
    """Collection of structure entries bound to a specific primary base."""

    primary_base_stroka: int
    root_strokas: Sequence[int]
    entries: List[StructureEntry]

    def to_lines(self) -> List[str]:
        """Return editable lines in the form '<label>: <offset>'."""
        return [f"{entry.display}: {entry.relative_offset}" for entry in self.entries]

    def to_text(self, *, line_separator: str = "\n") -> str:
        """Serialise the layout to a single string."""
        return line_separator.join(self.to_lines())


@dataclass(slots=True)
class LayoutEdits:
    """Relative overrides produced after parsing edited layout text."""

    relative_overrides: Dict[int, int]  # original_stroka -> new relative offset


@dataclass(slots=True)
class CompiledLayout:
    """Result of applying relative overrides for a concrete generation run."""

    base_assignment: Dict[int, int]
    stroka_overrides: Dict[int, int]


def build_layout_from_source(
    source_tvr_path: str,
    base_strokas: Sequence[int],
    *,
    duplicate_shared_filters: bool = False,
    primary_base: int | None = None,
) -> StructureLayout:
    """Load a TVR file, parse selected bases and build a structure layout."""

    normalized = load_and_normalize(source_tvr_path)
    robot = parse_robot(
        normalized,
        base_strokas,
        duplicate_shared_filters=duplicate_shared_filters,
    )
    return build_layout(robot, primary_base=primary_base)


def build_layout(robot: ParsedRobot, *, primary_base: int | None = None) -> StructureLayout:
    """Generate structure entries for a parsed robot."""

    if not robot.roots:
        raise ValueError("Parsed robot contains no roots")

    primary_root = robot.roots[0]
    base_stroka = primary_base if primary_base is not None else primary_root.base_stroka

    entries: List[StructureEntry] = []
    visited: set[int] = set()
    order = 0

    for root in robot.roots:
        order = _collect_entries(
            node=root.node,
            root=root,
            base_stroka=base_stroka,
            depth=0,
            entries=entries,
            visited=visited,
            order_start=order,
        )

    return StructureLayout(
        primary_base_stroka=base_stroka,
        root_strokas=[root.base_stroka for root in robot.roots],
        entries=entries,
    )


def parse_layout_text(
    layout: StructureLayout,
    text: str,
    *,
    line_separator: str = "\n",
) -> LayoutEdits:
    """Parse user-edited layout text back into relative overrides."""

    raw_lines = text.split(line_separator)
    lines = [line.rstrip() for line in raw_lines if line.strip() != ""]
    if len(lines) != len(layout.entries):
        raise ValueError(
            "Edited layout does not match template: "
            f"expected {len(layout.entries)} lines, got {len(lines)}"
        )

    overrides: Dict[int, int] = {}

    for entry, line in zip(layout.entries, lines):
        prefix = f"{entry.display}:"
        if not line.startswith(prefix):
            raise ValueError(
                f"Line '{line}' does not match expected prefix '{prefix}'"
            )
        tail = line[len(prefix):].strip()
        if not tail:
            raise ValueError(f"Line '{line}' must specify current offset")

        parts = [part.strip() for part in tail.split(":")]
        current_token = parts[0]
        try:
            current_value = int(current_token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid current offset '{current_token}' in line '{line}'"
            ) from exc

        if current_value != entry.relative_offset:
            raise ValueError(
                f"Current offset changed for line '{line}'. "
                "Only specify a new offset after a second ':'."
            )

        if len(parts) >= 2 and parts[1] != "":
            new_token = parts[1]
            try:
                new_value = int(new_token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid new offset '{new_token}' in line '{line}'"
                ) from exc
            if new_value < 0:
                raise ValueError(f"Offset must be non-negative: {new_value}")
            overrides[entry.original_stroka] = new_value

    return LayoutEdits(relative_overrides=overrides)


def compile_layout(
    layout: StructureLayout,
    edits: LayoutEdits,
    base_assignment: Mapping[int, int],
) -> CompiledLayout:
    """Convert relative overrides into absolute stroka mapping."""

    if layout.primary_base_stroka not in base_assignment:
        raise KeyError(
            "base_assignment must contain a mapping for the primary base stroka "
            f"{layout.primary_base_stroka}"
        )

    base_value = base_assignment[layout.primary_base_stroka]
    compiled_assignment = dict(base_assignment)
    stroka_overrides: Dict[int, int] = {}
    used_values: Dict[int, int] = {}

    relative_overrides = edits.relative_overrides

    for entry in layout.entries:
        relative_offset = relative_overrides.get(entry.original_stroka, entry.relative_offset)
        absolute = base_value + relative_offset
        if absolute <= 0:
            raise ValueError(
                f"Computed stroka {absolute} for node {entry.original_stroka} must be positive"
            )
        owner = used_values.get(absolute)
        if owner is not None and owner != entry.original_stroka:
            raise ValueError(
                f"Collision: stroka {absolute} would be used by nodes "
                f"{owner} and {entry.original_stroka}"
            )
        used_values[absolute] = entry.original_stroka

        if entry.is_root:
            compiled_assignment[entry.original_stroka] = absolute
        elif entry.original_stroka in relative_overrides:
            stroka_overrides[entry.original_stroka] = absolute

    return CompiledLayout(
        base_assignment=compiled_assignment,
        stroka_overrides=stroka_overrides,
    )


def _collect_entries(
    node: ParsedNode,
    *,
    root: ParsedRoot,
    base_stroka: int,
    depth: int,
    entries: List[StructureEntry],
    visited: set[int],
    order_start: int,
) -> int:
    if node.stroka in visited:
        return order_start

    visited.add(node.stroka)
    order = order_start + 1
    mode_suffix = "_&_".join(sorted(node.modes_present)) if node.modes_present else node.mode_label

    if depth == 0:
        base_label = root.label
        display = f"#{base_label}"
    else:
        base_label = f"filter_{depth}_{mode_suffix}"
        display = f"{'#' * (depth + 1)}{base_label}"

    entry = StructureEntry(
        order=order,
        label=base_label,
        display=display,
        original_stroka=node.stroka,
        relative_offset=node.stroka - base_stroka,
        depth=depth,
        is_root=(depth == 0 and node.stroka == root.base_stroka),
    )
    entries.append(entry)

    for edge in sorted(node.edges, key=lambda e: e.node.stroka):
        order = _collect_entries(
            node=edge.node,
            root=root,
            base_stroka=base_stroka,
            depth=depth + 1,
            entries=entries,
            visited=visited,
            order_start=order,
        )

    return order
