"""Starter-only frontier-promotion candidate menu."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from experiment.dsl import (
    GRID_SIZE,
    Grid,
    Program,
    UNARY_OPS,
    UNIVERSE,
    execute,
    primitive,
    program_to_string,
)
from experiment.generator import Task, program_op_count
from experiment.solver import LibraryItem, SolveConfig, build_frontier_index, primitive_library

FRONTIER_CONFIG = SolveConfig(node_budget=30_000, max_program_size=7, max_solutions=1)


@dataclass(frozen=True)
class FrontierCandidate:
    program: Program
    program_string: str
    output: Grid
    op_count: int
    support_task_ids: tuple[str, ...]
    first_hit_cost: int

    @property
    def support_count(self) -> int:
        return len(self.support_task_ids)


@dataclass(frozen=True)
class FrontierMenu:
    candidates: tuple[FrontierCandidate, ...]
    raw_candidate_count: int
    frontier_unique_outputs: int
    frontier_candidates_tried_total: int
    frontier_hit_budget: bool
    min_op_count: int
    max_op_count: int
    min_support: int
    cap: int


def frontier_promotion_menu(
    tasks_start: Sequence[Task],
    *,
    min_op_count: int = 1,
    max_op_count: int = 4,
    min_support: int = 2,
    cap: int = 50,
    frontier_config: SolveConfig = FRONTIER_CONFIG,
) -> FrontierMenu:
    if any(task.split != "start" for task in tasks_start):
        raise ValueError("frontier promotion only accepts starter tasks")
    if min_op_count < 0 or max_op_count < min_op_count:
        raise ValueError("invalid op-count range")
    if min_support < 1:
        raise ValueError("min_support must be at least 1")
    if cap < 1:
        raise ValueError("cap must be at least 1")

    frontier = build_frontier_index(primitive_library(), frontier_config)
    frontier_outputs = set(frontier.entries)
    primitive_outputs = {item.target for item in primitive_library()}
    blank = execute(primitive("blank"))

    support_by_output: dict[Grid, set[str]] = {}
    entry_by_output = {}
    raw_count = 0
    for output, entry in frontier.entries.items():
        op_count = program_op_count(entry.program)
        if not (min_op_count <= op_count <= max_op_count):
            continue
        if output == blank or output in primitive_outputs:
            continue
        raw_count += 1
        unary_outputs = {_unary_output(op, output) for op in UNARY_OPS}
        supporters = {
            task.id
            for task in tasks_start
            if _supports_target(output, task.target, frontier_outputs, unary_outputs)
        }
        if supporters:
            support_by_output[output] = supporters
            entry_by_output[output] = entry

    kept_outputs = [
        output
        for output, supporters in support_by_output.items()
        if len(supporters) >= min_support
    ]
    kept_outputs.sort(
        key=lambda output: (
            -len(support_by_output[output]),
            entry_by_output[output].candidates_tried_at_first_solution,
            program_to_string(entry_by_output[output].program),
        )
    )

    candidates = tuple(
        FrontierCandidate(
            program=entry_by_output[output].program,
            program_string=program_to_string(entry_by_output[output].program),
            output=output,
            op_count=program_op_count(entry_by_output[output].program),
            support_task_ids=tuple(sorted(support_by_output[output])),
            first_hit_cost=entry_by_output[output].candidates_tried_at_first_solution,
        )
        for output in kept_outputs[:cap]
    )

    return FrontierMenu(
        candidates=candidates,
        raw_candidate_count=raw_count,
        frontier_unique_outputs=frontier.unique_outputs,
        frontier_candidates_tried_total=frontier.candidates_tried_total,
        frontier_hit_budget=frontier.hit_budget,
        min_op_count=min_op_count,
        max_op_count=max_op_count,
        min_support=min_support,
        cap=cap,
    )


def candidate_library_items(
    candidates: Sequence[FrontierCandidate], prefix: str = "C"
) -> tuple[LibraryItem, ...]:
    return tuple(
        LibraryItem(f"{prefix}{index:04d}", candidate.output, candidate.program)
        for index, candidate in enumerate(candidates)
    )


def menu_diagnostics(menu: FrontierMenu) -> dict:
    return {
        "menu_size": len(menu.candidates),
        "raw_candidate_count": menu.raw_candidate_count,
        "op_count_distribution": _distribution(candidate.op_count for candidate in menu.candidates),
        "support_distribution": _distribution(
            candidate.support_count for candidate in menu.candidates
        ),
        "frontier_unique_outputs": menu.frontier_unique_outputs,
        "frontier_candidates_tried_total": menu.frontier_candidates_tried_total,
        "frontier_hit_budget": menu.frontier_hit_budget,
    }


def _supports_target(
    candidate_output: Grid,
    target: Grid,
    frontier_outputs: set[Grid],
    unary_outputs: set[Grid],
) -> bool:
    if candidate_output < target and (target - candidate_output) in frontier_outputs:
        return True
    if target < candidate_output and (candidate_output - target) in frontier_outputs:
        return True
    return target in unary_outputs


def _unary_output(op: str, grid: Grid) -> Grid:
    if op == "invert":
        return UNIVERSE - grid
    if op == "reflect_horizontal":
        return frozenset((GRID_SIZE - 1 - row, col) for row, col in grid)
    if op == "reflect_vertical":
        return frozenset((row, GRID_SIZE - 1 - col) for row, col in grid)
    if op == "reflect_diag":
        return frozenset((col, row) for row, col in grid)
    raise ValueError(f"unknown unary op: {op}")


def _distribution(values) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}
