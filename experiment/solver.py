"""Deterministic bottom-up search over DSL programs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from experiment.dsl import (
    GRID_SIZE,
    Grid,
    PRIMITIVES,
    Program,
    UNIVERSE,
    call,
    execute,
    primitive,
    program_to_string,
)

PRIMITIVE_ORDER = (
    "blank",
    "line_horizontal",
    "line_vertical",
    "diagonal",
    "square",
    "triangle",
)
UNARY_OP_ORDER = ("invert", "reflect_horizontal", "reflect_vertical", "reflect_diag")
BINARY_OP_ORDER = ("add", "subtract", "overlap")
COMMUTATIVE_OPS = {"add", "overlap"}


@dataclass(frozen=True)
class LibraryItem:
    name: str
    target: Grid
    program: Program | None = None

    def leaf(self) -> Program:
        return Program(self.name)


@dataclass(frozen=True)
class SolveConfig:
    node_budget: int = 30_000
    max_program_size: int = 7
    max_solutions: int = 3


@dataclass(frozen=True)
class SolveResult:
    solved: bool
    solutions: tuple[Program, ...]
    candidates_tried_total: int
    candidates_tried_at_first_solution: int | None
    hit_budget: bool
    unique_outputs: int


@dataclass(frozen=True)
class FrontierEntry:
    program: Program
    candidates_tried_at_first_solution: int
    abstract_search_size: int


@dataclass(frozen=True)
class FrontierIndex:
    entries: dict[Grid, FrontierEntry]
    candidates_tried_total: int
    hit_budget: bool
    candidates_tried_by_size: tuple[int, ...]

    @property
    def unique_outputs(self) -> int:
        return len(self.entries)

    def score(self, target: Grid) -> SolveResult:
        entry = self.entries.get(target)
        if entry is None:
            return SolveResult(
                solved=False,
                solutions=(),
                candidates_tried_total=self.candidates_tried_total,
                candidates_tried_at_first_solution=None,
                hit_budget=self.hit_budget,
                unique_outputs=self.unique_outputs,
            )
        return SolveResult(
            solved=True,
            solutions=(entry.program,),
            candidates_tried_total=self.candidates_tried_total,
            candidates_tried_at_first_solution=entry.candidates_tried_at_first_solution,
            hit_budget=self.hit_budget,
            unique_outputs=self.unique_outputs,
        )


def primitive_library() -> tuple[LibraryItem, ...]:
    return tuple(
        LibraryItem(name=name, target=execute(primitive(name))) for name in PRIMITIVE_ORDER
    )


def build_frontier_index(
    library: Sequence[LibraryItem] | None = None,
    config: SolveConfig = SolveConfig(max_solutions=1),
) -> FrontierIndex:
    _validate_config(config)
    library = tuple(library or primitive_library())

    programs_by_size: dict[int, list[Program]] = {}
    program_outputs: dict[Program, Grid] = {}
    entries: dict[Grid, FrontierEntry] = {}
    candidates_tried = 0
    candidates_tried_by_size = [0] * (config.max_program_size + 1)
    hit_budget = False

    def result() -> FrontierIndex:
        return FrontierIndex(
            entries,
            candidates_tried,
            hit_budget,
            tuple(candidates_tried_by_size),
        )

    def try_program(program: Program, size: int, output: Grid) -> bool:
        nonlocal candidates_tried, hit_budget
        if candidates_tried >= config.node_budget:
            hit_budget = True
            return False

        candidates_tried += 1
        candidates_tried_by_size[size] += 1
        program_outputs[program] = output
        if output not in entries:
            entries[output] = FrontierEntry(program, candidates_tried, size)
            programs_by_size.setdefault(size, []).append(program)
        return True

    for item in library:
        if not try_program(item.leaf(), 0, item.target):
            return result()

    for size in range(1, config.max_program_size + 1):
        for op in UNARY_OP_ORDER:
            for child in programs_by_size.get(size - 1, ()):
                program = call(op, child)
                if not try_program(program, size, _unary_output(op, program_outputs[child])):
                    return result()
        for op in BINARY_OP_ORDER:
            for left_size in range(size):
                right_size = size - 1 - left_size
                for left in programs_by_size.get(left_size, ()):
                    for right in programs_by_size.get(right_size, ()):
                        if _skip_commutative_duplicate(op, left, right):
                            continue
                        program = call(op, left, right)
                        output = _binary_output(
                            op, program_outputs[left], program_outputs[right]
                        )
                        if not try_program(program, size, output):
                            return result()

    return result()


def solve_task(
    target: Grid,
    library: Sequence[LibraryItem] | None = None,
    config: SolveConfig = SolveConfig(),
) -> SolveResult:
    _validate_config(config)
    library = tuple(library or primitive_library())

    programs_by_size: dict[int, list[Program]] = {}
    program_outputs: dict[Program, Grid] = {}
    seen_outputs: dict[Grid, Program] = {}
    solution_strings: set[str] = set()
    solutions: list[Program] = []
    candidates_tried = 0
    first_solution_at: int | None = None
    hit_budget = False

    def result() -> SolveResult:
        return SolveResult(
            solved=bool(solutions),
            solutions=tuple(solutions),
            candidates_tried_total=candidates_tried,
            candidates_tried_at_first_solution=first_solution_at,
            hit_budget=hit_budget,
            unique_outputs=len(seen_outputs),
        )

    def try_program(program: Program, size: int, output: Grid) -> bool:
        nonlocal candidates_tried, first_solution_at, hit_budget
        if candidates_tried >= config.node_budget:
            hit_budget = True
            return False

        candidates_tried += 1
        text = program_to_string(program)
        program_outputs[program] = output

        if output == target and text not in solution_strings:
            if first_solution_at is None:
                first_solution_at = candidates_tried
            solutions.append(program)
            solution_strings.add(text)
            if len(solutions) >= config.max_solutions:
                return False

        if output not in seen_outputs:
            seen_outputs[output] = program
            programs_by_size.setdefault(size, []).append(program)
        return True

    for item in library:
        if not try_program(item.leaf(), 0, item.target):
            return result()

    for size in range(1, config.max_program_size + 1):
        for op in UNARY_OP_ORDER:
            for child in programs_by_size.get(size - 1, ()):
                program = call(op, child)
                if not try_program(program, size, _unary_output(op, program_outputs[child])):
                    return result()
        for op in BINARY_OP_ORDER:
            for left_size in range(size):
                right_size = size - 1 - left_size
                for left in programs_by_size.get(left_size, ()):
                    for right in programs_by_size.get(right_size, ()):
                        if _skip_commutative_duplicate(op, left, right):
                            continue
                        program = call(op, left, right)
                        output = _binary_output(
                            op, program_outputs[left], program_outputs[right]
                        )
                        if not try_program(program, size, output):
                            return result()

    return result()


def solve_tasks(
    targets: Sequence[Grid],
    library: Sequence[LibraryItem] | None = None,
    config: SolveConfig = SolveConfig(),
) -> tuple[SolveResult, ...]:
    return tuple(solve_task(target, library=library, config=config) for target in targets)


def _validate_config(config: SolveConfig) -> None:
    if config.node_budget < 1:
        raise ValueError("node_budget must be at least 1")
    if config.max_program_size < 0:
        raise ValueError("max_program_size cannot be negative")
    if config.max_solutions < 1:
        raise ValueError("max_solutions must be at least 1")


def _try_unary_programs(size, programs_by_size, try_program) -> bool:
    children = programs_by_size.get(size - 1, ())
    for op in UNARY_OP_ORDER:
        for child in children:
            if not try_program(call(op, child), size):
                return False
    return True


def _try_binary_programs(size, programs_by_size, try_program) -> bool:
    for op in BINARY_OP_ORDER:
        for left_size in range(size):
            right_size = size - 1 - left_size
            for left in programs_by_size.get(left_size, ()):
                for right in programs_by_size.get(right_size, ()):
                    if _skip_commutative_duplicate(op, left, right):
                        continue
                    if not try_program(call(op, left, right), size):
                        return False
    return True


def _skip_commutative_duplicate(op: str, left: Program, right: Program) -> bool:
    return op in COMMUTATIVE_OPS and program_to_string(left) > program_to_string(right)


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


def _binary_output(op: str, left: Grid, right: Grid) -> Grid:
    if op == "add":
        return left | right
    if op == "subtract":
        return left - right
    if op == "overlap":
        return left & right
    raise ValueError(f"unknown binary op: {op}")
