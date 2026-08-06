"""Primitives, programs, and execution for the Pattern Builder Task DSL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

GRID_SIZE = 10

Cell = tuple[int, int]
Grid = frozenset[Cell]


@dataclass(frozen=True, slots=True)
class Program:
    name: str
    args: tuple["Program", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _canonical_name(self.name))
        object.__setattr__(self, "args", tuple(self.args))


UNIVERSE: Grid = frozenset(
    (row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE)
)

PRIMITIVES = {
    "blank",
    "line_horizontal",
    "line_vertical",
    "diagonal",
    "square",
    "triangle",
}

BINARY_OPS = {"add", "subtract", "overlap"}
UNARY_OPS = {"invert", "reflect_horizontal", "reflect_vertical", "reflect_diag"}


def primitive(name: str) -> Program:
    return Program(name)


def call(name: str, *args: Program) -> Program:
    return Program(name, args)


def execute(program: Program, helpers: Mapping[str, Grid] | None = None) -> Grid:
    helpers = helpers or {}
    name = _canonical_name(program.name)

    if not program.args:
        if name in PRIMITIVES:
            return _primitive_grid(name)
        if name in helpers:
            return _validate_grid(helpers[name])
        raise ValueError(f"unknown primitive or helper: {program.name}")

    if name in BINARY_OPS:
        _require_arity(program, 2)
        left, right = (execute(arg, helpers) for arg in program.args)
        if name == "add":
            return left | right
        if name == "subtract":
            return left - right
        return left & right

    if name in UNARY_OPS:
        _require_arity(program, 1)
        grid = execute(program.args[0], helpers)
        if name == "invert":
            return UNIVERSE - grid
        if name == "reflect_horizontal":
            return frozenset((GRID_SIZE - 1 - row, col) for row, col in grid)
        if name == "reflect_vertical":
            return frozenset((row, GRID_SIZE - 1 - col) for row, col in grid)
        return frozenset((col, row) for row, col in grid)

    raise ValueError(f"unknown operator: {program.name}")


def render_grid(grid: Grid) -> str:
    grid = _validate_grid(grid)
    return "\n".join(
        "".join("#" if (row, col) in grid else "." for col in range(GRID_SIZE))
        for row in range(GRID_SIZE)
    )


def grid_from_rows(rows: Sequence[str]) -> Grid:
    if len(rows) != GRID_SIZE:
        raise ValueError(f"expected {GRID_SIZE} rows")
    cells = set()
    for row, line in enumerate(rows):
        if len(line) != GRID_SIZE:
            raise ValueError(f"row {row} has length {len(line)}")
        for col, char in enumerate(line):
            if char == "#":
                cells.add((row, col))
            elif char != ".":
                raise ValueError(f"unexpected grid character: {char!r}")
    return frozenset(cells)


def program_to_string(program: Program) -> str:
    name = _canonical_name(program.name)
    if not program.args:
        return name
    args = ",".join(program_to_string(arg) for arg in program.args)
    return f"{name}({args})"


def program_size(program: Program) -> int:
    return 1 + sum(program_size(arg) for arg in program.args)


def _canonical_name(name: str) -> str:
    return "overlap" if name == "intersect" else name


def _require_arity(program: Program, expected: int) -> None:
    if len(program.args) != expected:
        raise ValueError(
            f"{program.name} expects {expected} args, got {len(program.args)}"
        )


def _validate_grid(grid: Grid) -> Grid:
    bad_cells = [
        cell
        for cell in grid
        if len(cell) != 2
        or not all(isinstance(value, int) for value in cell)
        or not (0 <= cell[0] < GRID_SIZE and 0 <= cell[1] < GRID_SIZE)
    ]
    if bad_cells:
        raise ValueError(f"grid contains out-of-bounds cells: {bad_cells[:3]}")
    return frozenset(grid)


def _primitive_grid(name: str) -> Grid:
    center = GRID_SIZE // 2

    if name == "blank":
        return frozenset()
    if name == "line_horizontal":
        return frozenset((center, col) for col in range(GRID_SIZE))
    if name == "line_vertical":
        return frozenset((row, center) for row in range(GRID_SIZE))
    if name == "diagonal":
        return frozenset((i, i) for i in range(GRID_SIZE))
    if name == "square":
        return frozenset(
            (row, col)
            for row in range(GRID_SIZE)
            for col in range(GRID_SIZE)
            if row in {0, GRID_SIZE - 1} or col in {0, GRID_SIZE - 1}
        )
    if name == "triangle":
        return frozenset(
            (row, col) for row in range(GRID_SIZE) for col in range(row + 1)
        )
    raise ValueError(f"unknown primitive: {name}")
