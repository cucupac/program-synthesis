"""Candidate subchain extraction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from experiment.dsl import Grid, Program, execute, program_to_string
from experiment.generator import program_op_count

DEDUPE_BY = "program_string"


@dataclass(frozen=True)
class Candidate:
    program: Program
    program_string: str
    output: Grid
    op_count: int
    solution_ids: tuple[str, ...]

    @property
    def support_count(self) -> int:
        return len(self.solution_ids)


@dataclass(frozen=True)
class CandidateMenu:
    candidates: tuple[Candidate, ...]
    raw_candidate_count: int
    min_op_count: int
    min_support: int
    dedupe_by: str = DEDUPE_BY


def program_subtrees(program: Program) -> tuple[Program, ...]:
    return (program,) + tuple(
        subtree for arg in program.args for subtree in program_subtrees(arg)
    )


def extract_candidate_menu(
    solutions: Sequence[tuple[str, Program]],
    min_op_count: int = 2,
    min_support: int = 2,
) -> CandidateMenu:
    if min_op_count < 0:
        raise ValueError("min_op_count cannot be negative")
    if min_support < 1:
        raise ValueError("min_support must be at least 1")

    raw_candidate_count = 0
    programs_by_key: dict[str, Program] = {}
    outputs_by_key: dict[str, Grid] = {}
    op_counts_by_key: dict[str, int] = {}
    support_by_key: dict[str, set[str]] = {}

    for solution_id, solution in solutions:
        seen_in_solution = set()
        for subtree in program_subtrees(solution):
            op_count = program_op_count(subtree)
            if op_count < min_op_count:
                continue
            raw_candidate_count += 1
            key = program_to_string(subtree)
            if key not in programs_by_key:
                programs_by_key[key] = subtree
                outputs_by_key[key] = execute(subtree)
                op_counts_by_key[key] = op_count
            seen_in_solution.add(key)

        for key in seen_in_solution:
            support_by_key.setdefault(key, set()).add(solution_id)

    candidates = tuple(
        Candidate(
            program=programs_by_key[key],
            program_string=key,
            output=outputs_by_key[key],
            op_count=op_counts_by_key[key],
            solution_ids=tuple(sorted(support_by_key[key])),
        )
        for key in sorted(support_by_key)
        if len(support_by_key[key]) >= min_support
    )
    return CandidateMenu(
        candidates=candidates,
        raw_candidate_count=raw_candidate_count,
        min_op_count=min_op_count,
        min_support=min_support,
    )


def candidate_size_distribution(candidates: Sequence[Candidate]) -> dict[str, int]:
    counts = Counter(candidate.op_count for candidate in candidates)
    return {str(size): counts[size] for size in sorted(counts)}
