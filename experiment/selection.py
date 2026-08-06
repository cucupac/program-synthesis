"""Selection rules for frontier-promotion candidates."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import os
import random

from experiment.dsl import Grid, Program, execute
from experiment.frontier_promotion import FrontierCandidate, candidate_library_items
from experiment.solver import (
    LibraryItem,
    SolveConfig,
    build_frontier_index,
    primitive_library,
)


@dataclass(frozen=True)
class SelectionResult:
    candidates: tuple[FrontierCandidate, ...]
    cost: dict
    prefix_costs: tuple[dict, ...] = ()
    round_diagnostics: tuple[dict, ...] = ()


def _require_exact_k(candidates: Sequence[FrontierCandidate], k: int) -> None:
    if k < 0:
        raise ValueError("K cannot be negative")
    if len(candidates) < k:
        raise ValueError(f"insufficient candidates: requested K={k}, available={len(candidates)}")


def select_random_k(
    candidates: Sequence[FrontierCandidate], k: int, draw_seed: str
) -> tuple[FrontierCandidate, ...]:
    _require_exact_k(candidates, k)
    rng = random.Random(draw_seed)
    pool = list(candidates)
    rng.shuffle(pool)
    return tuple(pool[:k])


def select_most_frequent_k(
    candidates: Sequence[FrontierCandidate], k: int
) -> tuple[FrontierCandidate, ...]:
    _require_exact_k(candidates, k)
    return tuple(
        sorted(candidates, key=lambda c: (-c.support_count, c.program_string))[:k]
    )


def select_compression_k(
    candidates: Sequence[FrontierCandidate],
    starter_solutions: Sequence[Program],
    k: int,
) -> tuple[FrontierCandidate, ...]:
    return select_compression_k_with_cost(candidates, starter_solutions, k).candidates


def select_compression_k_with_cost(
    candidates: Sequence[FrontierCandidate],
    starter_solutions: Sequence[Program],
    k: int,
    *,
    trace: bool = False,
) -> SelectionResult:
    _require_exact_k(candidates, k)
    selected: list[FrontierCandidate] = []
    remaining = list(candidates)
    trial_libraries = 0
    prefix_costs = []
    diagnostics = []
    for round_number in range(1, k + 1):
        objective_before = (
            _total_segmentation_cost(starter_solutions, selected) if trace else None
        )
        scored = []
        for candidate in remaining:
            trial_libraries += 1
            scored.append(
                (
                    _total_segmentation_cost(starter_solutions, selected + [candidate]),
                    candidate.program_string,
                    candidate,
                )
            )
        best_row = min(scored, key=lambda row: (row[0], row[1]))
        best_score, _, best = best_row
        selected.append(best)
        remaining.remove(best)
        if trace:
            prefix_costs.append(
                _compression_selection_cost(
                    trial_libraries, len(starter_solutions)
                )
            )
            marginal = objective_before - best_score
            diagnostics.append(
                {
                    "round": round_number,
                    "selected_program": best.program_string,
                    "objective_before": objective_before,
                    "objective_after": best_score,
                    "marginal_objective_change": marginal,
                    "direction": _scalar_direction(marginal),
                    "best_tie_count": sum(row[0] == best_score for row in scored),
                }
            )
    cost = _compression_selection_cost(trial_libraries, len(starter_solutions))
    return SelectionResult(tuple(selected), cost, tuple(prefix_costs), tuple(diagnostics))


def _compression_selection_cost(trial_libraries: int, solution_count: int) -> dict:
    return {
            "selection_cost_candidate_programs_tried": 0,
            "input_solution_search_candidate_programs_tried": 0,
            "trial_libraries_evaluated": trial_libraries,
            "segmentation_evaluations": trial_libraries,
            "solution_segmentations_evaluated": trial_libraries * solution_count,
            "frontier_candidates_tried_total": 0,
        }


def select_utility_k(
    candidates: Sequence[FrontierCandidate],
    val_targets: Sequence[Grid],
    k: int,
    config: SolveConfig = SolveConfig(max_solutions=1),
    workers: int | None = None,
) -> tuple[FrontierCandidate, ...]:
    return greedy_by_frontier_score_with_cost(
        candidates, val_targets, k, config, workers=workers
    ).candidates


def greedy_by_frontier_score(
    candidates: Sequence[FrontierCandidate],
    targets: Sequence[Grid],
    k: int,
    config: SolveConfig = SolveConfig(max_solutions=1),
    workers: int | None = None,
) -> tuple[FrontierCandidate, ...]:
    return greedy_by_frontier_score_with_cost(
        candidates, targets, k, config, workers=workers
    ).candidates


def greedy_by_frontier_score_with_cost(
    candidates: Sequence[FrontierCandidate],
    targets: Sequence[Grid],
    k: int,
    config: SolveConfig = SolveConfig(max_solutions=1),
    workers: int | None = None,
    *,
    trace: bool = False,
) -> SelectionResult:
    return _greedy_by_trial_score_with_cost(
        candidates,
        targets,
        k,
        config,
        workers,
        lambda row: (-row["summary"]["mean_search_cost"], row["summary"]["solved_count"], _tie_break(row["candidate"].program_string)),
        "utility",
        trace,
    )


def greedy_by_solved_count_with_cost(
    candidates: Sequence[FrontierCandidate],
    targets: Sequence[Grid],
    k: int,
    config: SolveConfig = SolveConfig(max_solutions=1),
    workers: int | None = None,
    *,
    trace: bool = False,
) -> SelectionResult:
    return _greedy_by_trial_score_with_cost(
        candidates,
        targets,
        k,
        config,
        workers,
        lambda row: (row["summary"]["solved_count"], -row["summary"]["mean_search_cost"], _tie_break(row["candidate"].program_string)),
        "solved",
        trace,
    )


def _greedy_by_trial_score_with_cost(
    candidates: Sequence[FrontierCandidate],
    targets: Sequence[Grid],
    k: int,
    config: SolveConfig,
    workers: int | None,
    score_key,
    objective_kind: str,
    trace: bool,
) -> SelectionResult:
    _require_exact_k(candidates, k)
    selected: list[FrontierCandidate] = []
    remaining = list(candidates)
    trial_libraries = 0
    frontier_cost = 0
    prefix_costs = []
    diagnostics = []
    objective_before = (
        _solve_summary_for_candidates((), targets, config) if trace else None
    )
    worker_count = _worker_count(workers, len(remaining))
    executor = (
        nullcontext(None)
        if worker_count == 1
        else ProcessPoolExecutor(max_workers=worker_count)
    )
    with executor as pool:
        for round_number in range(1, k + 1):
            jobs = [
                (tuple(selected), candidate, tuple(targets), config)
                for candidate in remaining
            ]
            mapper = map if pool is None else pool.map
            scored = list(mapper(_score_trial, jobs))
            trial_libraries += len(scored)
            frontier_cost += sum(row["summary"]["frontier_candidates_tried_total"] for row in scored)
            best_row = max(scored, key=score_key)
            best = best_row["candidate"]
            selected.append(best)
            remaining.remove(best)
            if trace:
                prefix_costs.append(
                    _frontier_selection_cost(trial_libraries, frontier_cost)
                )
                diagnostics.append(
                    _trial_round_diagnostic(
                        round_number,
                        best_row,
                        scored,
                        objective_before,
                        objective_kind,
                    )
                )
                objective_before = best_row["summary"]
    return SelectionResult(
        tuple(selected),
        _frontier_selection_cost(trial_libraries, frontier_cost),
        tuple(prefix_costs),
        tuple(diagnostics),
    )


def _trial_round_diagnostic(
    round_number: int,
    best_row: dict,
    scored: Sequence[dict],
    before: dict,
    objective_kind: str,
) -> dict:
    after = best_row["summary"]
    before_objective = _trial_objective(before)
    after_objective = _trial_objective(after)
    before_rank = _trial_rank(before_objective, objective_kind)
    after_rank = _trial_rank(after_objective, objective_kind)
    best_rank = _trial_rank(after_objective, objective_kind)
    return {
        "round": round_number,
        "selected_program": best_row["candidate"].program_string,
        "objective_before": before_objective,
        "objective_after": after_objective,
        "marginal_objective_change": {
            "mean_search_cost_reduction": (
                before_objective["mean_search_cost"]
                - after_objective["mean_search_cost"]
            ),
            "solved_count_change": (
                after_objective["solved_count"] - before_objective["solved_count"]
            ),
        },
        "direction": (
            "positive"
            if after_rank > before_rank
            else "zero" if after_rank == before_rank else "negative"
        ),
        "best_tie_count": sum(
            _trial_rank(_trial_objective(row["summary"]), objective_kind) == best_rank
            for row in scored
        ),
    }


def _trial_objective(summary: dict) -> dict:
    return {
        "mean_search_cost": summary["mean_search_cost"],
        "solved_count": summary["solved_count"],
    }


def _trial_rank(objective: dict, objective_kind: str) -> tuple:
    if objective_kind == "utility":
        return (-objective["mean_search_cost"], objective["solved_count"])
    if objective_kind == "solved":
        return (objective["solved_count"], -objective["mean_search_cost"])
    raise ValueError(f"unknown objective kind: {objective_kind}")


def _scalar_direction(value: int | float) -> str:
    return "positive" if value > 0 else "zero" if value == 0 else "negative"


def _frontier_selection_cost(trial_libraries: int, frontier_cost: int) -> dict:
    return {
        "selection_cost_candidate_programs_tried": frontier_cost,
        "input_solution_search_candidate_programs_tried": 0,
        "trial_libraries_evaluated": trial_libraries,
        "segmentation_evaluations": 0,
        "solution_segmentations_evaluated": 0,
        "frontier_candidates_tried_total": frontier_cost,
    }


def _score_trial(args):
    selected, candidate, targets, config = args
    summary = _solve_summary_for_candidates(tuple(selected) + (candidate,), targets, config)
    return {"candidate": candidate, "summary": summary}


def _worker_count(workers: int | None, item_count: int) -> int:
    if workers is not None:
        return max(1, min(workers, item_count or 1))
    return max(1, min(os.cpu_count() or 1, item_count or 1, 8))


def solve_library_summary(
    targets: Sequence[Grid],
    library: Sequence[LibraryItem],
    config: SolveConfig = SolveConfig(max_solutions=1),
) -> dict:
    return solve_library_summaries((targets,), library, config)[0]


def solve_library_summaries(
    target_groups: Sequence[Sequence[Grid]],
    library: Sequence[LibraryItem],
    config: SolveConfig = SolveConfig(max_solutions=1),
) -> tuple[dict, ...]:
    index = build_frontier_index(library, config)
    return tuple(_summary_from_index(index, targets) for targets in target_groups)


def _summary_from_index(index, targets: Sequence[Grid]) -> dict:
    results = [index.score(target) for target in targets]
    solved = [result for result in results if result.solved]
    costs = [
        result.candidates_tried_at_first_solution
        if result.solved
        else result.candidates_tried_total
        for result in results
    ]
    first_costs = [
        result.candidates_tried_at_first_solution
        for result in solved
        if result.candidates_tried_at_first_solution is not None
    ]
    return {
        "solved_count": len(solved),
        "task_count": len(targets),
        "solve_rate": len(solved) / len(targets) if targets else 0.0,
        "mean_search_cost": sum(costs) / len(costs) if costs else 0.0,
        "mean_first_solution_cost": (
            sum(first_costs) / len(first_costs) if first_costs else None
        ),
        "frontier_candidates_tried_total": index.candidates_tried_total,
        "hit_budget": index.hit_budget,
        "unique_outputs": index.unique_outputs,
    }


def candidates_to_library(
    candidates: Sequence[FrontierCandidate],
) -> tuple[LibraryItem, ...]:
    return primitive_library() + candidate_library_items(candidates)


def _solve_summary_for_candidates(
    candidates: Sequence[FrontierCandidate],
    targets: Sequence[Grid],
    config: SolveConfig,
) -> dict:
    return solve_library_summary(targets, candidates_to_library(candidates), config)


def _total_segmentation_cost(
    solutions: Sequence[Program], helpers: Sequence[FrontierCandidate]
) -> int:
    helper_outputs = {candidate.output for candidate in helpers}
    return sum(_segmentation_cost(solution, helper_outputs) for solution in solutions)


def _segmentation_cost(program: Program, helper_outputs: set[Grid]) -> int:
    if execute(program) in helper_outputs:
        return 0
    if not program.args:
        return 0
    return 1 + sum(_segmentation_cost(arg, helper_outputs) for arg in program.args)


def compression_score(candidate: FrontierCandidate, solutions: Sequence[Program]) -> int:
    return _total_segmentation_cost(solutions, []) - _total_segmentation_cost(
        solutions, [candidate]
    )

def _tie_break(text: str) -> tuple[int, ...]:
    return tuple(-ord(char) for char in text)
