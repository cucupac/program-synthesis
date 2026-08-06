"""Run the formal selector experiment after review."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import random
import time

from experiment.commands.run_selection import _motif_candidates
from experiment.frontier_promotion import FrontierCandidate, frontier_promotion_menu, menu_diagnostics
from experiment.generator import DEFAULT_CONFIG_PATH, Task, load_config, make_world
from experiment.generator import spearman_rho
from experiment.selection import (
    SelectionResult,
    candidates_to_library,
    greedy_by_frontier_score_with_cost,
    greedy_by_solved_count_with_cost,
    select_compression_k_with_cost,
    select_most_frequent_k,
    select_random_k,
    solve_library_summary,
)
from experiment.solver import SolveConfig, primitive_library, solve_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "full_selection_experiment_primary_k10"
SMOKE_EXPERIMENT_NAME = "full_selection_experiment_smoke"
LEGACY_EXPERIMENT_NAMES = {"full_selection_experiment", EXPERIMENT_NAME}
FORMAL_SEEDS = tuple(range(6481, 6511))
SMOKE_SEEDS = (6460,)
CONDITIONS = (
    "reversed_a0",
    "reversed_a05",
    "reversed_a1",
    "permuted_a0",
    "permuted_a05",
    "permuted_a1",
    "stale_reversed",
)
DEFAULT_OUTPUT_PATH = "experiment/data/selection/full_selection_experiment.json"
SMOKE_OUTPUT_PATH = "experiment/data/selection/full_selection_experiment_smoke.json"
DEFAULT_WORKERS = 6
DEFAULT_K = 10
RANDOM_DRAWS = 20
MATCHED_STARTER_COUNT = 25
ASSISTED_VALIDATION_SOLVE_CONFIG = SolveConfig(node_budget=90_000, max_solutions=1)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    payload = (
        run_smoke_experiment(workers=args.workers)
        if args.smoke
        else run_formal_experiment(workers=args.workers)
    )
    output_path = Path(SMOKE_OUTPUT_PATH if args.smoke else DEFAULT_OUTPUT_PATH)
    print(
        f"{payload['experiment_name']}: cells={len(payload['cells'])} "
        f"smoke={payload['smoke']} wrote {output_path}"
    )


def run_formal_experiment(*, workers: int = DEFAULT_WORKERS) -> dict:
    output_path = Path(DEFAULT_OUTPUT_PATH)
    return _run_cells(
        config_path=DEFAULT_CONFIG_PATH,
        output_path=output_path,
        seeds=FORMAL_SEEDS,
        conditions=CONDITIONS,
        k=DEFAULT_K,
        random_draws=RANDOM_DRAWS,
        workers=workers,
        force=False,
        smoke=False,
    )


def run_smoke_experiment(*, workers: int = 1) -> dict:
    return _run_cells(
        config_path=DEFAULT_CONFIG_PATH,
        output_path=Path(SMOKE_OUTPUT_PATH),
        seeds=SMOKE_SEEDS,
        conditions=("reversed_a0",),
        k=1,
        random_draws=1,
        workers=workers,
        force=True,
        smoke=True,
    )


def _run_cells(
    *,
    config_path: str,
    output_path: Path,
    seeds: Sequence[int],
    conditions: Sequence[str],
    k: int,
    random_draws: int,
    workers: int,
    force: bool,
    smoke: bool,
) -> dict:
    _require_exact_integer_seeds(seeds)
    if k < 1:
        raise ValueError("k must be at least 1")
    if random_draws < 1:
        raise ValueError("random_draws must be at least 1")
    _validate_fixed_run_shape(
        config_path=config_path,
        output_path=output_path,
        seeds=seeds,
        conditions=conditions,
        k=k,
        random_draws=random_draws,
        force=force,
        smoke=smoke,
    )
    config_path = _repo_path(config_path)
    output_path = _repo_path(output_path)
    config = load_config(str(config_path))
    condition_map = {condition.name: condition for condition in config.conditions}
    missing = set(conditions) - set(condition_map)
    if missing:
        raise ValueError(f"config missing conditions: {sorted(missing)}")

    formal_run = bool(set(seeds) & set(FORMAL_SEEDS))
    experiment_name = EXPERIMENT_NAME if formal_run else SMOKE_EXPERIMENT_NAME
    cell_dir = output_path.with_suffix("") / "cells"
    run_started = time.perf_counter()
    if formal_run:
        _claim_formal_run_directory(cell_dir)
        _progress(
            f"FORMAL START cells={len(seeds) * len(conditions)} "
            f"workers={workers} output={output_path}"
        )
    else:
        cell_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = build_cell_jobs(
        config_path=config_path,
        cell_dir=cell_dir,
        seeds=seeds,
        conditions=conditions,
        k=k,
        random_draws=random_draws,
        workers=workers,
        force=force,
        experiment_name=experiment_name,
    )

    if formal_run:
        def compute_formal_cell_from_world(*, world, job: dict) -> dict:
            # ponytail: duplicated formal body keeps formal seeds out of importable cell helpers.
            started = time.perf_counter()
            seed = job["seed"]
            condition_name = job["condition"]
            k = job["k"]
            random_draws = job["random_draws"]
            selector_workers = job["selector_workers"]
            menu = frontier_promotion_menu(world.tasks_start)
            solve_config = SolveConfig(max_solutions=1)
            test_targets = [task.target for task in world.tasks_test]
            val_targets = [task.target for task in world.tasks_val]
            matched_start = matched_starter_tasks(world.tasks_start, seed, MATCHED_STARTER_COUNT)
            matched_targets = [task.target for task in matched_start]

            matched_results = solve_tasks(matched_targets, primitive_library(), SolveConfig(max_solutions=3))
            validation_results = solve_tasks(val_targets, primitive_library(), SolveConfig(max_solutions=3))
            validation_assisted_results = solve_tasks(
                val_targets,
                primitive_library(),
                ASSISTED_VALIDATION_SOLVE_CONFIG,
            )
            all_start_results = solve_tasks(
                [task.target for task in world.tasks_start],
                primitive_library(),
                SolveConfig(max_solutions=3),
            )
            matched_solutions = _canonical_solutions_from_results(matched_results)
            validation_skip_solutions = _canonical_solutions_from_results(validation_results)
            validation_assisted_solutions = _canonical_solutions_from_results(validation_assisted_results)
            all_start_solutions = _canonical_solutions_from_results(all_start_results)

            selections = {
                "primitives_only": _static_selection(()),
                "most_frequent_k": _static_selection(select_most_frequent_k(menu.candidates, k)),
                "compression_on_matched_25_starter": select_compression_k_with_cost(
                    menu.candidates, matched_solutions, k
                ),
                "utility_on_matched_25_starter": greedy_by_frontier_score_with_cost(
                    menu.candidates, matched_targets, k, workers=selector_workers
                ),
                "compression_on_validation_skip": _selection_with_input_search_cost(
                    select_compression_k_with_cost(
                        menu.candidates, validation_skip_solutions, k
                    ),
                    validation_results,
                ),
                "compression_on_validation_assisted": _selection_with_input_search_cost(
                    select_compression_k_with_cost(
                        menu.candidates, validation_assisted_solutions, k
                    ),
                    validation_assisted_results,
                ),
                "utility_on_validation": greedy_by_frontier_score_with_cost(
                    menu.candidates, val_targets, k, workers=selector_workers
                ),
                "compression_on_all_100_starter": select_compression_k_with_cost(
                    menu.candidates, all_start_solutions, k
                ),
                "best_k_from_c_oracle": greedy_by_solved_count_with_cost(
                    menu.candidates, test_targets, k, workers=selector_workers
                ),
                "hidden_motif_oracle": greedy_by_solved_count_with_cost(
                    _motif_candidates(world.motifs), test_targets, k, workers=selector_workers
                ),
            }
            random_rows = [
                _arm_row(
                    f"random_{draw:02d}",
                    selected := select_random_k(
                        menu.candidates, k, f"{seed}:{condition_name}:{draw}"
                    ),
                    solve_library_summary(test_targets, candidates_to_library(selected), solve_config),
                    _zero_selection_cost(),
                    solve_library_summary(val_targets, candidates_to_library(selected), solve_config),
                    _motif_recovery(selected, world.motifs),
                )
                for draw in range(random_draws)
            ]
            arm_rows = {
                name: _arm_row(
                    name,
                    selection.candidates,
                    solve_library_summary(
                        test_targets,
                        primitive_library()
                        if name == "primitives_only"
                        else candidates_to_library(selection.candidates),
                        solve_config,
                    ),
                    selection.cost,
                    solve_library_summary(
                        val_targets,
                        primitive_library()
                        if name == "primitives_only"
                        else candidates_to_library(selection.candidates),
                        solve_config,
                    )
                    if name != "hidden_motif_oracle"
                    else None,
                    _motif_recovery(selection.candidates, world.motifs),
                )
                for name, selection in selections.items()
            }
            arm_rows["random_k"] = {
                "draws": random_rows,
                "median_solved_count": _median(
                    row["summary"]["solved_count"] for row in random_rows
                ),
            }

            primitive = arm_rows["primitives_only"]["summary"]["solved_count"]
            best = arm_rows["best_k_from_c_oracle"]["summary"]["solved_count"]
            motif = arm_rows["hidden_motif_oracle"]["summary"]["solved_count"]
            motif_delta = motif - primitive
            c_delta = best - primitive

            return {
                "experiment_name": job["experiment_name"],
                "seed": seed,
                "condition": condition_name,
                "formal_seed": True,
                "k": k,
                "random_draws": random_draws,
                "world_metadata": {
                    "realized_rho": world.metadata["realized_rho"],
                    "density_summary": world.metadata["density_summary"],
                    "expected_motif_length": world.metadata["expected_motif_length"],
                },
                "menu": menu_diagnostics(menu),
                "matched_starter_task_ids": [task.id for task in matched_start],
                "matched_starter_solved_count": sum(result.solved for result in matched_results),
                "validation_skip_solved_count": sum(result.solved for result in validation_results),
                "compression_input_diagnostics": {
                    "matched_25_starter": _solution_input_diagnostics(matched_results),
                    "validation_skip": _solution_input_diagnostics(validation_results),
                    "validation_assisted": {
                        **_solution_input_diagnostics(validation_assisted_results),
                        "solve_config": {
                            "node_budget": ASSISTED_VALIDATION_SOLVE_CONFIG.node_budget,
                            "max_program_size": ASSISTED_VALIDATION_SOLVE_CONFIG.max_program_size,
                            "max_solutions": ASSISTED_VALIDATION_SOLVE_CONFIG.max_solutions,
                        },
                    },
                    "all_100_starter": _solution_input_diagnostics(all_start_results),
                },
                "arms": arm_rows,
                "validation_test_pairs": _validation_test_pairs(arm_rows, random_rows),
                "selected_set_overlap": _selected_set_overlap(arm_rows),
                "capture_ratio": c_delta / motif_delta if motif_delta > 0 else None,
                "c_delta": c_delta,
                "motif_oracle_delta": motif_delta,
                "wall_clock_seconds": round(time.perf_counter() - started, 3),
            }

        def run_formal_job(job: dict, index: int, total: int) -> dict:
            path = Path(job["cell_path"])
            _validate_formal_cell_job(job)
            if path.exists():
                raise RuntimeError(f"formal cell artifact already exists: {path}")
            cell_started = time.perf_counter()
            _progress(
                f"CELL START {index:03d}/{total} seed={job['seed']} "
                f"condition={job['condition']}"
            )
            world = make_world(config, job["seed"], condition_map[job["condition"]])
            cell = compute_formal_cell_from_world(world=world, job=job)
            _validate_formal_cell_payload(cell, job)
            _write_json(path, cell)
            total_elapsed = time.perf_counter() - run_started
            eta = total_elapsed / index * (total - index)
            _progress(
                f"CELL DONE {index:03d}/{total} seed={job['seed']} "
                f"condition={job['condition']} "
                f"elapsed={_format_duration(time.perf_counter() - cell_started)} "
                f"total={_format_duration(total_elapsed)} "
                f"eta={_format_duration(eta)}"
            )
            return cell

        total = len(jobs)
        cells = [
            run_formal_job(job, index, total)
            for index, job in enumerate(jobs, start=1)
        ]
    elif workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            cells = list(executor.map(_run_or_load_cell, jobs))
    else:
        cells = [_run_or_load_cell(job) for job in jobs]

    if formal_run:
        _progress(
            f"AGGREGATE START cells={len(cells)} "
            f"elapsed={_format_duration(time.perf_counter() - run_started)}"
        )
    payload = {
        "experiment_name": experiment_name,
        "smoke": smoke,
        "registration": _registration(seeds, conditions, k, random_draws, experiment_name),
        "cells": sorted(cells, key=lambda cell: (cell["seed"], cell["condition"])),
        "aggregates": _aggregates(cells),
    }
    _write_json(output_path, payload)
    return payload


def build_cell_jobs(
    *,
    config_path: str,
    cell_dir: Path,
    seeds: Sequence[int],
    conditions: Sequence[str],
    k: int,
    random_draws: int,
    workers: int,
    force: bool,
    experiment_name: str = EXPERIMENT_NAME,
) -> list[dict]:
    _require_exact_integer_seeds(seeds)
    return [
        {
            "config_path": config_path,
            "seed": seed,
            "condition": condition,
            "k": k,
            "random_draws": random_draws,
            "selector_workers": 1 if workers > 1 else None,
            "cell_path": str(_cell_path(cell_dir, seed, condition)),
            "force": force,
            "experiment_name": experiment_name,
        }
        for seed in seeds
        for condition in conditions
    ]


def _run_or_load_cell(job: dict) -> dict:
    _require_exact_integer_seed(job["seed"])
    path = Path(job["cell_path"])
    if job["seed"] in FORMAL_SEEDS:
        raise RuntimeError("formal cells must run through the registered formal path")
    if path.exists() and not job["force"]:
        cell = json.loads(path.read_text(encoding="utf-8"))
        _validate_loaded_cell(cell, job)
        return cell
    cell = run_cell(
        config_path=job["config_path"],
        seed=job["seed"],
        condition_name=job["condition"],
        k=job["k"],
        random_draws=job["random_draws"],
        selector_workers=job["selector_workers"],
        experiment_name=job["experiment_name"],
    )
    _write_json(path, cell)
    return cell


def run_cell(
    *,
    config_path: str,
    seed: int,
    condition_name: str,
    k: int,
    random_draws: int,
    selector_workers: int | None,
    experiment_name: str = EXPERIMENT_NAME,
) -> dict:
    _require_exact_integer_seed(seed)
    if seed in FORMAL_SEEDS:
        raise RuntimeError("formal seed cells must use the registered formal path")
    return _compute_cell(
        config_path=config_path,
        seed=seed,
        condition_name=condition_name,
        k=k,
        random_draws=random_draws,
        selector_workers=selector_workers,
        experiment_name=experiment_name,
    )


def _compute_cell(
    *,
    config_path: str,
    seed: int,
    condition_name: str,
    k: int,
    random_draws: int,
    selector_workers: int | None,
    experiment_name: str = EXPERIMENT_NAME,
) -> dict:
    _require_exact_integer_seed(seed)
    if seed in FORMAL_SEEDS:
        raise RuntimeError("formal seed cells must use the registered formal path")
    return _compute_cell_body(
        config_path=config_path,
        seed=seed,
        condition_name=condition_name,
        k=k,
        random_draws=random_draws,
        selector_workers=selector_workers,
        experiment_name=experiment_name,
    )


def _compute_cell_body(
    *,
    config_path: str,
    seed: int,
    condition_name: str,
    k: int,
    random_draws: int,
    selector_workers: int | None,
    experiment_name: str,
) -> dict:
    _require_exact_integer_seed(seed)
    if seed in FORMAL_SEEDS:
        raise RuntimeError("formal seed cells must use the registered formal path")
    started = time.perf_counter()
    config = load_config(config_path)
    condition = {condition.name: condition for condition in config.conditions}[condition_name]
    world = make_world(config, seed, condition)
    return _compute_cell_from_world(
        world=world,
        seed=seed,
        condition_name=condition_name,
        k=k,
        random_draws=random_draws,
        selector_workers=selector_workers,
        experiment_name=experiment_name,
        started=started,
    )


def _compute_cell_from_world(
    *,
    world,
    seed: int,
    condition_name: str,
    k: int,
    random_draws: int,
    selector_workers: int | None,
    experiment_name: str,
    started: float | None = None,
) -> dict:
    _require_exact_integer_seed(seed)
    if seed in FORMAL_SEEDS:
        raise RuntimeError("formal seed cells must use the registered formal path")
    world_seed = getattr(world, "world_seed", seed)
    _require_exact_integer_seed(world_seed, label="world_seed")
    if world_seed in FORMAL_SEEDS:
        raise RuntimeError("formal seed worlds must use the registered formal path")
    started = time.perf_counter() if started is None else started
    menu = frontier_promotion_menu(world.tasks_start)
    solve_config = SolveConfig(max_solutions=1)
    test_targets = [task.target for task in world.tasks_test]
    val_targets = [task.target for task in world.tasks_val]
    matched_start = matched_starter_tasks(world.tasks_start, seed, MATCHED_STARTER_COUNT)
    matched_targets = [task.target for task in matched_start]

    matched_results = solve_tasks(matched_targets, primitive_library(), SolveConfig(max_solutions=3))
    validation_results = solve_tasks(val_targets, primitive_library(), SolveConfig(max_solutions=3))
    validation_assisted_results = solve_tasks(
        val_targets,
        primitive_library(),
        ASSISTED_VALIDATION_SOLVE_CONFIG,
    )
    all_start_results = solve_tasks(
        [task.target for task in world.tasks_start],
        primitive_library(),
        SolveConfig(max_solutions=3),
    )
    matched_solutions = _canonical_solutions_from_results(matched_results)
    validation_skip_solutions = _canonical_solutions_from_results(validation_results)
    validation_assisted_solutions = _canonical_solutions_from_results(validation_assisted_results)
    all_start_solutions = _canonical_solutions_from_results(all_start_results)

    selections = {
        "primitives_only": _static_selection(()),
        "most_frequent_k": _static_selection(select_most_frequent_k(menu.candidates, k)),
        "compression_on_matched_25_starter": select_compression_k_with_cost(
            menu.candidates, matched_solutions, k
        ),
        "utility_on_matched_25_starter": greedy_by_frontier_score_with_cost(
            menu.candidates, matched_targets, k, workers=selector_workers
        ),
        "compression_on_validation_skip": _selection_with_input_search_cost(
            select_compression_k_with_cost(
                menu.candidates, validation_skip_solutions, k
            ),
            validation_results,
        ),
        "compression_on_validation_assisted": _selection_with_input_search_cost(
            select_compression_k_with_cost(
                menu.candidates, validation_assisted_solutions, k
            ),
            validation_assisted_results,
        ),
        "utility_on_validation": greedy_by_frontier_score_with_cost(
            menu.candidates, val_targets, k, workers=selector_workers
        ),
        "compression_on_all_100_starter": select_compression_k_with_cost(
            menu.candidates, all_start_solutions, k
        ),
        "best_k_from_c_oracle": greedy_by_solved_count_with_cost(
            menu.candidates, test_targets, k, workers=selector_workers
        ),
        "hidden_motif_oracle": greedy_by_solved_count_with_cost(
            _motif_candidates(world.motifs), test_targets, k, workers=selector_workers
        ),
    }
    random_rows = [
        _arm_row(
            f"random_{draw:02d}",
            selected := select_random_k(menu.candidates, k, f"{seed}:{condition_name}:{draw}"),
            solve_library_summary(test_targets, candidates_to_library(selected), solve_config),
            _zero_selection_cost(),
            solve_library_summary(val_targets, candidates_to_library(selected), solve_config),
            _motif_recovery(selected, world.motifs),
        )
        for draw in range(random_draws)
    ]
    arm_rows = {
        name: _arm_row(
            name,
            selection.candidates,
            solve_library_summary(
                test_targets,
                primitive_library()
                if name == "primitives_only"
                else candidates_to_library(selection.candidates),
                solve_config,
            ),
            selection.cost,
            solve_library_summary(
                val_targets,
                primitive_library()
                if name == "primitives_only"
                else candidates_to_library(selection.candidates),
                solve_config,
            )
            if name != "hidden_motif_oracle"
            else None,
            _motif_recovery(selection.candidates, world.motifs),
        )
        for name, selection in selections.items()
    }
    arm_rows["random_k"] = {
        "draws": random_rows,
        "median_solved_count": _median(
            row["summary"]["solved_count"] for row in random_rows
        ),
    }

    primitive = arm_rows["primitives_only"]["summary"]["solved_count"]
    best = arm_rows["best_k_from_c_oracle"]["summary"]["solved_count"]
    motif = arm_rows["hidden_motif_oracle"]["summary"]["solved_count"]
    motif_delta = motif - primitive
    c_delta = best - primitive

    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "condition": condition_name,
        "formal_seed": seed in FORMAL_SEEDS,
        "k": k,
        "random_draws": random_draws,
        "world_metadata": {
            "realized_rho": world.metadata["realized_rho"],
            "density_summary": world.metadata["density_summary"],
            "expected_motif_length": world.metadata["expected_motif_length"],
        },
        "menu": menu_diagnostics(menu),
        "matched_starter_task_ids": [task.id for task in matched_start],
        "matched_starter_solved_count": sum(result.solved for result in matched_results),
        "validation_skip_solved_count": sum(result.solved for result in validation_results),
        "compression_input_diagnostics": {
            "matched_25_starter": _solution_input_diagnostics(matched_results),
            "validation_skip": _solution_input_diagnostics(validation_results),
            "validation_assisted": {
                **_solution_input_diagnostics(validation_assisted_results),
                "solve_config": {
                    "node_budget": ASSISTED_VALIDATION_SOLVE_CONFIG.node_budget,
                    "max_program_size": ASSISTED_VALIDATION_SOLVE_CONFIG.max_program_size,
                    "max_solutions": ASSISTED_VALIDATION_SOLVE_CONFIG.max_solutions,
                },
            },
            "all_100_starter": _solution_input_diagnostics(all_start_results),
        },
        "arms": arm_rows,
        "validation_test_pairs": _validation_test_pairs(arm_rows, random_rows),
        "selected_set_overlap": _selected_set_overlap(arm_rows),
        "capture_ratio": c_delta / motif_delta if motif_delta > 0 else None,
        "c_delta": c_delta,
        "motif_oracle_delta": motif_delta,
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }


def matched_starter_tasks(tasks: Sequence[Task], seed: int, count: int) -> tuple[Task, ...]:
    _require_exact_integer_seed(seed)
    rng = random.Random(f"cell25:{seed}")
    indexes = list(range(len(tasks)))
    rng.shuffle(indexes)
    return tuple(tasks[index] for index in indexes[:count])


def _solutions_from_results(results) -> list:
    return [solution for result in results for solution in result.solutions]


def _canonical_solutions_from_results(results) -> list:
    return [result.solutions[0] for result in results if result.solutions]


def _solution_input_diagnostics(results) -> dict:
    return {
        "solved_task_count": sum(result.solved for result in results),
        "solution_program_count_before_canonicalization": sum(
            len(result.solutions) for result in results
        ),
        "canonical_solution_count": len(_canonical_solutions_from_results(results)),
        "candidate_programs_tried_total": sum(
            result.candidates_tried_total for result in results
        ),
    }


def _static_selection(candidates: Sequence[FrontierCandidate]):
    return SelectionResult(tuple(candidates), _zero_selection_cost())


def _selection_with_input_search_cost(selection: SelectionResult, results) -> SelectionResult:
    input_cost = sum(result.candidates_tried_total for result in results)
    cost = dict(selection.cost)
    cost["input_solution_search_candidate_programs_tried"] = input_cost
    cost["selection_cost_candidate_programs_tried"] = (
        cost.get("selection_cost_candidate_programs_tried", 0) + input_cost
    )
    return SelectionResult(selection.candidates, cost)


def _zero_selection_cost() -> dict:
    return {
        "selection_cost_candidate_programs_tried": 0,
        "input_solution_search_candidate_programs_tried": 0,
        "trial_libraries_evaluated": 0,
        "segmentation_evaluations": 0,
        "solution_segmentations_evaluated": 0,
        "frontier_candidates_tried_total": 0,
    }


def _arm_row(
    name: str,
    selected: Sequence[FrontierCandidate],
    summary: dict,
    selection_cost: dict,
    validation_summary: dict | None = None,
    motif_recovery: dict | None = None,
) -> dict:
    return {
        "arm": name,
        "selected_programs": [candidate.program_string for candidate in selected],
        "selection_cost": selection_cost,
        "summary": summary,
        "validation_summary": validation_summary,
        "motif_recovery": motif_recovery,
    }


def _motif_recovery(selected: Sequence[FrontierCandidate], motifs) -> dict:
    motif_by_output = {motif.target: motif.id for motif in motifs}
    matched = sorted(
        {motif_by_output[candidate.output] for candidate in selected if candidate.output in motif_by_output}
    )
    selected_count = len(selected)
    motif_count = len(motifs)
    return {
        "selected_count": selected_count,
        "motif_match_count": len(matched),
        "precision": len(matched) / selected_count if selected_count else 0.0,
        "recall": len(matched) / motif_count if motif_count else 0.0,
        "matched_motif_ids": matched,
    }


def _validation_test_pairs(arm_rows: dict, random_rows: Sequence[dict]) -> list[dict]:
    rows = []
    primitive = arm_rows["primitives_only"]
    primitive_validation = primitive.get("validation_summary") or {}
    baseline_validation = primitive_validation.get("solved_count")
    baseline_test = primitive["summary"]["solved_count"]
    for row in random_rows:
        rows.append(_validation_pair(row, baseline_validation, baseline_test))
    for name, row in arm_rows.items():
        if name in {"primitives_only", "random_k", "hidden_motif_oracle"}:
            continue
        rows.append(_validation_pair(row, baseline_validation, baseline_test))
    return rows


def _validation_pair(row: dict, baseline_validation: int | None, baseline_test: int) -> dict:
    validation = row.get("validation_summary") or {}
    validation_solved = validation.get("solved_count")
    test_solved = row["summary"]["solved_count"]
    return {
        "arm": row["arm"],
        "validation_solved_count": validation_solved,
        "test_solved_count": test_solved,
        "validation_solved_gain": (
            None if validation_solved is None or baseline_validation is None
            else validation_solved - baseline_validation
        ),
        "test_solved_gain": test_solved - baseline_test,
    }


def _selected_set_overlap(arm_rows: dict) -> dict:
    selected = {
        name: set(row.get("selected_programs", ()))
        for name, row in arm_rows.items()
        if name != "random_k"
    }
    return {
        f"{left}|{right}": len(selected[left] & selected[right])
        for index, left in enumerate(selected)
        for right in list(selected)[index + 1 :]
    }


def _registration(seeds, conditions, k, random_draws, experiment_name=EXPERIMENT_NAME) -> dict:
    _require_exact_integer_seeds(seeds)
    return {
        "experiment_name": experiment_name,
        "seeds": list(seeds),
        "conditions": list(conditions),
        "k": k,
        "random_draws": random_draws,
        "arms": _arm_definitions(),
        "menu_recipe": "frontier_promotion",
        "assisted_validation_solve_config": {
            "node_budget": ASSISTED_VALIDATION_SOLVE_CONFIG.node_budget,
            "max_program_size": ASSISTED_VALIDATION_SOLVE_CONFIG.max_program_size,
            "max_solutions": ASSISTED_VALIDATION_SOLVE_CONFIG.max_solutions,
        },
        "selection_cost_policy": {
            "unit": "candidate programs tried",
            "starter_solution_search": "treated as pre-existing and not charged",
            "validation_skip_solution_search": "charge skip search only",
            "validation_assisted_solution_search": "charge assisted search only",
            "compression_segmentation_work": (
                "report separately; exclude from candidate-program totals"
            ),
        },
        "break_even_policy": {
            "comparison": (
                "utility_on_validation vs compression_on_validation_assisted"
            ),
            "upfront_cost": (
                "utility selection cost minus assisted-compression selection cost"
            ),
            "negative_increment": "floor at zero",
        },
        "k_sensitivity_status": "registered_follow_up_not_run_in_this_command",
        "matched_25_rule": "random.Random(f'cell25:{seed}') over starter tasks",
        "primary_metric": "solved_count_delta",
        "secondary_metric": "mean_search_cost_savings",
        "primary_data_effect": (
            "compression_on_validation_assisted - "
            "compression_on_matched_25_starter"
        ),
        "primary_scoring_effect": (
            "utility_on_validation - compression_on_validation_assisted"
        ),
        "robustness_data_effect": (
            "compression_on_validation_skip - compression_on_matched_25_starter"
        ),
        "stale_reversed": "reported separately; excluded from main rho curves",
    }


def _arm_definitions() -> dict:
    return {
        "primitives_only": "primitive solver baseline",
        "random_k": "20 deterministic random K draws from C",
        "most_frequent_k": "top K by starter-task support",
        "compression_on_matched_25_starter": "compression score on matched 25 starter tasks",
        "utility_on_matched_25_starter": "utility score on matched 25 starter tasks",
        "compression_on_validation_skip": "compression score on primitive-solved validation solutions only",
        "compression_on_validation_assisted": "compression score on assisted validation solutions",
        "utility_on_validation": "utility score on validation targets",
        "compression_on_all_100_starter": "compression score on all primitive-solved starter solutions",
        "best_k_from_c_oracle": "test-peeking best K from C diagnostic",
        "hidden_motif_oracle": "hidden motif diagnostic oracle",
    }


def _aggregates(cells: Sequence[dict]) -> dict:
    non_stale = [cell for cell in cells if cell["condition"] != "stale_reversed"]
    stale = [cell for cell in cells if cell["condition"] == "stale_reversed"]
    return {
        "data_effect_by_rho": _effect_rows(
            non_stale,
            "compression_on_validation_assisted",
            "compression_on_matched_25_starter",
        ),
        "scoring_effect_by_rho": _effect_rows(
            non_stale,
            "utility_on_validation",
            "compression_on_validation_assisted",
        ),
        "stale_foresight": _effect_rows(
            stale,
            "utility_on_validation",
            "compression_on_validation_assisted",
        ),
        "arm_solved_summary": _arm_solved_summary(cells),
        "break_even": _break_even_rows(cells),
        "validation_test_prediction": _validation_test_prediction(cells),
    }


def _effect_rows(cells: Sequence[dict], left: str, right: str) -> list[dict]:
    return [
        {
            "seed": cell["seed"],
            "condition": cell["condition"],
            "realized_start_test_rho": cell["world_metadata"]["realized_rho"][
                "realized_start_test"
            ],
            "solved_delta": (
                cell["arms"][left]["summary"]["solved_count"]
                - cell["arms"][right]["summary"]["solved_count"]
            ),
            "mean_search_cost_delta": (
                cell["arms"][left]["summary"]["mean_search_cost"]
                - cell["arms"][right]["summary"]["mean_search_cost"]
            ),
            "mean_search_cost_savings": (
                cell["arms"][right]["summary"]["mean_search_cost"]
                - cell["arms"][left]["summary"]["mean_search_cost"]
            ),
        }
        for cell in cells
    ]


def _arm_solved_summary(cells: Sequence[dict]) -> dict:
    values: dict[str, list[int]] = {}
    for cell in cells:
        for name, row in cell["arms"].items():
            if name == "random_k":
                continue
            values.setdefault(name, []).append(row["summary"]["solved_count"])
    return {
        name: {"mean": sum(items) / len(items), "min": min(items), "max": max(items)}
        for name, items in values.items()
    }


def _break_even_rows(cells: Sequence[dict]) -> list[dict]:
    rows = []
    for cell in cells:
        utility = cell["arms"]["utility_on_validation"]["summary"]
        compression = cell["arms"]["compression_on_validation_assisted"]["summary"]
        per_task_savings = compression["mean_search_cost"] - utility["mean_search_cost"]
        utility_upfront = cell["arms"]["utility_on_validation"]["selection_cost"][
            "selection_cost_candidate_programs_tried"
        ]
        compression_upfront = cell["arms"]["compression_on_validation_assisted"][
            "selection_cost"
        ][
            "selection_cost_candidate_programs_tried"
        ]
        incremental_upfront = max(0, utility_upfront - compression_upfront)
        rows.append(
            {
                "seed": cell["seed"],
                "condition": cell["condition"],
                "comparison": "utility_on_validation_vs_compression_on_validation_assisted",
                "utility_selection_cost": utility_upfront,
                "compression_selection_cost": compression_upfront,
                "incremental_selection_cost": incremental_upfront,
                "utility_vs_compression_mean_search_cost_savings": per_task_savings,
                "break_even_future_tasks": (
                    None
                    if per_task_savings <= 0
                    else math.ceil(incremental_upfront / per_task_savings)
                ),
            }
        )
    return rows


def _validation_test_prediction(cells: Sequence[dict]) -> dict:
    pairs = [
        pair
        for cell in cells
        if cell["condition"] != "stale_reversed"
        for pair in cell.get("validation_test_pairs", ())
        if pair.get("validation_solved_gain") is not None
    ]
    if not pairs:
        return {"spearman_rho": None, "pairs": []}
    return {
        "spearman_rho": spearman_rho(
            [pair["validation_solved_gain"] for pair in pairs],
            [pair["test_solved_gain"] for pair in pairs],
        ),
        "pairs": pairs,
    }


def _claim_formal_run_directory(cell_dir: Path) -> None:
    _fresh_look_guard(cell_dir)
    formal_dir = cell_dir.parent
    formal_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        formal_dir.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"formal run already claimed: {formal_dir}") from exc
    cell_dir.mkdir()


def _fresh_look_guard(cell_dir: Path) -> None:
    formal_output = _repo_path(DEFAULT_OUTPUT_PATH)
    if formal_output.exists():
        raise RuntimeError(f"fresh-look guard blocked by prior artifact: {formal_output}")
    formal_dir = formal_output.with_suffix("")
    if formal_dir.exists():
        raise RuntimeError(f"fresh-look guard blocked by formal artifact directory: {formal_dir}")
    cell_dir = cell_dir.resolve()
    if cell_dir.exists() and any(cell_dir.iterdir()):
        raise RuntimeError(f"fresh-look guard blocked by formal cell artifacts: {cell_dir}")
    selection_dir = _repo_path("experiment/data/selection")
    if not selection_dir.exists():
        return
    for path in selection_dir.rglob("*"):
        if path.is_dir():
            continue
        formal_like = _looks_like_formal_artifact(path)
        if path.suffix != ".json":
            if formal_like:
                raise RuntimeError(f"fresh-look guard blocked formal-looking artifact: {path}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if formal_like:
                raise RuntimeError(f"fresh-look guard blocked malformed artifact: {path}")
            continue

        artifact_name = data.get("experiment_name")
        registration_name = data.get("registration", {}).get("experiment_name")
        if artifact_name in LEGACY_EXPERIMENT_NAMES or registration_name in LEGACY_EXPERIMENT_NAMES:
            raise RuntimeError(f"fresh-look guard blocked by prior formal artifact: {path}")
        if formal_like:
            raise RuntimeError(f"fresh-look guard blocked malformed artifact: {path}")
        registered_seeds = data.get("registration", {}).get("seeds", data.get("seeds", ()))
        recorded = (
            tuple(registered_seeds)
            if isinstance(registered_seeds, (list, tuple, set))
            else (registered_seeds,)
        )
        if "seed" in data:
            recorded += (data["seed"],)
        if any(_metadata_value_is_formal_seed(seed) for seed in recorded):
            raise RuntimeError(f"fresh-look guard blocked by prior artifact: {path}")


def _looks_like_formal_artifact(path: Path) -> bool:
    try:
        path.resolve().relative_to(_repo_path("experiment/data/selection").resolve())
    except ValueError:
        return False
    return (
        path.name == "full_selection_experiment.json"
        or path.name == f"{EXPERIMENT_NAME}.json"
        or any(str(seed) in path.name for seed in FORMAL_SEEDS)
        or "full_selection_experiment/" in path.as_posix()
        or f"{EXPERIMENT_NAME}/" in path.as_posix()
        or "full_selection_experiment/cells" in path.as_posix()
    )


def _validate_loaded_cell(cell: dict, job: dict) -> None:
    if cell.get("seed") != job["seed"]:
        raise ValueError(f"cached cell seed mismatch: {job['cell_path']}")
    if cell.get("condition") != job["condition"]:
        raise ValueError(f"cached cell condition mismatch: {job['cell_path']}")


def _validate_formal_cell_job(job: dict) -> None:
    _require_exact_integer_seed(job.get("seed"))
    if job.get("force"):
        raise RuntimeError("--force is not allowed for formal seed cells")
    if _repo_path(job.get("config_path")).resolve() != _repo_path(DEFAULT_CONFIG_PATH).resolve():
        raise ValueError("formal cells must use the default config path")
    if job.get("k") != DEFAULT_K:
        raise ValueError("formal cells must use K=10")
    if job.get("random_draws") != RANDOM_DRAWS:
        raise ValueError("formal cells must use 20 random draws")
    if job.get("condition") not in CONDITIONS:
        raise ValueError("formal cells must use a registered condition")
    expected = _cell_path(
        _repo_path(DEFAULT_OUTPUT_PATH).with_suffix("") / "cells",
        job["seed"],
        job["condition"],
    ).resolve()
    if Path(job["cell_path"]).resolve() != expected:
        raise ValueError("formal cells must use the formal cell path")


def _validate_formal_cell_payload(cell: dict, job: dict) -> None:
    def fail(detail: str) -> None:
        raise RuntimeError(f"formal cell invariant failed: {detail}")

    expected_metadata = {
        "experiment_name": job["experiment_name"],
        "seed": job["seed"],
        "condition": job["condition"],
        "formal_seed": True,
        "k": job["k"],
        "random_draws": job["random_draws"],
    }
    for field, expected in expected_metadata.items():
        if cell.get(field) != expected:
            fail(f"{field}={cell.get(field)!r}, expected {expected!r}")

    arms = cell.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(_arm_definitions()):
        fail("arm set does not match registration")

    def selected_count(row) -> int | None:
        programs = row.get("selected_programs") if isinstance(row, dict) else None
        return len(programs) if isinstance(programs, list) else None

    if selected_count(arms["primitives_only"]) != 0:
        fail("primitives_only must select zero programs")
    for name, row in arms.items():
        if name in {"primitives_only", "random_k"}:
            continue
        if selected_count(row) != job["k"]:
            fail(f"{name} must select exactly K={job['k']} programs")

    random_arm = arms["random_k"]
    draws = random_arm.get("draws") if isinstance(random_arm, dict) else None
    if not isinstance(draws, list) or len(draws) != job["random_draws"]:
        fail(f"random_k must contain exactly {job['random_draws']} draws")
    for index, row in enumerate(draws):
        if selected_count(row) != job["k"]:
            fail(f"random draw {index} must select exactly K={job['k']} programs")

    diagnostics = cell.get("compression_input_diagnostics")
    assisted_diagnostics = (
        diagnostics.get("validation_assisted") if isinstance(diagnostics, dict) else None
    )
    diagnostic_cost = (
        assisted_diagnostics.get("candidate_programs_tried_total")
        if isinstance(assisted_diagnostics, dict)
        else None
    )
    assisted_cost = arms["compression_on_validation_assisted"].get("selection_cost")
    input_cost = (
        assisted_cost.get("input_solution_search_candidate_programs_tried")
        if isinstance(assisted_cost, dict)
        else None
    )
    total_cost = (
        assisted_cost.get("selection_cost_candidate_programs_tried")
        if isinstance(assisted_cost, dict)
        else None
    )
    if type(diagnostic_cost) is not int or diagnostic_cost < 0:
        fail("assisted solution-search cost diagnostic is missing or invalid")
    if input_cost != diagnostic_cost or total_cost != diagnostic_cost:
        fail("assisted solution-search cost is missing or inconsistent")


def _validate_fixed_run_shape(
    *,
    config_path: str,
    output_path: Path,
    seeds: Sequence[int],
    conditions: Sequence[str],
    k: int,
    random_draws: int,
    force: bool,
    smoke: bool,
) -> None:
    _require_exact_integer_seeds(seeds)
    if smoke:
        if _repo_path(config_path).resolve() != _repo_path(DEFAULT_CONFIG_PATH).resolve():
            raise ValueError("smoke run must use the default config path")
        if _repo_path(output_path).resolve() != _repo_path(SMOKE_OUTPUT_PATH).resolve():
            raise ValueError("smoke run must write to the smoke output path")
        if tuple(seeds) != SMOKE_SEEDS:
            raise ValueError("smoke run must use seed 6460")
        if tuple(conditions) != ("reversed_a0",):
            raise ValueError("smoke run must use reversed_a0 only")
        if k != 1 or random_draws != 1 or not force:
            raise ValueError("smoke run must use K=1, one random draw, and force=True")
        return

    if _repo_path(config_path).resolve() != _repo_path(DEFAULT_CONFIG_PATH).resolve():
        raise ValueError("formal run must use the default config path")
    if _repo_path(output_path).resolve() != _repo_path(DEFAULT_OUTPUT_PATH).resolve():
        raise ValueError("formal run must write to the formal output path")
    if tuple(seeds) != FORMAL_SEEDS:
        raise ValueError("formal run must use the full formal seed set")
    if tuple(conditions) != CONDITIONS:
        raise ValueError("formal run must use the full formal condition set")
    if k != DEFAULT_K:
        raise ValueError("formal run must use K=10")
    if random_draws != RANDOM_DRAWS:
        raise ValueError("formal run must use 20 random draws")
    if force:
        raise RuntimeError("--force is not allowed for formal seeds")


def _cell_path(cell_dir: Path, seed: int, condition: str) -> Path:
    _require_exact_integer_seed(seed)
    return cell_dir / f"{seed}_{condition}.json"


def _require_exact_integer_seed(seed, *, label: str = "seed") -> int:
    if type(seed) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return seed


def _require_exact_integer_seeds(seeds: Sequence[int]) -> None:
    for seed in seeds:
        _require_exact_integer_seed(seed)


def _metadata_value_is_formal_seed(value) -> bool:
    if type(value) in {int, float}:
        return value in FORMAL_SEEDS
    if isinstance(value, str):
        try:
            return int(value.strip()) in FORMAL_SEEDS
        except ValueError:
            return False
    return False


def _progress(message: str) -> None:
    print(message, flush=True)


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _median(values) -> float:
    items = sorted(values)
    middle = len(items) // 2
    if len(items) % 2:
        return float(items[middle])
    return (items[middle - 1] + items[middle]) / 2


if __name__ == "__main__":
    main()
