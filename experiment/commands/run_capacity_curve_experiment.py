"""Run the registered fresh-seed K=0..20 capacity-curve experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import time

from experiment import capacity_curve as capacity
from experiment.commands.run_full_selection_experiment import (
    ASSISTED_VALIDATION_SOLVE_CONFIG,
    _canonical_solutions_from_results,
    _format_duration,
    _progress,
    _solution_input_diagnostics,
)
from experiment.commands.run_k_sweep_experiment import write_json_atomic
from experiment.dsl import program_to_string
from experiment.frontier_promotion import frontier_promotion_menu, menu_diagnostics
from experiment.generator import DEFAULT_CONFIG_PATH, load_config, make_world
from experiment.selection import (
    SelectionResult,
    candidates_to_library,
    greedy_by_frontier_score_with_cost,
    greedy_by_solved_count_with_cost,
    select_compression_k_with_cost,
    select_random_k,
    solve_library_summaries,
)
from experiment.solver import SolveConfig, primitive_library, solve_tasks


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION_ROOT = Path("experiment/data/selection")
OUTPUT_ROOT = SELECTION_ROOT / "capacity_curve"
OUTPUT_PATH = OUTPUT_ROOT / f"{capacity.EXPERIMENT_NAME}.json"
SMOKE_OUTPUT_ROOT = SELECTION_ROOT / "capacity_curve_smoke"
SMOKE_OUTPUT_PATH = (
    SMOKE_OUTPUT_ROOT / f"{capacity.SMOKE_EXPERIMENT_NAME}.json"
)
K_SWEEP_RESULTS_PATH = (
    SELECTION_ROOT / "k_sweep/full_selection_experiment_k_sweep.json"
)
EVALUATION_CONFIG = SolveConfig(max_solutions=1)
STARTER_SOLUTION_CONFIG = SolveConfig(max_solutions=3)
_SEED_METADATA_KEYS = {"seed", "seeds", "world_seed"}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    payload = (
        run_smoke_experiment(workers=args.workers)
        if args.smoke
        else run_formal_experiment(workers=args.workers)
    )
    _progress(
        f"{payload['experiment_name']}: cells={len(payload['cells'])} "
        f"smoke={payload['smoke']}"
    )


def run_formal_experiment(*, workers: int = 1) -> dict:
    return _run_experiment(workers=workers, smoke=False)


def run_smoke_experiment(*, workers: int = 1) -> dict:
    return _run_experiment(workers=workers, smoke=True)


def build_jobs(cell_root: Path, *, workers: int, smoke: bool) -> list[dict]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    seeds = capacity.SMOKE_SEEDS if smoke else capacity.FORMAL_SEEDS
    conditions = capacity.SMOKE_CONDITIONS if smoke else capacity.CONDITIONS
    jobs = [
        {
            "seed": seed,
            "condition": condition,
            "formal_seed": not smoke,
            "selector_workers": 1 if workers > 1 else None,
            "cell_path": str(cell_root / f"{seed}_{condition}.json"),
        }
        for seed in seeds
        for condition in conditions
    ]
    for index, job in enumerate(jobs, start=1):
        job.update(index=index, total=len(jobs))
    return jobs


def _run_experiment(*, workers: int, smoke: bool) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    root = _repo_path(SMOKE_OUTPUT_ROOT if smoke else OUTPUT_ROOT)
    selection_root = _repo_path(SELECTION_ROOT)
    if smoke:
        root.mkdir(parents=True, exist_ok=True)
    else:
        claim_formal_output(root, selection_root=selection_root)
    write_json_atomic(root / "registration.json", capacity.registration(smoke=smoke))
    jobs = build_jobs(root / "cells", workers=workers, smoke=smoke)
    started = time.perf_counter()
    _progress(
        f"CAPACITY CURVE{' SMOKE' if smoke else ''} START "
        f"cells={len(jobs)} workers={workers} output={root}"
    )
    cells = _execute_jobs(jobs, workers=workers)
    payload = {
        "experiment_name": (
            capacity.SMOKE_EXPERIMENT_NAME if smoke else capacity.EXPERIMENT_NAME
        ),
        "smoke": smoke,
        "registration": capacity.registration(smoke=smoke),
        "cells": sorted(
            cells,
            key=lambda cell: (
                cell["seed"],
                capacity.CONDITIONS.index(cell["condition"]),
            ),
        ),
    }
    capacity.validate_aggregate(payload, formal=not smoke)
    output = _repo_path(SMOKE_OUTPUT_PATH if smoke else OUTPUT_PATH)
    write_json_atomic(output, payload)
    total = time.perf_counter() - started
    if smoke:
        timing = cells[0]["timings"]
        _progress(
            "SMOKE TIMING "
            f"selection={_format_duration(timing['selection_seconds'])} "
            f"real_prefix={_format_duration(timing['real_prefix_seconds'])} "
            f"random_prefix={_format_duration(timing['random_prefix_seconds'])} "
            f"total={_format_duration(total)} "
            f"projected_serial={_format_duration(total * 180)}"
        )
    return payload


def _execute_jobs(jobs: list[dict], *, workers: int) -> list[dict]:
    started = time.perf_counter()

    def collect(results) -> list[dict]:
        cells = []
        for index, (cell, elapsed) in enumerate(results, start=1):
            cells.append(cell)
            total_elapsed = time.perf_counter() - started
            eta = total_elapsed / index * (len(jobs) - index)
            _progress(
                f"CELL DONE {index:03d}/{len(jobs):03d} "
                f"seed={cell['seed']} condition={cell['condition']} "
                f"elapsed={_format_duration(elapsed)} "
                f"total={_format_duration(total_elapsed)} eta={_format_duration(eta)}"
            )
        return cells

    if workers == 1:
        return collect(map(_run_job, jobs))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return collect(executor.map(_run_job, jobs))


def _run_job(job: dict) -> tuple[dict, float]:
    _validate_job(job)
    started = time.perf_counter()
    _progress(
        f"CELL START {job['index']:03d}/{job['total']:03d} "
        f"seed={job['seed']} condition={job['condition']}"
    )
    config = load_config(str(_repo_path(DEFAULT_CONFIG_PATH)))
    conditions = {condition.name: condition for condition in config.conditions}
    world = make_world(config, job["seed"], conditions[job["condition"]])
    cell = compute_capacity_cell(
        world,
        formal_seed=job["formal_seed"],
        selector_workers=job["selector_workers"],
    )
    capacity.validate_cell(cell, formal=job["formal_seed"])
    if not job["formal_seed"]:
        validate_smoke_parity(cell, _repo_path(K_SWEEP_RESULTS_PATH))
    write_json_atomic(Path(job["cell_path"]), cell)
    return cell, time.perf_counter() - started


def compute_capacity_cell(world, *, formal_seed: bool, selector_workers: int | None) -> dict:
    started = time.perf_counter()
    menu = frontier_promotion_menu(world.tasks_start)
    if len(menu.candidates) < capacity.K_MAX:
        raise RuntimeError(
            "registered capacity run requires at least 20 menu candidates; "
            f"seed={world.world_seed} condition={world.condition.name} "
            f"available={len(menu.candidates)}"
        )
    validation_targets = [task.target for task in world.tasks_val]
    test_targets = [task.target for task in world.tasks_test]
    starter_results = solve_tasks(
        [task.target for task in world.tasks_start],
        primitive_library(),
        STARTER_SOLUTION_CONFIG,
    )
    assisted_results = solve_tasks(
        validation_targets,
        primitive_library(),
        ASSISTED_VALIDATION_SOLVE_CONFIG,
    )
    starter_solutions = _canonical_solutions_from_results(starter_results)
    assisted_solutions = _canonical_solutions_from_results(assisted_results)

    selection_started = time.perf_counter()
    selections = {
        capacity.COMPRESSION_ALL_ARM: select_compression_k_with_cost(
            menu.candidates,
            starter_solutions,
            capacity.K_MAX,
            trace=True,
        ),
        capacity.COMPRESSION_VALIDATION_ARM: with_input_search_cost(
            select_compression_k_with_cost(
                menu.candidates,
                assisted_solutions,
                capacity.K_MAX,
                trace=True,
            ),
            assisted_results,
        ),
        capacity.UTILITY_ARM: greedy_by_frontier_score_with_cost(
            menu.candidates,
            validation_targets,
            capacity.K_MAX,
            workers=selector_workers,
            trace=True,
        ),
        capacity.TEST_PEEK_ARM: greedy_by_solved_count_with_cost(
            menu.candidates,
            test_targets,
            capacity.K_MAX,
            workers=selector_workers,
            trace=True,
        ),
    }
    selection_seconds = time.perf_counter() - selection_started

    primitive_validation, primitive_test = _capacity_summaries(
        (validation_targets, test_targets), primitive_library(), EVALUATION_CONFIG
    )
    primitive = {
        "validation_summary": primitive_validation,
        "test_summary": primitive_test,
    }
    real_started = time.perf_counter()
    arms = {
        name: _greedy_arm_payload(
            selection,
            validation_targets,
            test_targets,
            primitive,
        )
        for name, selection in selections.items()
    }
    real_prefix_seconds = time.perf_counter() - real_started

    random_started = time.perf_counter()
    random_draws = []
    for draw in range(capacity.RANDOM_DRAWS):
        selected = select_random_k(
            menu.candidates,
            capacity.K_MAX,
            f"{world.world_seed}:{draw}",
        )
        random_draws.append(
            {
                "draw": draw,
                "selected_programs": [item.program_string for item in selected],
                "prefixes": evaluate_prefixes(
                    selected,
                    validation_targets,
                    test_targets,
                    tuple(capacity.zero_cost() for _ in selected),
                    primitive,
                ),
            }
        )
    random_prefix_seconds = time.perf_counter() - random_started
    arms[capacity.RANDOM_ARM] = {"draws": random_draws}

    hashes, menu_programs = shared_world_hashes(world, menu=menu)
    diagnostics = menu_diagnostics(menu)
    diagnostics["cap"] = menu.cap
    return {
        "experiment_name": (
            capacity.EXPERIMENT_NAME
            if formal_seed
            else capacity.SMOKE_EXPERIMENT_NAME
        ),
        "seed": world.world_seed,
        "world_seed": world.world_seed,
        "condition": world.condition.name,
        "formal_seed": formal_seed,
        "motif_count": len(world.motifs),
        "starter_task_count": len(world.tasks_start),
        "k_max": capacity.K_MAX,
        "random_draws": capacity.RANDOM_DRAWS,
        "validation_task_count": len(world.tasks_val),
        "test_task_count": len(world.tasks_test),
        "world_metadata": {
            "condition": {
                "name": world.condition.name,
                "alt_kind": world.condition.alt_kind,
                "alpha_val": world.condition.alpha_val,
                "alpha_test": world.condition.alpha_test,
            },
            "realized_rho": world.metadata["realized_rho"],
            "density_summary": world.metadata["density_summary"],
            "expected_motif_length": world.metadata["expected_motif_length"],
        },
        "shared_hashes": hashes,
        "menu": diagnostics,
        "candidate_menu_programs": menu_programs,
        "input_solution_diagnostics": {
            "all_100_starter": _solution_input_diagnostics(starter_results),
            "validation_assisted": {
                **_solution_input_diagnostics(assisted_results),
                "solve_config": {
                    "node_budget": ASSISTED_VALIDATION_SOLVE_CONFIG.node_budget,
                    "max_program_size": ASSISTED_VALIDATION_SOLVE_CONFIG.max_program_size,
                    "max_solutions": ASSISTED_VALIDATION_SOLVE_CONFIG.max_solutions,
                },
            },
        },
        "primitive": primitive,
        "arms": arms,
        "timings": {
            "selection_seconds": round(selection_seconds, 3),
            "real_prefix_seconds": round(real_prefix_seconds, 3),
            "random_prefix_seconds": round(random_prefix_seconds, 3),
        },
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }


def _greedy_arm_payload(
    selection: SelectionResult,
    validation_targets,
    test_targets,
    primitive: dict,
) -> dict:
    if len(selection.prefix_costs) != capacity.K_MAX:
        raise RuntimeError("traced selection must contain 20 cumulative prefix costs")
    return {
        "selected_programs": [item.program_string for item in selection.candidates],
        "prefixes": evaluate_prefixes(
            selection.candidates,
            validation_targets,
            test_targets,
            selection.prefix_costs,
            primitive,
        ),
        "round_diagnostics": list(selection.round_diagnostics),
    }


def evaluate_prefixes(
    selected,
    validation_targets,
    test_targets,
    prefix_costs,
    primitive: dict,
) -> list[dict]:
    rows = [
        {
            "k": 0,
            "validation_summary": dict(primitive["validation_summary"]),
            "test_summary": dict(primitive["test_summary"]),
            "selection_cost": capacity.zero_cost(),
        }
    ]
    for k in range(1, capacity.K_MAX + 1):
        library = candidates_to_library(selected[:k])
        validation_summary, test_summary = _capacity_summaries(
            (validation_targets, test_targets), library, EVALUATION_CONFIG
        )
        rows.append(
            {
                "k": k,
                "validation_summary": validation_summary,
                "test_summary": test_summary,
                "selection_cost": dict(prefix_costs[k - 1]),
            }
        )
    return rows


def with_input_search_cost(selection: SelectionResult, results) -> SelectionResult:
    input_cost = sum(result.candidates_tried_total for result in results)

    def charged(cost: dict) -> dict:
        updated = dict(cost)
        updated["input_solution_search_candidate_programs_tried"] = input_cost
        updated["selection_cost_candidate_programs_tried"] += input_cost
        return updated

    return SelectionResult(
        selection.candidates,
        charged(selection.cost),
        tuple(charged(cost) for cost in selection.prefix_costs),
        selection.round_diagnostics,
    )


def shared_world_hashes(world, *, menu=None) -> tuple[dict, list[str]]:
    menu = frontier_promotion_menu(world.tasks_start) if menu is None else menu
    programs = [candidate.program_string for candidate in menu.candidates]
    motifs = [
        {
            "id": motif.id,
            "program": program_to_string(motif.program),
            "target": sorted([list(cell) for cell in motif.target]),
        }
        for motif in world.motifs
    ]
    tasks = [
        {
            "id": task.id,
            "split": task.split,
            "target": sorted([list(cell) for cell in task.target]),
            "hidden_program": program_to_string(task.hidden_program),
            "motif_ids": list(task.motif_ids),
            "combine_ops": list(task.combine_ops),
            "glue_ops": list(task.glue_ops),
        }
        for task in world.tasks_start
    ]
    return (
        {
            "hidden_motifs": capacity.canonical_hash(motifs),
            "p_start": capacity.canonical_hash(list(world.p_start)),
            "starter_tasks": capacity.canonical_hash(tasks),
            "candidate_menu": capacity.canonical_hash(programs),
        },
        programs,
    )


def validate_smoke_parity(cell: dict, stored_path: Path) -> None:
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    rows = {
        row["k"]: row
        for row in stored.get("cells", [])
        if row.get("seed") == cell["seed"]
        and row.get("condition") == cell["condition"]
        and row.get("k") in {2, 5, 10}
    }
    if set(rows) != {2, 5, 10}:
        raise RuntimeError("smoke parity requires stored K=2,5,10 anchor cells")
    for k, stored_cell in rows.items():
        for arm_name in capacity.GREEDY_ARMS:
            new_arm = cell["arms"][arm_name]
            old_arm = stored_cell["arms"][arm_name]
            prefix = new_arm["prefixes"][k]
            checks = {
                "selected path": new_arm["selected_programs"][:k]
                == old_arm["selected_programs"],
                "test summary": _legacy_summary(prefix["test_summary"])
                == old_arm["summary"],
                "validation summary": _legacy_summary(prefix["validation_summary"])
                == old_arm["validation_summary"],
                "selection cost": prefix["selection_cost"]
                == old_arm["selection_cost"],
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise RuntimeError(
                    f"smoke parity failed at K={k} arm={arm_name}: {', '.join(failed)}"
                )


def _capacity_summaries(target_groups, library, config) -> tuple[dict, ...]:
    summaries = solve_library_summaries(target_groups, library, config)
    for summary in summaries:
        summary["failure_count"] = summary["task_count"] - summary["solved_count"]
    return summaries


def _legacy_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "failure_count"}


def fresh_look_guard(selection_root: Path) -> None:
    if not selection_root.exists():
        return
    for path in selection_root.rglob("*"):
        if path.is_dir():
            continue
        formal_like = _looks_like_formal_artifact(path)
        if path.suffix != ".json":
            if formal_like:
                raise RuntimeError(f"fresh-look guard blocked formal artifact: {path}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if formal_like:
                raise RuntimeError(
                    f"fresh-look guard blocked malformed formal artifact: {path}"
                ) from exc
            continue
        if _contains_reserved_metadata(data) or _claims_formal_experiment(data):
            raise RuntimeError(f"fresh-look guard blocked prior formal evidence: {path}")
        if formal_like:
            raise RuntimeError(f"fresh-look guard blocked formal-looking artifact: {path}")


def claim_formal_output(output_root: Path, *, selection_root: Path) -> None:
    fresh_look_guard(selection_root.parent)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_root.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"formal capacity-curve output already claimed: {output_root}"
        ) from exc


def _validate_job(job: dict) -> None:
    formal = job.get("formal_seed") is True
    seeds = capacity.FORMAL_SEEDS if formal else capacity.SMOKE_SEEDS
    conditions = capacity.CONDITIONS if formal else capacity.SMOKE_CONDITIONS
    if type(job.get("seed")) is not int or job["seed"] not in seeds:
        raise RuntimeError("capacity job seed does not match registration")
    if job.get("condition") not in conditions:
        raise RuntimeError("capacity job condition does not match registration")
    if type(job.get("formal_seed")) is not bool:
        raise RuntimeError("capacity job formal_seed must be boolean")


def _contains_reserved_metadata(value) -> bool:
    if isinstance(value, dict):
        return any(
            (_contains_reserved_seed(item) if key in _SEED_METADATA_KEYS else False)
            or _contains_reserved_metadata(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_reserved_metadata(item) for item in value)
    return False


def _contains_reserved_seed(value) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_reserved_seed(key) or _contains_reserved_seed(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_reserved_seed(item) for item in value)
    if type(value) is int:
        return value in capacity.FORMAL_SEEDS
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
        return int(value) in capacity.FORMAL_SEEDS
    return False


def _claims_formal_experiment(value) -> bool:
    if isinstance(value, dict):
        return any(
            item == capacity.EXPERIMENT_NAME or _claims_formal_experiment(item)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_claims_formal_experiment(item) for item in value)
    return False


def _looks_like_formal_artifact(path: Path) -> bool:
    return (
        "capacity_curve" in path.parts
        or path.name == f"{capacity.EXPERIMENT_NAME}.json"
        or any(re.search(rf"(?<!\d){seed}(?!\d)", path.name) for seed in capacity.FORMAL_SEEDS)
    )


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
