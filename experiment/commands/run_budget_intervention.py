"""Run the registered fixed-library evaluation-budget intervention."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

from experiment import budget_intervention as intervention
from experiment import capacity_curve as capacity
from experiment.commands.run_capacity_curve_experiment import shared_world_hashes
from experiment.commands.run_full_selection_experiment import _format_duration, _progress
from experiment.commands.run_k_sweep_experiment import write_json_atomic
from experiment.frontier_promotion import frontier_promotion_menu, menu_diagnostics
from experiment.generator import DEFAULT_CONFIG_PATH, load_config, make_world
from experiment.selection import candidates_to_library
from experiment.solver import primitive_library


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION_ROOT = Path("experiment/data/selection")
OUTPUT_ROOT = SELECTION_ROOT / "budget_intervention"
OUTPUT_PATH = OUTPUT_ROOT / f"{intervention.EXPERIMENT_NAME}.json"
SMOKE_OUTPUT_ROOT = SELECTION_ROOT / "budget_intervention_smoke"
SMOKE_OUTPUT_PATH = (
    SMOKE_OUTPUT_ROOT / f"{intervention.SMOKE_EXPERIMENT_NAME}.json"
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    payload = run_experiment(workers=args.workers, smoke=args.smoke)
    _progress(
        f"{payload['experiment_name']}: cells={len(payload['cells'])} "
        f"smoke={payload['smoke']}"
    )


def run_experiment(*, workers: int, smoke: bool) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    source_file = _repo_path(intervention.source_path(smoke=smoke))
    _require_source_hash(source_file, smoke=smoke)
    source = json.loads(source_file.read_text(encoding="utf-8"))
    capacity.validate_aggregate(source, formal=not smoke)

    root = _repo_path(SMOKE_OUTPUT_ROOT if smoke else OUTPUT_ROOT)
    if smoke:
        root.mkdir(parents=True, exist_ok=True)
    else:
        claim_formal_output(root)
    write_json_atomic(root / "registration.json", intervention.registration(smoke=smoke))

    jobs = build_jobs(source, root / "cells", workers=workers, smoke=smoke)
    started = time.perf_counter()
    _progress(
        f"BUDGET INTERVENTION{' SMOKE' if smoke else ''} START "
        f"cells={len(jobs)} workers={workers} output={root}"
    )
    cells, source_anchors = _execute_jobs(jobs, workers=workers)
    _require_source_hash(source_file, smoke=smoke)
    payload = {
        "experiment_name": (
            intervention.SMOKE_EXPERIMENT_NAME
            if smoke
            else intervention.EXPERIMENT_NAME
        ),
        "smoke": smoke,
        "registration": intervention.registration(smoke=smoke),
        "cells": sorted(
            cells,
            key=lambda cell: (
                cell["seed"],
                capacity.CONDITIONS.index(cell["condition"]),
            ),
        ),
    }
    intervention.validate_aggregate(
        payload,
        formal=not smoke,
        source_anchors=source_anchors,
    )
    output = _repo_path(SMOKE_OUTPUT_PATH if smoke else OUTPUT_PATH)
    write_json_atomic(output, payload)
    elapsed = time.perf_counter() - started
    if smoke:
        _progress(
            f"SMOKE TIMING total={_format_duration(elapsed)} "
            f"projected_serial={_format_duration(elapsed * 180)}"
        )
    return payload


def build_jobs(source: dict, cell_root: Path, *, workers: int, smoke: bool) -> list[dict]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    cells = source["cells"]
    jobs = [
        {
            "smoke": smoke,
            "source": _source_anchor(cell),
            "cell_path": str(cell_root / f"{cell['seed']}_{cell['condition']}.json"),
        }
        for cell in cells
    ]
    for index, job in enumerate(jobs, start=1):
        job.update(index=index, total=len(jobs))
    expected = 1 if smoke else 180
    if len(jobs) != expected:
        raise RuntimeError(f"registered intervention requires exactly {expected} jobs")
    return jobs


def _source_anchor(cell: dict) -> dict:
    arms = {}
    for name in intervention.INCLUDED_ARMS:
        source_arm = (
            cell["arms"][name]["draws"][intervention.RANDOM_DRAW]
            if name == capacity.RANDOM_ARM
            else cell["arms"][name]
        )
        arms[name] = {
            "selected_programs": source_arm["selected_programs"][:2],
            "prefixes": [
                {
                    "k": k,
                    "validation_summary": source_arm["prefixes"][k][
                        "validation_summary"
                    ],
                    "test_summary": source_arm["prefixes"][k]["test_summary"],
                }
                for k in (1, 2)
            ],
        }
    return {
        "seed": cell["seed"],
        "condition": cell["condition"],
        "source_cell_hash": capacity.canonical_hash(cell),
        "world_metadata": cell["world_metadata"],
        "shared_hashes": cell["shared_hashes"],
        "menu": cell["menu"],
        "candidate_menu_programs": cell["candidate_menu_programs"],
        "task_counts": {
            "starter": cell["starter_task_count"],
            "validation": cell["validation_task_count"],
            "test": cell["test_task_count"],
        },
        "primitive": cell["primitive"],
        "arms": arms,
    }


def _execute_jobs(jobs: list[dict], *, workers: int) -> tuple[list[dict], list[dict]]:
    started = time.perf_counter()

    def collect(results) -> tuple[list[dict], list[dict]]:
        cells = []
        anchors = []
        for count, (cell, anchor, elapsed) in enumerate(results, start=1):
            cells.append(cell)
            anchors.append(anchor)
            total_elapsed = time.perf_counter() - started
            eta = total_elapsed / count * (len(jobs) - count)
            _progress(
                f"CELL DONE {count:03d}/{len(jobs):03d} "
                f"seed={cell['seed']} condition={cell['condition']} "
                f"elapsed={_format_duration(elapsed)} "
                f"total={_format_duration(total_elapsed)} eta={_format_duration(eta)}"
            )
        return cells, anchors

    if workers == 1:
        return collect(map(_run_job, jobs))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return collect(executor.map(_run_job, jobs))


def _run_job(job: dict) -> tuple[dict, dict, float]:
    _validate_job(job)
    started = time.perf_counter()
    source = job["source"]
    _progress(
        f"CELL START {job['index']:03d}/{job['total']:03d} "
        f"seed={source['seed']} condition={source['condition']}"
    )
    config = load_config(str(_repo_path(DEFAULT_CONFIG_PATH)))
    conditions = {condition.name: condition for condition in config.conditions}
    world = make_world(config, source["seed"], conditions[source["condition"]])
    menu = frontier_promotion_menu(world.tasks_start)
    _validate_regenerated_source(world, menu, source)

    task_hashes = {
        "validation": _task_hash_rows(world.tasks_val),
        "test": _task_hash_rows(world.tasks_test),
    }
    source_anchor = {**source, "task_hashes": task_hashes}
    primitive = intervention.evaluate_library(
        primitive_library(),
        k=0,
        selected_programs=(),
        validation_tasks=world.tasks_val,
        test_tasks=world.tasks_test,
    )
    _require_30k_parity(primitive, source["primitive"], "primitives", 0)

    by_program = {candidate.program_string: candidate for candidate in menu.candidates}
    arms = {}
    for name in intervention.INCLUDED_ARMS:
        selected_programs = source["arms"][name]["selected_programs"]
        selected = tuple(by_program[program] for program in selected_programs)
        prefixes = []
        for k in (1, 2):
            evaluation = intervention.evaluate_library(
                candidates_to_library(selected[:k]),
                k=k,
                selected_programs=selected_programs[:k],
                validation_tasks=world.tasks_val,
                test_tasks=world.tasks_test,
            )
            _require_30k_parity(
                evaluation,
                source["arms"][name]["prefixes"][k - 1],
                name,
                k,
            )
            prefixes.append(evaluation)
        arms[name] = {
            "selected_programs": selected_programs,
            "prefixes": prefixes,
        }

    smoke = job["smoke"]
    cell = {
        "experiment_name": (
            intervention.SMOKE_EXPERIMENT_NAME
            if smoke
            else intervention.EXPERIMENT_NAME
        ),
        "smoke": smoke,
        "seed": source["seed"],
        "condition": source["condition"],
        "source": intervention.registration(smoke=smoke)["source"],
        "source_cell_hash": source["source_cell_hash"],
        "world_metadata": source["world_metadata"],
        "shared_hashes": source["shared_hashes"],
        "candidate_menu_programs": source["candidate_menu_programs"],
        "task_hashes": task_hashes,
        "primitive": primitive,
        "arms": arms,
        "mechanism": {
            name: intervention.mechanism_rows(
                arms[name]["prefixes"][0], arms[name]["prefixes"][1]
            )
            for name in intervention.PRIMARY_ARMS
        },
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
    }
    intervention.validate_cell(
        cell,
        formal=not smoke,
        source_anchor=source_anchor,
    )
    write_json_atomic(Path(job["cell_path"]), cell)
    return cell, source_anchor, time.perf_counter() - started


def _validate_regenerated_source(world, menu, source: dict) -> None:
    hashes, programs = shared_world_hashes(world, menu=menu)
    diagnostics = menu_diagnostics(menu)
    diagnostics["cap"] = menu.cap
    metadata = {
        "condition": {
            "name": world.condition.name,
            "alt_kind": world.condition.alt_kind,
            "alpha_val": world.condition.alpha_val,
            "alpha_test": world.condition.alpha_test,
        },
        "realized_rho": world.metadata["realized_rho"],
        "density_summary": world.metadata["density_summary"],
        "expected_motif_length": world.metadata["expected_motif_length"],
    }
    actual_counts = {
        "starter": len(world.tasks_start),
        "validation": len(world.tasks_val),
        "test": len(world.tasks_test),
    }
    checks = {
        "world metadata": metadata == source["world_metadata"],
        "shared hashes": hashes == source["shared_hashes"],
        "task counts": actual_counts == source["task_counts"],
        "candidate-menu order": programs == source["candidate_menu_programs"],
        "candidate-menu diagnostics": diagnostics == source["menu"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "regenerated source differs from authoritative capacity cell: "
            + ", ".join(failed)
        )
    selected = {
        program
        for arm in source["arms"].values()
        for program in arm["selected_programs"]
    }
    if not selected <= set(programs):
        raise RuntimeError("stored selected prefix is absent from regenerated menu")


def _require_30k_parity(evaluation: dict, source: dict, arm: str, k: int) -> None:
    row = evaluation["budgets"][0]
    checks = {
        "validation summary": row["validation_summary"]
        == source["validation_summary"],
        "test summary": row["test_summary"] == source["test_summary"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"30,000-candidate parity failed for {arm} K={k}: "
            + ", ".join(failed)
        )


def _task_hash_rows(tasks) -> list[dict]:
    return [
        {"task_id": task.id, "target_hash": intervention.target_hash(task)}
        for task in sorted(tasks, key=lambda item: item.id)
    ]


def _validate_job(job: dict) -> None:
    if not isinstance(job, dict) or set(job) != {
        "smoke",
        "source",
        "cell_path",
        "index",
        "total",
    }:
        raise RuntimeError("intervention job fields do not match schema")
    smoke = job["smoke"]
    if type(smoke) is not bool:
        raise RuntimeError("intervention job smoke flag must be boolean")
    source = job["source"]
    seeds = capacity.SMOKE_SEEDS if smoke else capacity.FORMAL_SEEDS
    conditions = capacity.SMOKE_CONDITIONS if smoke else capacity.CONDITIONS
    if type(source.get("seed")) is not int or source["seed"] not in seeds:
        raise RuntimeError("intervention job seed does not match registration")
    if source.get("condition") not in conditions:
        raise RuntimeError("intervention job condition does not match registration")


def claim_formal_output(root: Path) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"formal budget-intervention output already claimed: {root}"
        ) from exc


def _require_source_hash(path: Path, *, smoke: bool) -> None:
    if not path.is_file():
        raise RuntimeError(f"registered source is missing: {path}")
    expected = intervention.source_sha256(smoke=smoke)
    actual = intervention.file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"registered source SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
