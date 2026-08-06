"""Run the registered K-sensitivity sweep."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

from experiment.commands.run_full_selection_experiment import (
    DEFAULT_CONFIG_PATH,
    RANDOM_DRAWS,
    _arm_definitions,
    _format_duration,
    _progress,
    _registration as _primary_registration,
    run_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "full_selection_experiment_k_sweep"
SMOKE_EXPERIMENT_NAME = "full_selection_experiment_k_sweep_smoke"
FORMAL_SEEDS = tuple(range(6511, 6541))
SMOKE_SEEDS = (6460,)
K_VALUES = (2, 5, 10)
CONDITIONS = (
    "reversed_a0",
    "reversed_a05",
    "reversed_a1",
    "permuted_a0",
    "permuted_a05",
    "permuted_a1",
)
SMOKE_CONDITIONS = ("reversed_a0",)
VALIDATION_TASK_COUNT = 25
OUTPUT_ROOT = Path("experiment/data/selection/k_sweep")
OUTPUT_PATH = OUTPUT_ROOT / f"{EXPERIMENT_NAME}.json"
SMOKE_OUTPUT_ROOT = Path("experiment/data/selection/k_sweep_smoke")
SMOKE_OUTPUT_PATH = SMOKE_OUTPUT_ROOT / f"{SMOKE_EXPERIMENT_NAME}.json"


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


def _run_experiment(*, workers: int, smoke: bool) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    root = _repo_path(SMOKE_OUTPUT_ROOT if smoke else OUTPUT_ROOT)
    if smoke:
        root.mkdir(parents=True, exist_ok=True)
    else:
        claim_formal_output(root)
    registration = _registration(smoke=smoke)
    write_json_atomic(root / "registration.json", registration)
    jobs = build_jobs(root / "cells", workers=workers, smoke=smoke)
    output_path = _repo_path(SMOKE_OUTPUT_PATH if smoke else OUTPUT_PATH)
    started = time.perf_counter()
    _progress(
        f"K SWEEP{' SMOKE' if smoke else ''} START cells={len(jobs)} "
        f"workers={workers} output={output_path}"
    )
    cells = execute_jobs(jobs, workers=workers)
    _progress(
        f"AGGREGATE START cells={len(cells)} "
        f"elapsed={_format_duration(time.perf_counter() - started)}"
    )
    payload = {
        "experiment_name": SMOKE_EXPERIMENT_NAME if smoke else EXPERIMENT_NAME,
        "smoke": smoke,
        "registration": registration,
        "cells": sorted(cells, key=lambda cell: (cell["k"], cell["seed"], cell["condition"])),
    }
    write_json_atomic(output_path, payload)
    return payload


def _registration(*, smoke: bool) -> dict:
    seeds = SMOKE_SEEDS if smoke else FORMAL_SEEDS
    conditions = SMOKE_CONDITIONS if smoke else CONDITIONS
    random_draws = 1 if smoke else RANDOM_DRAWS
    experiment_name = SMOKE_EXPERIMENT_NAME if smoke else EXPERIMENT_NAME
    registration = _primary_registration(
        seeds,
        conditions,
        K_VALUES[-1],
        random_draws,
        experiment_name,
    )
    registration.pop("k")
    registration["arms"]["random_k"] = (
        f"{random_draws} deterministic random K "
        f"{'draw' if random_draws == 1 else 'draws'} from C"
    )
    registration.update(
        {
            "k_values": list(K_VALUES),
            "cell_count": len(K_VALUES) * len(seeds) * len(conditions),
            "validation_task_count": VALIDATION_TASK_COUNT,
            "analysis_status": "raw_data_only",
            "k_sensitivity_status": "registered_raw_data_generation",
            "stale_reversed": "excluded from the registered K sweep",
        }
    )
    return registration


def build_jobs(cell_root: Path, *, workers: int, smoke: bool = False) -> list[dict]:
    seeds = SMOKE_SEEDS if smoke else FORMAL_SEEDS
    conditions = SMOKE_CONDITIONS if smoke else CONDITIONS
    random_draws = 1 if smoke else RANDOM_DRAWS
    experiment_name = SMOKE_EXPERIMENT_NAME if smoke else EXPERIMENT_NAME
    jobs = [
        {
            "experiment_name": experiment_name,
            "seed": seed,
            "condition": condition,
            "k": k,
            "random_draws": random_draws,
            "formal_seed": not smoke,
            "selector_workers": 1 if workers > 1 else None,
            "cell_path": str(cell_root / f"k{k}" / f"{seed}_{condition}.json"),
        }
        for k in K_VALUES
        for seed in seeds
        for condition in conditions
    ]
    for index, job in enumerate(jobs, start=1):
        job.update(index=index, total=len(jobs))
    return jobs


def validate_cell(cell: dict, job: dict) -> None:
    def fail(detail: str) -> None:
        raise RuntimeError(f"K-sweep cell invariant failed: {detail}")

    for field in (
        "experiment_name",
        "seed",
        "condition",
        "formal_seed",
        "k",
        "random_draws",
    ):
        if cell.get(field) != job[field]:
            fail(f"{field}={cell.get(field)!r}, expected {job[field]!r}")

    arms = cell.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(_arm_definitions()):
        fail("arm set does not match registration")

    def selected_count(row) -> int | None:
        selected = row.get("selected_programs") if isinstance(row, dict) else None
        return len(selected) if isinstance(selected, list) else None

    if selected_count(arms["primitives_only"]) != 0:
        fail("primitives_only must select zero programs")
    if "diagnostic_candidate_count" in arms["hidden_motif_oracle"]:
        fail("hidden-motif capacity metadata is not part of the registered sweep")
    for name, row in arms.items():
        if name not in {"primitives_only", "random_k"} and selected_count(row) != job["k"]:
            fail(f"{name} must select exactly K={job['k']} programs")

    draws = arms["random_k"].get("draws")
    if not isinstance(draws, list) or len(draws) != job["random_draws"]:
        fail(f"random_k must contain exactly {job['random_draws']} draws")
    if any(selected_count(row) != job["k"] for row in draws):
        fail(f"every random draw must select exactly K={job['k']} programs")


def run_job(job: dict) -> tuple[dict, float]:
    started = time.perf_counter()
    _progress(
        f"CELL START {job['index']:03d}/{job['total']:03d} "
        f"k={job['k']} seed={job['seed']} condition={job['condition']}"
    )
    cell = run_cell(
        config_path=DEFAULT_CONFIG_PATH,
        seed=job["seed"],
        condition_name=job["condition"],
        k=job["k"],
        random_draws=job["random_draws"],
        selector_workers=job["selector_workers"],
        experiment_name=job["experiment_name"],
    )
    cell["formal_seed"] = job["formal_seed"]
    validate_cell(cell, job)
    write_json_atomic(Path(job["cell_path"]), cell)
    return cell, time.perf_counter() - started


def execute_jobs(jobs: list[dict], *, workers: int) -> list[dict]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    started = time.perf_counter()

    def collect(results) -> list[dict]:
        cells = []
        total = len(jobs)
        for index, (cell, elapsed) in enumerate(results, start=1):
            cells.append(cell)
            total_elapsed = time.perf_counter() - started
            eta = total_elapsed / index * (total - index)
            _progress(
                f"CELL DONE {index:03d}/{total:03d} "
                f"k={cell['k']} seed={cell['seed']} condition={cell['condition']} "
                f"elapsed={_format_duration(elapsed)} "
                f"total={_format_duration(total_elapsed)} eta={_format_duration(eta)}"
            )
        return cells

    if workers == 1:
        return collect(map(run_job, jobs))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return collect(executor.map(run_job, jobs))


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def claim_formal_output(root: Path) -> None:
    _fresh_look_guard(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"formal K-sweep output already claimed: {root}") from exc


def _fresh_look_guard(root: Path) -> None:
    selection_root = root.parent
    if not selection_root.exists():
        return
    for path in selection_root.rglob("*"):
        if path.is_dir():
            continue
        formal_like = _looks_like_sweep_artifact(path)
        if path.suffix != ".json":
            if formal_like:
                raise RuntimeError(f"fresh-look guard blocked K-sweep artifact: {path}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if formal_like:
                raise RuntimeError(f"fresh-look guard blocked malformed K-sweep artifact: {path}")
            continue
        if not isinstance(data, dict):
            if formal_like:
                raise RuntimeError(f"fresh-look guard blocked malformed K-sweep artifact: {path}")
            continue
        registration = data.get("registration", {})
        if not isinstance(registration, dict):
            registration = {}
        if (
            data.get("experiment_name") in {EXPERIMENT_NAME}
            or registration.get("experiment_name") == EXPERIMENT_NAME
            or _contains_sweep_seed(data.get("seed"))
            or _contains_sweep_seed(registration.get("seeds"))
            or formal_like
        ):
            raise RuntimeError(f"fresh-look guard blocked prior K-sweep artifact: {path}")


def _looks_like_sweep_artifact(path: Path) -> bool:
    return (
        "k_sweep" in path.parts
        or path.name in {
            f"{EXPERIMENT_NAME}.json",
            f".{EXPERIMENT_NAME}.json.tmp",
        }
        or any(str(seed) in path.name for seed in FORMAL_SEEDS)
    )


def _contains_sweep_seed(value) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_contains_sweep_seed(item) for item in value)
    if type(value) in {int, float}:
        return value in FORMAL_SEEDS
    if isinstance(value, str):
        try:
            return int(value.strip()) in FORMAL_SEEDS
        except ValueError:
            return False
    return False


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
