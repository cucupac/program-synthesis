"""Frozen schema and threshold logic for the fixed-library budget intervention."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
from pathlib import Path
import re

from experiment import capacity_curve as capacity
from experiment.generator import Task
from experiment.solver import (
    FrontierIndex,
    LibraryItem,
    SolveConfig,
    build_frontier_index,
)


EXPERIMENT_NAME = "full_selection_experiment_budget_intervention"
SMOKE_EXPERIMENT_NAME = "full_selection_experiment_budget_intervention_smoke"
FORMAL_SOURCE = (
    "experiment/data/selection/capacity_curve/"
    "full_selection_experiment_capacity_curve.json"
)
FORMAL_SOURCE_SHA256 = (
    "a70e2156897da8473e2891d2f2e9daaf78142400fbda696f479bae985fa62f3d"
)
SMOKE_SOURCE = (
    "experiment/data/selection/capacity_curve_smoke/"
    "full_selection_experiment_capacity_curve_smoke.json"
)
SMOKE_SOURCE_SHA256 = (
    "2a26b7026b3fab20830fdd5488f5b7b8fb8ecb0e4aedf59e1d2abc30f900fcc1"
)
CALIBRATION_SOURCE = (
    "experiment/data/selection/k_sweep/full_selection_experiment_k_sweep.json"
)
CALIBRATION_SOURCE_SHA256 = (
    "1454be9e733ff655bb9761465bab5d4166a2cfe1a9bb7b233442bd16fa7b787e"
)
BUDGETS = (30_000, 45_000, 60_000, 90_000)
MAX_BUDGET = BUDGETS[-1]
K_VALUES = (0, 1, 2)
MAX_PROGRAM_SIZE = 7
SIZE_COUNT = MAX_PROGRAM_SIZE + 1
RANDOM_DRAW = 0
INCLUDED_ARMS = (
    capacity.COMPRESSION_ALL_ARM,
    capacity.COMPRESSION_VALIDATION_ARM,
    capacity.UTILITY_ARM,
    capacity.TEST_PEEK_ARM,
    capacity.RANDOM_ARM,
)
PRIMARY_ARMS = (capacity.COMPRESSION_ALL_ARM, capacity.UTILITY_ARM)


def source_path(*, smoke: bool) -> str:
    return SMOKE_SOURCE if smoke else FORMAL_SOURCE


def source_sha256(*, smoke: bool) -> str:
    return SMOKE_SOURCE_SHA256 if smoke else FORMAL_SOURCE_SHA256


def registration(*, smoke: bool) -> dict:
    seeds = capacity.SMOKE_SEEDS if smoke else capacity.FORMAL_SEEDS
    conditions = capacity.SMOKE_CONDITIONS if smoke else capacity.CONDITIONS
    return {
        "experiment_name": SMOKE_EXPERIMENT_NAME if smoke else EXPERIMENT_NAME,
        "registration": "R14",
        "registration_timing": "before_formal_execution",
        "follow_up_status": "registered_mechanism_follow_up_on_inspected_worlds",
        "independent_confirmation": False,
        "smoke": smoke,
        "source": {
            "path": source_path(smoke=smoke),
            "sha256": source_sha256(smoke=smoke),
            "experiment_name": (
                capacity.SMOKE_EXPERIMENT_NAME
                if smoke
                else capacity.EXPERIMENT_NAME
            ),
        },
        "calibration": {
            "path": CALIBRATION_SOURCE,
            "sha256": CALIBRATION_SOURCE_SHA256,
            "seeds": list(range(6511, 6521)),
            "condition": "reversed_a0",
            "timing": "before_inspecting_above_30000_outcomes_for_formal_seeds",
            "observed": {
                capacity.COMPRESSION_ALL_ARM: {
                    "size4_reach": [0, 10, 10, 10],
                    "mean_k2_minus_k1": [-9.2, 1.1, 3.2, 3.4],
                },
                capacity.COMPRESSION_VALIDATION_ARM: {
                    "size4_reach": [1, 10, 10, 10],
                    "mean_k2_minus_k1": [-8.5, 0.2, 1.0, 1.8],
                },
                capacity.UTILITY_ARM: {
                    "size4_reach": [1, 10, 10, 10],
                    "mean_k2_minus_k1": [-5.8, -1.8, 1.3, 1.5],
                },
                capacity.TEST_PEEK_ARM: {
                    "size4_reach": [9, 10, 10, 10],
                    "mean_k2_minus_k1": [-0.6, -0.3, 0.8, 0.9],
                },
            },
        },
        "seeds": list(seeds),
        "conditions": list(conditions),
        "cell_count": len(seeds) * len(conditions),
        "k_values": list(K_VALUES),
        "budgets": list(BUDGETS),
        "maximum_frontier_budget": MAX_BUDGET,
        "max_program_size": MAX_PROGRAM_SIZE,
        "evaluation_order": "existing_deterministic_bottom_up_order",
        "helper_order": "stored_selected_order_named_C0000_then_C0001",
        "atomic_helper_search_size": 0,
        "lower_budget_rule": "retain_first_hits_at_or_below_budget_from_90000_prefix",
        "candidate_counts_by_size": "all_attempts_including_duplicate_outputs",
        "library_source": "stored_exact_k1_and_k2_prefixes_without_reselection",
        "arms": ["primitives_only", *INCLUDED_ARMS],
        "random_draw": RANDOM_DRAW,
        "starter_task_count": 100,
        "validation_task_count": capacity.VALIDATION_TASK_COUNT,
        "test_task_count": capacity.TEST_TASK_COUNT,
        "primary_methods": list(PRIMARY_ARMS),
        "primary_estimands": {
            "J_m(k,B)": "test_problems_solved_out_of_100",
            "D_m(B)": "J_m(2,B)-J_m(1,B)",
            "I_m": "D_m(90000)-D_m(30000)",
            "remaining_gap_at_90000": (
                "D_m(90000)_reported_to_distinguish_attenuation_from_elimination"
            ),
            "cell_contrasts": "average_six_conditions_within_seed",
            "seed_estimate": "mean_across_30_seeds",
        },
        "inference": {
            "family": list(PRIMARY_ARMS),
            "method": "single_step_max_t",
            "bootstrap_statistic": (
                "maximum_absolute_centered_studentized_deviation"
            ),
            "seed": 20260714,
            "draws": 10_000,
            "cluster": "seed",
            "conditions_per_seed": 6,
            "standard_error": "sample_sd_of_30_seed_values_divided_by_sqrt_30",
            "critical_value": "zero_based_sorted_element_9499",
            "interval": "estimate_plus_or_minus_critical_value_times_standard_error",
            "zero_standard_error": "point_interval_omitted_from_maximum",
            "p_values": False,
        },
        "mechanism_checks": {
            "lost_set": "solved_at_k1_b30000_and_unsolved_at_k2_b30000",
            "budgets": list(BUDGETS),
            "abstract_search_sizes": list(range(SIZE_COUNT)),
            "descriptive_only": True,
            "formal_mediation_claim": False,
        },
        "fixed_test_peeking_path": "selected_using_the_30000_budget_test_set",
        "no_reselection": True,
        "no_early_stopping": True,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_hash(task: Task) -> str:
    return capacity.canonical_hash(sorted([list(cell) for cell in task.target]))


def evaluate_library(
    library: Sequence[LibraryItem],
    *,
    k: int,
    selected_programs: Sequence[str],
    validation_tasks: Sequence[Task],
    test_tasks: Sequence[Task],
) -> dict:
    if k not in K_VALUES or len(selected_programs) != k:
        raise ValueError("library prefix does not match registered K")
    index = build_frontier_index(
        library,
        SolveConfig(
            node_budget=MAX_BUDGET,
            max_program_size=MAX_PROGRAM_SIZE,
            max_solutions=1,
        ),
    )
    validation_targets = _target_rows(validation_tasks, index)
    test_targets = _target_rows(test_tasks, index)
    return {
        "k": k,
        "selected_programs": list(selected_programs),
        "max_frontier": _max_frontier(index),
        "budgets": [
            {
                "budget": budget,
                "candidates_tried_by_size": list(
                    candidates_tried_by_size(index, budget)
                ),
                "validation_summary": _summary(validation_targets, index, budget),
                "test_summary": _summary(test_targets, index, budget),
            }
            for budget in BUDGETS
        ],
        "validation_targets": validation_targets,
        "test_targets": test_targets,
    }


def candidates_tried_by_size(index: FrontierIndex, budget: int) -> tuple[int, ...]:
    if budget not in BUDGETS:
        raise ValueError("budget is not registered")
    return _threshold_counts(
        index.candidates_tried_by_size,
        min(budget, index.candidates_tried_total),
    )


def mechanism_rows(k1: dict, k2: dict) -> list[dict]:
    rows1 = {row["task_id"]: row for row in k1["test_targets"]}
    rows2 = {row["task_id"]: row for row in k2["test_targets"]}
    lost = {
        task_id
        for task_id, row in rows1.items()
        if _solved_by(row, BUDGETS[0]) and not _solved_by(rows2[task_id], BUDGETS[0])
    }
    return [
        _mechanism_row(budget, lost, rows2)
        for budget in BUDGETS
    ]


def _mechanism_row(budget: int, lost: set[str], k2_rows: dict[str, dict]) -> dict:
    recovered = [
        k2_rows[task_id]
        for task_id in sorted(lost)
        if _solved_by(k2_rows[task_id], budget)
    ]
    by_size = [0] * SIZE_COUNT
    for row in recovered:
        by_size[row["abstract_search_size"]] += 1
    return {
        "budget": budget,
        "lost_count": len(lost),
        "lost_test_rate": len(lost) / capacity.TEST_TASK_COUNT,
        "recovered_count": len(recovered),
        "recovered_test_rate": len(recovered) / capacity.TEST_TASK_COUNT,
        "recovered_by_abstract_search_size": by_size,
    }


def _max_frontier(index: FrontierIndex) -> dict:
    first_size4_rank = (
        sum(index.candidates_tried_by_size[:4]) + 1
        if len(index.candidates_tried_by_size) > 4
        and index.candidates_tried_by_size[4] > 0
        else None
    )
    return {
        "candidates_tried_total": index.candidates_tried_total,
        "candidates_tried_by_size": list(index.candidates_tried_by_size),
        "hit_budget": index.hit_budget,
        "unique_outputs": index.unique_outputs,
        "first_size4_rank": first_size4_rank,
    }


def _target_rows(tasks: Sequence[Task], index: FrontierIndex) -> list[dict]:
    rows = []
    for task in sorted(tasks, key=lambda item: item.id):
        entry = index.entries.get(task.target)
        rows.append(
            {
                "task_id": task.id,
                "target_hash": target_hash(task),
                "first_hit_rank": (
                    entry.candidates_tried_at_first_solution if entry else None
                ),
                "abstract_search_size": entry.abstract_search_size if entry else None,
            }
        )
    return rows


def _summary(rows: Sequence[dict], index: FrontierIndex, budget: int) -> dict:
    total = min(budget, index.candidates_tried_total)
    solved_rows = [row for row in rows if _solved_by(row, budget)]
    first_costs = [row["first_hit_rank"] for row in solved_rows]
    costs = [
        row["first_hit_rank"] if _solved_by(row, budget) else total
        for row in rows
    ]
    solved = len(solved_rows)
    return {
        "solved_count": solved,
        "failure_count": len(rows) - solved,
        "task_count": len(rows),
        "solve_rate": solved / len(rows),
        "mean_search_cost": sum(costs) / len(costs),
        "mean_first_solution_cost": (
            sum(first_costs) / solved if solved else None
        ),
        "frontier_candidates_tried_total": total,
        "hit_budget": (
            budget < index.candidates_tried_total
            or (budget == index.candidates_tried_total and index.hit_budget)
        ),
        "unique_outputs": sum(
            entry.candidates_tried_at_first_solution <= budget
            for entry in index.entries.values()
        ),
    }


def _solved_by(row: dict, budget: int) -> bool:
    rank = row["first_hit_rank"]
    return rank is not None and rank <= budget


def validate_cell(cell: dict, *, formal: bool, source_anchor: dict) -> None:
    expected_name = EXPERIMENT_NAME if formal else SMOKE_EXPERIMENT_NAME
    expected_seeds = capacity.FORMAL_SEEDS if formal else capacity.SMOKE_SEEDS
    expected_conditions = capacity.CONDITIONS if formal else capacity.SMOKE_CONDITIONS
    required = {
        "experiment_name",
        "smoke",
        "seed",
        "condition",
        "source",
        "source_cell_hash",
        "world_metadata",
        "shared_hashes",
        "candidate_menu_programs",
        "task_hashes",
        "primitive",
        "arms",
        "mechanism",
        "wall_clock_seconds",
    }
    if not isinstance(cell, dict) or set(cell) != required:
        _fail("cell fields do not match schema")
    if cell["experiment_name"] != expected_name or cell["smoke"] is not (not formal):
        _fail("cell experiment provenance does not match run")
    if type(cell["seed"]) is not int or cell["seed"] not in expected_seeds:
        _fail("cell seed does not match registration")
    if cell["condition"] not in expected_conditions:
        _fail("cell condition does not match registration")
    if cell["source"] != registration(smoke=not formal)["source"]:
        _fail("cell source does not match registration")
    _validate_sha256(cell["source_cell_hash"], "source cell hash")
    _validate_world_metadata(cell["world_metadata"], cell["condition"])
    if (
        not isinstance(cell["shared_hashes"], dict)
        or set(cell["shared_hashes"]) != set(capacity.HASH_KEYS)
    ):
        _fail("shared hashes must be an object")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in cell["shared_hashes"].values()
    ):
        _fail("shared hashes must be lowercase SHA-256 values")
    menu = cell["candidate_menu_programs"]
    if (
        not isinstance(menu, list)
        or not capacity.K_MAX <= len(menu) <= capacity.MENU_CAP
        or len(menu) != len(set(menu))
        or not all(isinstance(program, str) for program in menu)
    ):
        _fail("candidate menu must contain unique program strings")
    if cell["shared_hashes"]["candidate_menu"] != capacity.canonical_hash(menu):
        _fail("candidate-menu hash does not match ordered programs")
    _validate_task_hashes(cell["task_hashes"])
    _validate_evaluation(cell["primitive"], expected_k=0)
    _require_task_rows(cell["primitive"], cell["task_hashes"])
    arms = cell["arms"]
    if not isinstance(arms, dict) or set(arms) != set(INCLUDED_ARMS):
        _fail("included arm set does not match registration")
    for name, arm in arms.items():
        if not isinstance(arm, dict) or set(arm) != {"selected_programs", "prefixes"}:
            _fail(f"{name} fields do not match schema")
        selected = arm["selected_programs"]
        if (
            not isinstance(selected, list)
            or len(selected) != 2
            or len(set(selected)) != 2
            or not set(selected) <= set(menu)
        ):
            _fail(f"{name} must contain its exact two-program prefix")
        prefixes = arm["prefixes"]
        if not isinstance(prefixes, list) or len(prefixes) != 2:
            _fail(f"{name} must contain K=1 and K=2 evaluations")
        for k, row in enumerate(prefixes, start=1):
            _validate_evaluation(row, expected_k=k)
            _require_task_rows(row, cell["task_hashes"])
            if row["selected_programs"] != selected[:k]:
                _fail(f"{name} evaluation does not match selected prefix")
    _validate_source_binding(cell, source_anchor)
    mechanism = cell["mechanism"]
    if not isinstance(mechanism, dict) or set(mechanism) != set(PRIMARY_ARMS):
        _fail("mechanism checks must cover the two primary methods")
    for name in PRIMARY_ARMS:
        expected = mechanism_rows(arms[name]["prefixes"][0], arms[name]["prefixes"][1])
        if mechanism[name] != expected:
            _fail(f"{name} mechanism rows do not reconstruct from targets")
    if not _finite_nonnegative(cell["wall_clock_seconds"]):
        _fail("cell wall-clock time must be finite and nonnegative")


def validate_aggregate(
    payload: dict, *, formal: bool, source_anchors: Sequence[dict]
) -> None:
    expected_name = EXPERIMENT_NAME if formal else SMOKE_EXPERIMENT_NAME
    expected_seeds = capacity.FORMAL_SEEDS if formal else capacity.SMOKE_SEEDS
    expected_conditions = capacity.CONDITIONS if formal else capacity.SMOKE_CONDITIONS
    if not isinstance(payload, dict) or set(payload) != {
        "experiment_name",
        "smoke",
        "registration",
        "cells",
    }:
        _fail("aggregate fields do not match schema")
    if payload["experiment_name"] != expected_name or payload["smoke"] is not (not formal):
        _fail("aggregate provenance does not match run")
    if payload["registration"] != registration(smoke=not formal):
        _fail("aggregate registration does not match frozen values")
    cells = payload["cells"]
    expected = {
        (seed, condition)
        for seed in expected_seeds
        for condition in expected_conditions
    }
    if not isinstance(cells, list) or len(cells) != len(expected):
        _fail(f"aggregate must contain exactly {len(expected)} cells")
    actual = [(cell.get("seed"), cell.get("condition")) for cell in cells]
    if set(actual) != expected or len(set(actual)) != len(actual):
        _fail("aggregate cells must be unique and complete")
    if actual != sorted(
        actual, key=lambda item: (item[0], capacity.CONDITIONS.index(item[1]))
    ):
        _fail("aggregate cells must use deterministic seed-condition order")
    if not isinstance(source_anchors, (list, tuple)):
        _fail("source anchors must be a sequence")
    anchors = {
        (anchor.get("seed"), anchor.get("condition")): anchor
        for anchor in source_anchors
        if isinstance(anchor, dict)
    }
    if (
        len(source_anchors) != len(expected)
        or set(anchors) != expected
        or len(anchors) != len(source_anchors)
    ):
        _fail("source anchors must be unique and complete")
    for cell in cells:
        validate_cell(
            cell,
            formal=formal,
            source_anchor=anchors[(cell["seed"], cell["condition"])],
        )
    _validate_seed_pairing(cells)


def _validate_evaluation(row: dict, *, expected_k: int) -> None:
    if not isinstance(row, dict) or set(row) != {
        "k",
        "selected_programs",
        "max_frontier",
        "budgets",
        "validation_targets",
        "test_targets",
    }:
        _fail("library evaluation fields do not match schema")
    if row["k"] != expected_k or len(row["selected_programs"]) != expected_k:
        _fail("library evaluation K does not match selected programs")
    frontier = row["max_frontier"]
    _validate_max_frontier(frontier, expected_k=expected_k)
    _validate_target_rows(
        row["validation_targets"], capacity.VALIDATION_TASK_COUNT, frontier
    )
    _validate_target_rows(row["test_targets"], capacity.TEST_TASK_COUNT, frontier)
    budgets = row["budgets"]
    budget_fields = {
        "budget",
        "candidates_tried_by_size",
        "validation_summary",
        "test_summary",
    }
    if (
        not isinstance(budgets, list)
        or len(budgets) != len(BUDGETS)
        or any(not isinstance(item, dict) or set(item) != budget_fields for item in budgets)
        or [item["budget"] for item in budgets] != list(BUDGETS)
    ):
        _fail("budget rows must use the registered order")
    maximum_counts = tuple(frontier["candidates_tried_by_size"])
    unique_outputs = []
    for budget_row in budgets:
        expected_counts = _threshold_counts(
            maximum_counts,
            min(budget_row["budget"], frontier["candidates_tried_total"]),
        )
        if budget_row.get("candidates_tried_by_size") != list(expected_counts):
            _fail("budget size counts do not match maximum frontier prefix")
        _validate_summary(
            budget_row.get("validation_summary"),
            row["validation_targets"],
            frontier,
            budget_row["budget"],
        )
        _validate_summary(
            budget_row.get("test_summary"),
            row["test_targets"],
            frontier,
            budget_row["budget"],
        )
        validation_summary = budget_row["validation_summary"]
        test_summary = budget_row["test_summary"]
        diagnostics = (
            "frontier_candidates_tried_total",
            "hit_budget",
            "unique_outputs",
        )
        if any(validation_summary[key] != test_summary[key] for key in diagnostics):
            _fail("validation and test summaries must share one frontier")
        unique_outputs.append(validation_summary["unique_outputs"])
    if unique_outputs != sorted(unique_outputs):
        _fail("unique-output counts must be nondecreasing with budget")
    if unique_outputs[-1] != frontier["unique_outputs"]:
        _fail("maximum-budget unique outputs must match maximum frontier")


def _validate_max_frontier(frontier: dict, *, expected_k: int) -> None:
    if not isinstance(frontier, dict) or set(frontier) != {
        "candidates_tried_total",
        "candidates_tried_by_size",
        "hit_budget",
        "unique_outputs",
        "first_size4_rank",
    }:
        _fail("maximum frontier fields do not match schema")
    total = frontier["candidates_tried_total"]
    counts = frontier["candidates_tried_by_size"]
    expected_leaves = capacity.PRIMITIVE_LIBRARY_SIZE + expected_k
    if (
        type(total) is not int
        or not 1 <= total <= MAX_BUDGET
        or not isinstance(counts, list)
        or len(counts) != SIZE_COUNT
        or any(type(count) is not int or count < 0 for count in counts)
        or counts[0] != expected_leaves
        or sum(counts) != total
        or type(frontier["hit_budget"]) is not bool
        or (frontier["hit_budget"] and total != MAX_BUDGET)
        or type(frontier["unique_outputs"]) is not int
        or not expected_leaves <= frontier["unique_outputs"] <= total
    ):
        _fail("maximum frontier size-zero or output diagnostics are inconsistent")
    expected_rank = sum(counts[:4]) + 1 if counts[4] else None
    if frontier["first_size4_rank"] != expected_rank:
        _fail("first size-four rank is inconsistent with size counts")


def _validate_target_rows(rows, expected_count: int, frontier: dict) -> None:
    if not isinstance(rows, list) or len(rows) != expected_count:
        _fail(f"target rows must contain exactly {expected_count} tasks")
    ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "task_id",
            "target_hash",
            "first_hit_rank",
            "abstract_search_size",
        }:
            _fail("target row fields do not match schema")
        ids.append(row["task_id"])
        if not isinstance(row["task_id"], str):
            _fail("task IDs must be strings")
        _validate_sha256(row["target_hash"], "target hash")
        rank = row["first_hit_rank"]
        size = row["abstract_search_size"]
        if (rank is None) != (size is None):
            _fail("target rank and size must both be null or both be present")
        if rank is not None and (
            type(rank) is not int
            or not 1 <= rank <= frontier["candidates_tried_total"]
            or type(size) is not int
            or not 0 <= size <= MAX_PROGRAM_SIZE
        ):
            _fail("target first-hit fields are outside registered bounds")
        if rank is not None:
            counts = frontier["candidates_tried_by_size"]
            lower = sum(counts[:size]) + 1
            upper = sum(counts[: size + 1])
            if not lower <= rank <= upper:
                _fail("target first-hit rank contradicts its abstract search size")
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        _fail("target rows must have unique task IDs in sorted order")


def _validate_summary(summary, targets, frontier, budget: int) -> None:
    expected_total = min(budget, frontier["candidates_tried_total"])
    solved_rows = [row for row in targets if _solved_by(row, budget)]
    solved = len(solved_rows)
    first_cost = (
        sum(row["first_hit_rank"] for row in solved_rows) / solved
        if solved
        else None
    )
    mean_cost = sum(
        row["first_hit_rank"] if _solved_by(row, budget) else expected_total
        for row in targets
    ) / len(targets)
    expected = {
        "solved_count": solved,
        "failure_count": len(targets) - solved,
        "task_count": len(targets),
        "solve_rate": solved / len(targets),
        "mean_search_cost": mean_cost,
        "mean_first_solution_cost": first_cost,
        "frontier_candidates_tried_total": expected_total,
        "hit_budget": (
            budget < frontier["candidates_tried_total"]
            or (
                budget == frontier["candidates_tried_total"]
                and frontier["hit_budget"]
            )
        ),
        "unique_outputs": summary.get("unique_outputs") if isinstance(summary, dict) else None,
    }
    if not isinstance(summary, dict) or set(summary) != set(expected):
        _fail("budget summary fields do not match schema")
    if summary != expected:
        _fail("budget summary does not reconstruct from target rows")
    if (
        type(summary["unique_outputs"]) is not int
        or not max(1, solved) <= summary["unique_outputs"] <= expected_total
    ):
        _fail("budget unique-output count is invalid")


def _threshold_counts(counts: Sequence[int], total: int) -> tuple[int, ...]:
    remaining = total
    result = []
    for maximum in counts:
        value = min(maximum, remaining)
        result.append(value)
        remaining -= value
    if remaining:
        _fail("maximum size counts do not cover threshold total")
    return tuple(result)


def _validate_task_hashes(hashes) -> None:
    if not isinstance(hashes, dict) or set(hashes) != {"validation", "test"}:
        _fail("task hashes must cover validation and test")
    for name, rows in hashes.items():
        expected = (
            capacity.VALIDATION_TASK_COUNT if name == "validation" else capacity.TEST_TASK_COUNT
        )
        if not isinstance(rows, list) or len(rows) != expected:
            _fail(f"{name} task hashes have wrong length")
        ids = [row.get("task_id") for row in rows if isinstance(row, dict)]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            _fail(f"{name} task hashes must have unique sorted IDs")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"task_id", "target_hash"}:
                _fail("task hash row fields do not match schema")
            if not isinstance(row["task_id"], str):
                _fail("task hash IDs must be strings")
            _validate_sha256(row["target_hash"], "task hash")


def _require_task_rows(evaluation: dict, hashes: dict) -> None:
    for split in ("validation", "test"):
        actual = [
            {"task_id": row["task_id"], "target_hash": row["target_hash"]}
            for row in evaluation[f"{split}_targets"]
        ]
        if actual != hashes[split]:
            _fail(f"{split} target rows do not match regenerated task hashes")


def _validate_source_binding(cell: dict, anchor: dict) -> None:
    required = {
        "seed",
        "condition",
        "source_cell_hash",
        "world_metadata",
        "shared_hashes",
        "candidate_menu_programs",
        "task_hashes",
        "primitive",
        "arms",
    }
    if not isinstance(anchor, dict) or not required <= set(anchor):
        _fail("source anchor fields do not match schema")
    comparisons = {
        "identity": (cell["seed"], cell["condition"])
        == (anchor["seed"], anchor["condition"]),
        "cell hash": cell["source_cell_hash"] == anchor["source_cell_hash"],
        "world metadata": cell["world_metadata"] == anchor["world_metadata"],
        "shared hashes": cell["shared_hashes"] == anchor["shared_hashes"],
        "ordered menu": cell["candidate_menu_programs"]
        == anchor["candidate_menu_programs"],
        "regenerated task hashes": cell["task_hashes"] == anchor["task_hashes"],
    }
    failed = [name for name, matches in comparisons.items() if not matches]
    if failed:
        _fail("source binding differs: " + ", ".join(failed))
    if set(anchor["arms"]) != set(INCLUDED_ARMS):
        _fail("source anchor arm set does not match registration")
    primitive_30k = cell["primitive"]["budgets"][0]
    if (
        primitive_30k["validation_summary"]
        != anchor["primitive"]["validation_summary"]
        or primitive_30k["test_summary"] != anchor["primitive"]["test_summary"]
    ):
        _fail("primitive 30,000 summary differs from source")
    for name in INCLUDED_ARMS:
        source_arm = anchor["arms"][name]
        arm = cell["arms"][name]
        if arm["selected_programs"] != source_arm["selected_programs"]:
            _fail(f"{name} source selected prefix differs")
        source_prefixes = source_arm.get("prefixes")
        if not isinstance(source_prefixes, list) or len(source_prefixes) != 2:
            _fail(f"{name} source prefix summaries are incomplete")
        for k, (row, source_row) in enumerate(
            zip(arm["prefixes"], source_prefixes), start=1
        ):
            budget_30k = row["budgets"][0]
            if source_row.get("k") != k or any(
                budget_30k[f"{split}_summary"]
                != source_row.get(f"{split}_summary")
                for split in ("validation", "test")
            ):
                _fail(f"{name} K={k} 30,000 summary differs from source")


def _validate_world_metadata(metadata, condition: str) -> None:
    if (
        not isinstance(metadata, dict)
        or set(metadata)
        != {"condition", "realized_rho", "density_summary", "expected_motif_length"}
        or metadata["condition"] != capacity.CONDITION_SPECS[condition]
    ):
        _fail("world metadata does not match the registered condition")
    rho = metadata["realized_rho"]
    if (
        not isinstance(rho, dict)
        or set(rho)
        != {"realized_start_test", "realized_start_val", "realized_val_test"}
        or any(
            type(value) not in {int, float}
            or not math.isfinite(value)
            or not -1 <= value <= 1
            for value in rho.values()
        )
        or not isinstance(metadata["density_summary"], dict)
        or not isinstance(metadata["expected_motif_length"], dict)
    ):
        _fail("world distribution metadata is invalid")


def _validate_seed_pairing(cells: Sequence[dict]) -> None:
    by_seed: dict[int, list[dict]] = {}
    for cell in cells:
        by_seed.setdefault(cell["seed"], []).append(cell)
    for seed_cells in by_seed.values():
        reference = seed_cells[0]
        for cell in seed_cells[1:]:
            if cell["shared_hashes"] != reference["shared_hashes"]:
                _fail("shared world hashes differ within a seed")
            if cell["candidate_menu_programs"] != reference["candidate_menu_programs"]:
                _fail("ordered candidate menu differs within a seed")


def _validate_sha256(value, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail(f"{name} must be a lowercase SHA-256 value")


def _finite_nonnegative(value) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def _fail(message: str) -> None:
    raise ValueError(f"budget-intervention invariant failed: {message}")
