"""Run the selector-relevance gate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from statistics import median

from experiment.dsl import render_grid
from experiment.frontier_promotion import (
    FrontierCandidate,
    frontier_promotion_menu,
    menu_diagnostics,
)
from experiment.generator import (
    DEFAULT_CONFIG_PATH,
    Motif,
    load_config,
    make_world,
    spearman_rho,
)
from experiment.selection import (
    candidates_to_library,
    greedy_by_solved_count_with_cost,
    select_compression_k,
    select_most_frequent_k,
    select_random_k,
    select_utility_k,
    solve_library_summary,
)
from experiment.solver import SolveConfig, primitive_library, solve_tasks

DEFAULT_OUTPUT_PATH = "experiment/data/selection/selector_relevance_gate.json"
GATE_NAME = "selector_relevance_gate"
GATE_SEEDS = (6477, 6478, 6479, 6480)
CONDITIONS = ("reversed_a0", "reversed_a1")
K = 10
RANDOM_DRAWS = 20
THRESHOLD_PATH = "experiment/data/extracted_subchains/gate3_candidate_menu_diagnostics.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in GATE_SEEDS))
    parser.add_argument("--random-draws", type=int, default=RANDOM_DRAWS)
    args = parser.parse_args(argv)

    seeds = tuple(int(seed) for seed in args.seeds.split(",") if seed)
    if args.random_draws < 1:
        raise ValueError("random_draws must be at least 1")
    _fresh_look_guard(seeds, Path(args.output))

    payload = run_selector_gate(
        config_path=args.config,
        seeds=seeds,
        random_draws=args.random_draws,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")

    print(
        f"{payload['gate_name']}: pass={payload['overall_pass']} "
        f"selector_ready={payload['selector_ready']} wrote {output_path}"
    )


def run_selector_gate(
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
    seeds: Sequence[int] = GATE_SEEDS,
    random_draws: int = RANDOM_DRAWS,
) -> dict:
    config = load_config(config_path)
    if random_draws < 1:
        raise ValueError("random_draws must be at least 1")
    conditions = [c for c in config.conditions if c.name in CONDITIONS]
    if {condition.name for condition in conditions} != set(CONDITIONS):
        raise ValueError("config must include reversed_a0 and reversed_a1")

    thresholds = _load_thresholds()
    registration = {
        "gate_name": GATE_NAME,
        "gate_seeds": list(seeds),
        "conditions": list(CONDITIONS),
        "recipe": "frontier_promotion",
        "k": K,
        "random_draws": random_draws,
        "candidate_cap": 50,
        "pass_rule": {
            "endpoint": (
                "median across worlds of "
                "best_k_oracle_solved - median(random_k_solved) >= threshold"
            ),
            "validation": (
                "reversed_a0 Spearman rho between validation and test solved "
                "counts across random and real libraries >= 0.5"
            ),
            "overall": "both endpoints pass and reversed_a0 validation premise passes",
        },
        "thresholds": thresholds,
    }

    cells = []
    validation_pairs: list[dict] = []
    for seed in seeds:
        for condition in conditions:
            print(f"running {seed}/{condition.name}", flush=True)
            cell, pairs = _run_cell(config, seed, condition, random_draws)
            cells.append(cell)
            if condition.name == "reversed_a0":
                validation_pairs.extend(pairs)

    endpoint_decisions = {
        name: _endpoint_decision(name, cells, thresholds[name])
        for name in CONDITIONS
    }
    validation_decision = _validation_decision(validation_pairs)
    overall_pass = all(item["pass"] for item in endpoint_decisions.values()) and validation_decision[
        "pass"
    ]

    return {
        "gate_name": GATE_NAME,
        "registration": registration,
        "cells": cells,
        "endpoint_decisions": endpoint_decisions,
        "validation_test_prediction": validation_decision,
        "overall_pass": overall_pass,
        "selector_ready": overall_pass,
        "next_step": (
            "full_selector_experiment_on_6481_plus"
            if overall_pass
            else _failure_next_step(endpoint_decisions, validation_decision)
        ),
    }


def _run_cell(config, seed: int, condition, random_draws: int) -> tuple[dict, list[dict]]:
    world = make_world(config, seed, condition)
    menu = frontier_promotion_menu(world.tasks_start)
    test_targets = [task.target for task in world.tasks_test]
    val_targets = [task.target for task in world.tasks_val]
    solve_config = SolveConfig(max_solutions=1)

    primitive_summary = solve_library_summary(test_targets, primitive_library(), solve_config)
    random_rows = []
    validation_pairs = []
    for draw in range(random_draws):
        selected = select_random_k(menu.candidates, K, f"{seed}:{condition.name}:{draw}")
        test_summary = solve_library_summary(
            test_targets, candidates_to_library(selected), solve_config
        )
        val_summary = solve_library_summary(
            val_targets, candidates_to_library(selected), solve_config
        )
        random_rows.append(_arm_row(f"random_{draw:02d}", selected, test_summary))
        validation_pairs.append(
            _pair(seed, condition.name, f"random_{draw:02d}", val_summary, test_summary)
        )

    starter_solutions = _starter_solutions(world.tasks_start)
    arms = {
        "most_frequent_k": select_most_frequent_k(menu.candidates, K),
        "compression_on_starter": select_compression_k(
            menu.candidates, starter_solutions, K
        ),
        "utility_on_validation": select_utility_k(menu.candidates, val_targets, K),
        "best_k_from_c_oracle": greedy_by_solved_count_with_cost(
            menu.candidates, test_targets, K
        ).candidates,
        "hidden_motif_oracle": greedy_by_solved_count_with_cost(
            _motif_candidates(world.motifs), test_targets, K
        ).candidates,
    }

    arm_rows = {
        "primitives_only": {
            "selected_candidate_ids": [],
            "selected_programs": [],
            "summary": primitive_summary,
        },
        "random_k": {
            "draws": random_rows,
            "median_solved_count": median(
                row["summary"]["solved_count"] for row in random_rows
            ),
        },
    }
    for name, selected in arms.items():
        summary = solve_library_summary(test_targets, candidates_to_library(selected), solve_config)
        arm_rows[name] = _arm_row(name, selected, summary)
        if name != "hidden_motif_oracle":
            val_summary = solve_library_summary(
                val_targets, candidates_to_library(selected), solve_config
            )
            validation_pairs.append(_pair(seed, condition.name, name, val_summary, summary))

    best = arm_rows["best_k_from_c_oracle"]["summary"]["solved_count"]
    motif = arm_rows["hidden_motif_oracle"]["summary"]["solved_count"]
    primitive = primitive_summary["solved_count"]
    motif_delta = motif - primitive
    c_delta = best - primitive

    return (
        {
            "seed": seed,
            "condition": condition.name,
            "world_metadata": {
                "realized_rho": world.metadata["realized_rho"],
                "density_summary": world.metadata["density_summary"],
                "expected_motif_length": world.metadata["expected_motif_length"],
            },
            "menu": menu_diagnostics(menu),
            "arms": arm_rows,
            "oracle_minus_random_median": best - arm_rows["random_k"]["median_solved_count"],
            "capture_ratio": c_delta / motif_delta if motif_delta > 0 else None,
            "c_delta": c_delta,
            "motif_oracle_delta": motif_delta,
        },
        validation_pairs,
    )


def _starter_solutions(tasks_start) -> tuple:
    results = solve_tasks(
        [task.target for task in tasks_start],
        library=primitive_library(),
        config=SolveConfig(max_solutions=3),
    )
    return tuple(solution for result in results for solution in result.solutions)


def _motif_candidates(motifs: Sequence[Motif]) -> tuple[FrontierCandidate, ...]:
    return tuple(
        FrontierCandidate(
            program=motif.program,
            program_string=f"motif:{motif.id}",
            output=motif.target,
            op_count=0,
            support_task_ids=(),
            first_hit_cost=0,
        )
        for motif in motifs
    )


def _arm_row(name: str, selected: Sequence[FrontierCandidate], summary: dict) -> dict:
    return {
        "arm": name,
        "selected_programs": [candidate.program_string for candidate in selected],
        "selected_targets": [render_grid(candidate.output) for candidate in selected],
        "selection_cost_candidate_programs_tried": 0,
        "summary": summary,
    }


def _pair(seed: int, condition: str, arm: str, val_summary: dict, test_summary: dict) -> dict:
    return {
        "seed": seed,
        "condition": condition,
        "arm": arm,
        "validation_solved_count": val_summary["solved_count"],
        "test_solved_count": test_summary["solved_count"],
    }


def _endpoint_decision(condition: str, cells: Sequence[dict], threshold: float) -> dict:
    deltas = [
        cell["oracle_minus_random_median"]
        for cell in cells
        if cell["condition"] == condition
    ]
    endpoint_median = median(deltas) if deltas else 0
    return {
        "condition": condition,
        "oracle_minus_random_median_by_seed": deltas,
        "median_oracle_minus_random": endpoint_median,
        "threshold": threshold,
        "pass": endpoint_median >= threshold,
    }


def _validation_decision(pairs: Sequence[dict]) -> dict:
    rho = spearman_rho(
        [pair["validation_solved_count"] for pair in pairs],
        [pair["test_solved_count"] for pair in pairs],
    )
    return {"spearman_rho": rho, "threshold": 0.5, "pass": rho >= 0.5, "pairs": list(pairs)}


def _failure_next_step(endpoint_decisions: dict, validation_decision: dict) -> str:
    if not all(item["pass"] for item in endpoint_decisions.values()):
        return "finding_menu_candidates_interchangeable_at_k10"
    if not validation_decision["pass"]:
        return "finding_validation_tasks_do_not_predict_test_gains"
    return "inspect_gate_outputs"


def _load_thresholds(path: str = THRESHOLD_PATH) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    conditions = data["primitive_noise"]["conditions"]
    return {
        name: float(conditions[name]["solved_delta_threshold"])
        for name in CONDITIONS
    }


def _fresh_look_guard(seeds: Sequence[int], output_path: Path) -> None:
    blocked = set(seeds)
    for path in Path("experiment/data").glob("**/*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("gate_name") != GATE_NAME and data.get("registration", {}).get(
            "gate_name"
        ) != GATE_NAME:
            continue
        recorded = set(data.get("gate_seeds", data.get("registration", {}).get("gate_seeds", ())))
        if blocked & recorded:
            raise RuntimeError(f"fresh-look guard blocked by prior artifact: {path}")


if __name__ == "__main__":
    main()
