"""Smoke-test primitive-only solving on generated starter tasks."""

from __future__ import annotations

import json
from pathlib import Path

from experiment.dsl import execute
from experiment.generator import DEFAULT_CONFIG_PATH, load_config, make_world
from experiment.solver import SolveConfig, primitive_library, solve_tasks


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    world = make_world(config, config.world_seeds[0], config.conditions[0])
    solve_config = SolveConfig()
    results = solve_tasks(
        [task.target for task in world.tasks_start],
        library=primitive_library(),
        config=solve_config,
    )

    solved = sum(result.solved for result in results)
    solved_results = [result for result in results if result.solved]
    average_first_cost = (
        sum(result.candidates_tried_at_first_solution or 0 for result in solved_results)
        / len(solved_results)
        if solved_results
        else None
    )
    motif_coverage = _motif_coverage(world, results)
    rare_motif_ids = _rare_motif_ids(world)
    rare_motifs_with_hits = sum(
        1 for motif_id in rare_motif_ids if motif_coverage[motif_id] > 0
    )
    rare_motif_solution_hits = sum(motif_coverage[motif_id] for motif_id in rare_motif_ids)
    solve_rate = solved / len(results)
    smoke_ok = _smoke_ok(solve_rate, rare_motifs_with_hits)

    output = {
        "world_seed": world.world_seed,
        "condition": world.condition.name,
        "solver": {
            "node_budget": solve_config.node_budget,
            "max_program_size": solve_config.max_program_size,
            "max_solutions": solve_config.max_solutions,
        },
        "task_count": len(results),
        "solved": solved,
        "solve_rate": solve_rate,
        "average_first_solution_cost": average_first_cost,
        "motif_coverage": _motif_coverage_rows(world, motif_coverage, rare_motif_ids),
        "rare_motif_ids": list(rare_motif_ids),
        "rare_motifs_with_hits": rare_motifs_with_hits,
        "rare_motif_solution_hits": rare_motif_solution_hits,
        "smoke_target": "solve_rate_60_to_90_percent_and_rare_motif_coverage_above_zero",
        "smoke_ok": smoke_ok,
    }

    output_path = Path("experiment/data/solution_sets/primitive_solver_smoke.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, sort_keys=True)
        file.write("\n")

    print(
        f"solved {solved}/{len(results)} "
        f"({solve_rate:.1%}); avg first cost={average_first_cost}; "
        f"rare motif hits={rare_motif_solution_hits}; "
        f"rare motifs covered={rare_motifs_with_hits}/{len(rare_motif_ids)}; "
        f"smoke_ok={smoke_ok}"
    )

    if not smoke_ok:
        raise SystemExit(1)


def _motif_coverage(world, results) -> dict[str, int]:
    motifs_by_id = {motif.id: motif.target for motif in world.motifs}
    coverage = {motif.id: 0 for motif in world.motifs}
    for result in results:
        for solution in result.solutions:
            for motif_id, motif_target in motifs_by_id.items():
                if _contains_subtree_output(solution, motif_target):
                    coverage[motif_id] += 1
    return coverage


def _motif_coverage_rows(world, coverage, rare_motif_ids) -> list[dict]:
    rare_motifs = set(rare_motif_ids)
    return [
        {
            "motif_id": motif.id,
            "starter_probability": probability,
            "solution_hits": coverage[motif.id],
            "is_starter_rare": motif.id in rare_motifs,
        }
        for motif, probability in zip(world.motifs, world.p_start)
    ]


def _rare_motif_ids(world) -> tuple[str, ...]:
    ranked = sorted(
        zip(world.motifs, world.p_start),
        key=lambda item: (item[1], item[0].id),
    )
    return tuple(motif.id for motif, _ in ranked[: len(ranked) // 2])


def _smoke_ok(solve_rate: float, rare_motifs_with_hits: int) -> bool:
    return 0.60 <= solve_rate <= 0.90 and rare_motifs_with_hits > 0


def _contains_subtree_output(program, target) -> bool:
    if execute(program) == target:
        return True
    return any(_contains_subtree_output(arg, target) for arg in program.args)


if __name__ == "__main__":
    main()
