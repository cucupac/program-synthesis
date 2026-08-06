"""Extract candidate subchains from starter solutions."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path

from experiment.dsl import render_grid
from experiment.generator import DEFAULT_CONFIG_PATH, GeneratedWorld, load_config, make_world
from experiment.solver import SolveConfig, primitive_library, solve_tasks
from experiment.subchains import (
    Candidate,
    CandidateMenu,
    candidate_size_distribution,
    extract_candidate_menu,
)

DEFAULT_OUTPUT_PATH = "experiment/data/extracted_subchains/candidates_smoke.json"
MAX_SOLUTIONS = 3


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    output = build_candidate_menu_payload(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, sort_keys=True)
        file.write("\n")

    print(
        f"solved {output['solved_starter_task_count']}/"
        f"{output['starter_task_count']} starter tasks; "
        f"kept {output['kept_candidate_count']}/"
        f"{output['raw_candidate_count']} raw candidates; "
        f"wrote {output_path}"
    )


def build_candidate_menu_payload(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    config = load_config(config_path)
    world = make_world(config, config.world_seeds[0], config.conditions[0])
    solve_config = SolveConfig(max_solutions=MAX_SOLUTIONS)
    results = solve_tasks(
        [task.target for task in world.tasks_start],
        library=primitive_library(),
        config=solve_config,
    )
    solutions = [
        (f"{task.id}:solution_{index}", solution)
        for task, result in zip(world.tasks_start, results)
        if result.solved
        for index, solution in enumerate(result.solutions[:MAX_SOLUTIONS])
    ]
    menu = extract_candidate_menu(solutions)

    return {
        "world_seed": world.world_seed,
        "condition": world.condition.name,
        "solver": {
            "node_budget": solve_config.node_budget,
            "max_program_size": solve_config.max_program_size,
            "max_solutions": solve_config.max_solutions,
        },
        "extraction": {
            "dedupe_by": menu.dedupe_by,
            "min_op_count": menu.min_op_count,
            "min_support": menu.min_support,
        },
        "starter_task_count": len(world.tasks_start),
        "solved_starter_task_count": sum(result.solved for result in results),
        "starter_solution_count": len(solutions),
        "raw_candidate_count": menu.raw_candidate_count,
        "kept_candidate_count": len(menu.candidates),
        "candidate_size_distribution": candidate_size_distribution(menu.candidates),
        "candidate_support_counts": {
            candidate.program_string: candidate.support_count
            for candidate in menu.candidates
        },
        "motif_coverage": _motif_coverage(world, menu),
        "candidates": [
            _candidate_to_dict(index, candidate)
            for index, candidate in enumerate(menu.candidates)
        ],
    }


def _candidate_to_dict(index: int, candidate: Candidate) -> dict:
    return {
        "id": f"C{index:04d}",
        "program": candidate.program_string,
        "program_op_count": candidate.op_count,
        "support_count": candidate.support_count,
        "support_solution_ids": list(candidate.solution_ids),
        "target": render_grid(candidate.output),
    }


def _motif_coverage(world: GeneratedWorld, menu: CandidateMenu) -> dict:
    candidate_targets = Counter(candidate.output for candidate in menu.candidates)
    covered_ids = {
        motif.id for motif in world.motifs if candidate_targets[motif.target] > 0
    }

    def covered_mass(distribution: tuple[float, ...]) -> float:
        return sum(
            probability
            for motif, probability in zip(world.motifs, distribution)
            if motif.id in covered_ids
        )

    return {
        "covered_motif_count": len(covered_ids),
        "motif_count": len(world.motifs),
        "covered_start_mass": covered_mass(world.p_start),
        "covered_val_mass": covered_mass(world.p_val),
        "covered_test_mass": covered_mass(world.p_test),
        "motifs": [
            {
                "motif_id": motif.id,
                "candidate_count": candidate_targets[motif.target],
                "covered": motif.id in covered_ids,
            }
            for motif in world.motifs
        ],
    }


if __name__ == "__main__":
    main()
