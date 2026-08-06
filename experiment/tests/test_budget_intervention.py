import copy
import tempfile
import unittest
from pathlib import Path

from experiment import budget_intervention as intervention
from experiment import capacity_curve as capacity
from experiment.commands.run_budget_intervention import (
    build_jobs,
    claim_formal_output,
    main,
)
from experiment.dsl import program_to_string
from experiment.solver import SolveConfig, build_frontier_index, primitive_library
from experiment.tests.test_capacity_curve import synthetic_cell


def _target_rows(prefix, count):
    return [
        {
            "task_id": f"{prefix}{index:03d}",
            "target_hash": "0" * 64,
            "first_hit_rank": None,
            "abstract_search_size": None,
        }
        for index in range(count)
    ]


def _summary(task_count):
    return {
        "solved_count": 0,
        "failure_count": task_count,
        "task_count": task_count,
        "solve_rate": 0.0,
        "mean_search_cost": 30_000.0,
        "mean_first_solution_cost": None,
        "frontier_candidates_tried_total": 30_000,
        "hit_budget": False,
        "unique_outputs": 100,
    }


def _evaluation(k, programs):
    validation = _target_rows("v", capacity.VALIDATION_TASK_COUNT)
    test = _target_rows("t", capacity.TEST_TASK_COUNT)
    leaves = capacity.PRIMITIVE_LIBRARY_SIZE + k
    return {
        "k": k,
        "selected_programs": programs[:k],
        "max_frontier": {
            "candidates_tried_total": 30_000,
            "candidates_tried_by_size": [
                leaves,
                30_000 - leaves,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "hit_budget": False,
            "unique_outputs": 100,
            "first_size4_rank": None,
        },
        "budgets": [
            {
                "budget": budget,
                "candidates_tried_by_size": [
                    leaves,
                    30_000 - leaves,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "validation_summary": _summary(capacity.VALIDATION_TASK_COUNT),
                "test_summary": _summary(capacity.TEST_TASK_COUNT),
            }
            for budget in intervention.BUDGETS
        ],
        "validation_targets": validation,
        "test_targets": test,
    }


def synthetic_intervention_cell(*, formal=True):
    programs = [f"program_{index}" for index in range(capacity.K_MAX)]
    primitive = _evaluation(0, programs)
    arms = {
        name: {
            "selected_programs": list(programs[:2]),
            "prefixes": [_evaluation(1, programs), _evaluation(2, programs)],
        }
        for name in intervention.INCLUDED_ARMS
    }
    task_hashes = {
        "validation": [
            {"task_id": row["task_id"], "target_hash": row["target_hash"]}
            for row in primitive["validation_targets"]
        ],
        "test": [
            {"task_id": row["task_id"], "target_hash": row["target_hash"]}
            for row in primitive["test_targets"]
        ],
    }
    return {
        "experiment_name": (
            intervention.EXPERIMENT_NAME
            if formal
            else intervention.SMOKE_EXPERIMENT_NAME
        ),
        "smoke": not formal,
        "seed": capacity.FORMAL_SEEDS[0] if formal else capacity.SMOKE_SEEDS[0],
        "condition": capacity.CONDITIONS[0],
        "source": intervention.registration(smoke=not formal)["source"],
        "source_cell_hash": "1" * 64,
        "world_metadata": {
            "condition": capacity.CONDITION_SPECS[capacity.CONDITIONS[0]],
            "realized_rho": {
                "realized_start_test": 0.5,
                "realized_start_val": 0.5,
                "realized_val_test": 0.5,
            },
            "density_summary": {},
            "expected_motif_length": {},
        },
        "shared_hashes": {
            "hidden_motifs": "2" * 64,
            "p_start": "3" * 64,
            "starter_tasks": "4" * 64,
            "candidate_menu": capacity.canonical_hash(programs),
        },
        "candidate_menu_programs": programs,
        "task_hashes": task_hashes,
        "primitive": primitive,
        "arms": arms,
        "mechanism": {
            name: intervention.mechanism_rows(
                arms[name]["prefixes"][0], arms[name]["prefixes"][1]
            )
            for name in intervention.PRIMARY_ARMS
        },
        "wall_clock_seconds": 1.0,
    }


def synthetic_source_anchor(cell):
    return {
        "seed": cell["seed"],
        "condition": cell["condition"],
        "source_cell_hash": cell["source_cell_hash"],
        "world_metadata": copy.deepcopy(cell["world_metadata"]),
        "shared_hashes": copy.deepcopy(cell["shared_hashes"]),
        "candidate_menu_programs": list(cell["candidate_menu_programs"]),
        "task_hashes": copy.deepcopy(cell["task_hashes"]),
        "primitive": {
            "validation_summary": copy.deepcopy(
                cell["primitive"]["budgets"][0]["validation_summary"]
            ),
            "test_summary": copy.deepcopy(
                cell["primitive"]["budgets"][0]["test_summary"]
            ),
        },
        "arms": {
            name: {
                "selected_programs": list(arm["selected_programs"]),
                "prefixes": [
                    {
                        "k": row["k"],
                        "validation_summary": copy.deepcopy(
                            row["budgets"][0]["validation_summary"]
                        ),
                        "test_summary": copy.deepcopy(
                            row["budgets"][0]["test_summary"]
                        ),
                    }
                    for row in arm["prefixes"]
                ],
            }
            for name, arm in cell["arms"].items()
        },
    }


def synthetic_smoke_aggregate():
    cell = synthetic_intervention_cell(formal=False)
    return (
        {
            "experiment_name": intervention.SMOKE_EXPERIMENT_NAME,
            "smoke": True,
            "registration": intervention.registration(smoke=True),
            "cells": [cell],
        },
        [synthetic_source_anchor(cell)],
    )


class BudgetInterventionTests(unittest.TestCase):
    def test_registration_freezes_source_budgets_and_estimands(self):
        row = intervention.registration(smoke=False)

        self.assertEqual(row["registration"], "R14")
        self.assertEqual(row["source"]["sha256"], intervention.FORMAL_SOURCE_SHA256)
        self.assertEqual(row["budgets"], [30_000, 45_000, 60_000, 90_000])
        self.assertFalse(row["independent_confirmation"])
        self.assertTrue(row["no_reselection"])
        self.assertEqual(
            row["primary_estimands"]["remaining_gap_at_90000"],
            "D_m(90000)_reported_to_distinguish_attenuation_from_elimination",
        )
        self.assertEqual(
            row["inference"]["bootstrap_statistic"],
            "maximum_absolute_centered_studentized_deviation",
        )
        mechanism = synthetic_intervention_cell()["mechanism"][
            capacity.UTILITY_ARM
        ][0]
        self.assertIn("lost_test_rate", mechanism)
        self.assertIn("recovered_test_rate", mechanism)
        self.assertNotIn("lost_rate", mechanism)
        self.assertNotIn("recovered_rate", mechanism)

    def test_threshold_counts_include_duplicate_attempts_in_size_order(self):
        from experiment.solver import FrontierIndex

        index = FrontierIndex(
            entries={},
            candidates_tried_total=90_000,
            hit_budget=True,
            candidates_tried_by_size=(8, 29_992, 20_000, 20_000, 20_000, 0, 0, 0),
        )

        self.assertEqual(
            intervention.candidates_tried_by_size(index, 30_000),
            (8, 29_992, 0, 0, 0, 0, 0, 0),
        )
        self.assertEqual(
            intervention.candidates_tried_by_size(index, 45_000),
            (8, 29_992, 15_000, 0, 0, 0, 0, 0),
        )

    def test_30000_frontier_is_exact_prefix_of_90000_frontier(self):
        direct = build_frontier_index(
            primitive_library(), SolveConfig(node_budget=30_000, max_solutions=1)
        )
        maximum = build_frontier_index(
            primitive_library(), SolveConfig(node_budget=90_000, max_solutions=1)
        )
        prefix = {
            output: entry
            for output, entry in maximum.entries.items()
            if entry.candidates_tried_at_first_solution <= 30_000
        }

        self.assertEqual(set(direct.entries), set(prefix))
        for output, entry in direct.entries.items():
            self.assertEqual(
                (
                    program_to_string(entry.program),
                    entry.candidates_tried_at_first_solution,
                    entry.abstract_search_size,
                ),
                (
                    program_to_string(prefix[output].program),
                    prefix[output].candidates_tried_at_first_solution,
                    prefix[output].abstract_search_size,
                ),
            )
        self.assertEqual(
            intervention.candidates_tried_by_size(maximum, 30_000),
            direct.candidates_tried_by_size,
        )

    def test_validates_cell_and_reconstructs_mechanism(self):
        cell = synthetic_intervention_cell()
        anchor = synthetic_source_anchor(cell)

        intervention.validate_cell(cell, formal=True, source_anchor=anchor)

        broken = copy.deepcopy(cell)
        broken["arms"][capacity.UTILITY_ARM]["prefixes"][1]["test_targets"][0][
            "first_hit_rank"
        ] = 10
        with self.assertRaisesRegex(ValueError, "both be null"):
            intervention.validate_cell(broken, formal=True, source_anchor=anchor)

    def test_rejects_mutated_test_summary_and_post_hoc_field(self):
        for mutation in ("solved_count", "p_value"):
            payload, anchors = synthetic_smoke_aggregate()
            summary = payload["cells"][0]["arms"][capacity.UTILITY_ARM][
                "prefixes"
            ][1]["budgets"][-1]["test_summary"]
            summary[mutation] = 1

            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ValueError, "summary"
            ):
                intervention.validate_aggregate(
                    payload, formal=False, source_anchors=anchors
                )

    def test_rejects_extra_budget_row_field(self):
        cell = synthetic_intervention_cell()
        anchor = synthetic_source_anchor(cell)
        cell["arms"][capacity.UTILITY_ARM]["prefixes"][1]["budgets"][-1][
            "p_value"
        ] = 0.01

        with self.assertRaisesRegex(ValueError, "budget row"):
            intervention.validate_cell(cell, formal=True, source_anchor=anchor)

    def test_rejects_library_substitution_against_source_anchor(self):
        cell = synthetic_intervention_cell()
        anchor = synthetic_source_anchor(cell)
        replacements = cell["candidate_menu_programs"][2:4]
        arm = cell["arms"][capacity.UTILITY_ARM]
        arm["selected_programs"] = replacements
        for k, prefix in enumerate(arm["prefixes"], start=1):
            prefix["selected_programs"] = replacements[:k]

        with self.assertRaisesRegex(ValueError, "source selected prefix"):
            intervention.validate_cell(cell, formal=True, source_anchor=anchor)

    def test_rejects_k2_frontier_with_primitive_only_leaf_count(self):
        cell = synthetic_intervention_cell()
        anchor = synthetic_source_anchor(cell)
        prefix = cell["arms"][capacity.UTILITY_ARM]["prefixes"][1]
        prefix["max_frontier"]["candidates_tried_by_size"][0] = 6
        prefix["max_frontier"]["candidates_tried_by_size"][1] += 2
        for row in prefix["budgets"]:
            row["candidates_tried_by_size"][0] = 6
            row["candidates_tried_by_size"][1] += 2

        with self.assertRaisesRegex(ValueError, "size-zero"):
            intervention.validate_cell(cell, formal=True, source_anchor=anchor)

    def test_builds_exact_formal_and_smoke_job_sets(self):
        formal_cells = [
            synthetic_cell(seed=seed, condition=condition)
            for seed in capacity.FORMAL_SEEDS
            for condition in capacity.CONDITIONS
        ]
        smoke_cell = synthetic_cell(seed=6511, condition="reversed_a0", formal=False)

        formal_jobs = build_jobs(
            {"cells": formal_cells}, Path("cells"), workers=8, smoke=False
        )
        smoke_jobs = build_jobs(
            {"cells": [smoke_cell]}, Path("cells"), workers=1, smoke=True
        )

        self.assertEqual(len(formal_jobs), 180)
        self.assertEqual(
            {(job["source"]["seed"], job["source"]["condition"]) for job in formal_jobs},
            {
                (seed, condition)
                for seed in capacity.FORMAL_SEEDS
                for condition in capacity.CONDITIONS
            },
        )
        self.assertEqual(len(smoke_jobs), 1)
        self.assertEqual(smoke_jobs[0]["source"]["seed"], 6511)

    def test_cli_rejects_unregistered_options(self):
        with self.assertRaises(SystemExit):
            main(["--k", "3"])

    def test_formal_output_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "budget_intervention"
            claim_formal_output(root)
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                claim_formal_output(root)


if __name__ == "__main__":
    unittest.main()
