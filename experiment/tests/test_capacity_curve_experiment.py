import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from experiment import capacity_curve as capacity
from experiment.commands import run_capacity_curve_experiment as runner
from experiment.dsl import execute, primitive
from experiment.frontier_promotion import FrontierCandidate
from experiment.generator import load_config, make_world
from experiment.selection import SelectionResult


class CapacityCurveRunnerTests(unittest.TestCase):
    def test_cli_rejects_arbitrary_experiment_shape(self):
        for option in (
            "--k",
            "--seed",
            "--seeds",
            "--condition",
            "--conditions",
            "--output",
            "--force",
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit), redirect_stderr(
                io.StringIO()
            ):
                runner.main([option, "x"])

    def test_builds_exact_formal_and_smoke_jobs(self):
        formal = runner.build_jobs(Path("cells"), workers=4, smoke=False)
        smoke = runner.build_jobs(Path("cells"), workers=1, smoke=True)

        self.assertEqual(len(formal), 180)
        self.assertEqual(
            {(job["seed"], job["condition"]) for job in formal},
            {
                (seed, condition)
                for seed in capacity.FORMAL_SEEDS
                for condition in capacity.CONDITIONS
            },
        )
        self.assertTrue(all(job["formal_seed"] is True for job in formal))
        self.assertTrue(all(job["selector_workers"] == 1 for job in formal))
        self.assertEqual(
            [(job["seed"], job["condition"], job["formal_seed"]) for job in smoke],
            [(6511, "reversed_a0", False)],
        )

    def test_fresh_look_guard_rejects_reserved_seed_and_formal_artifacts(self):
        cases = {
            "integer.json": {"world_seed": 6541},
            "string.json": {"registration": {"seeds": ["6541"]}},
            "nested.json": {"rows": [{"seed": 6541}]},
            "mapped_seed.json": {"seed": {"value": 6541}},
            "mapped_range.json": {
                "registration": {"seeds": {"start": 6541, "end": 6570}}
            },
            "seed_indexed.json": {
                "registration": {"seeds": {"6541": {"status": "generated"}}}
            },
        }
        for name, payload in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "fresh-look"):
                    runner.fresh_look_guard(root)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / f"{capacity.EXPERIMENT_NAME}.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                runner.fresh_look_guard(root)

    def test_fresh_look_guard_ignores_unrelated_metrics_and_spent_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "metric.json").write_text(
                json.dumps({"score": 6541, "seed": 6511}), encoding="utf-8"
            )

            runner.fresh_look_guard(root)

    def test_formal_claim_scans_all_experiment_data(self):
        with tempfile.TemporaryDirectory() as temp:
            data_root = Path(temp) / "data"
            selection_root = data_root / "selection"
            problem_root = data_root / "problem_sets"
            problem_root.mkdir(parents=True)
            (problem_root / "reserved.json").write_text(
                json.dumps({"world_seed": 6541}), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "fresh-look"):
                runner.claim_formal_output(
                    selection_root / "capacity_curve",
                    selection_root=selection_root,
                )

    def test_assisted_compression_charges_acquisition_once_per_nonzero_prefix(self):
        candidate = FrontierCandidate(
            program=primitive("square"),
            program_string="square",
            output=execute(primitive("square")),
            op_count=1,
            support_task_ids=("start_000",),
            first_hit_cost=1,
        )
        base_cost = capacity.zero_cost()
        base_cost["trial_libraries_evaluated"] = 1
        selection = SelectionResult(
            (candidate,),
            base_cost,
            (base_cost,),
            ({"round": 1},),
        )
        result = type("Result", (), {"candidates_tried_total": 7})()

        charged = runner.with_input_search_cost(selection, [result])

        self.assertEqual(
            charged.prefix_costs[0]["input_solution_search_candidate_programs_tried"],
            7,
        )
        self.assertEqual(
            charged.prefix_costs[0]["selection_cost_candidate_programs_tried"],
            7,
        )
        self.assertEqual(charged.prefix_costs[-1], charged.cost)
        self.assertEqual(charged.round_diagnostics, selection.round_diagnostics)

    def test_shared_hashes_match_across_conditions_with_same_seed(self):
        config = load_config("experiment/configs/generator.yaml")
        condition_by_name = {item.name: item for item in config.conditions}
        first = make_world(config, 6511, condition_by_name["reversed_a0"])
        second = make_world(config, 6511, condition_by_name["permuted_a1"])

        first_hashes, first_programs = runner.shared_world_hashes(first)
        second_hashes, second_programs = runner.shared_world_hashes(second)

        self.assertEqual(first_hashes, second_hashes)
        self.assertEqual(first_programs, second_programs)

    def test_formal_directory_can_be_claimed_only_once(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
            runner, "fresh_look_guard"
        ) as guard:
            selection_root = Path(temp) / "selection"
            output_root = selection_root / "capacity_curve"

            runner.claim_formal_output(output_root, selection_root=selection_root)

            guard.assert_called_once_with(selection_root.parent)
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                runner.claim_formal_output(output_root, selection_root=selection_root)


if __name__ == "__main__":
    unittest.main()
