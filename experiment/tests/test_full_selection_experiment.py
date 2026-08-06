import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

from experiment.commands import run_full_selection_experiment as full
from experiment.generator import DEFAULT_CONFIG_PATH, load_config, make_world


class FullSelectionExperimentTests(unittest.TestCase):
    def test_default_formal_cell_list_uses_30_fresh_seeds(self):
        jobs = [
            (seed, condition)
            for seed in full.FORMAL_SEEDS
            for condition in full.CONDITIONS
        ]

        self.assertEqual(len(jobs), 210)
        self.assertEqual(min(seed for seed, _ in jobs), 6481)
        self.assertEqual(max(seed for seed, _ in jobs), 6510)
        self.assertEqual(
            full.CONDITIONS,
            (
                "reversed_a0",
                "reversed_a05",
                "reversed_a1",
                "permuted_a0",
                "permuted_a05",
                "permuted_a1",
                "stale_reversed",
            ),
        )

    def test_cli_rejects_removed_shape_options(self):
        removed = ("--config", "--output", "--seeds", "--conditions", "--k", "--force")
        for option in removed:
            with self.subTest(option=option), self.assertRaises(SystemExit):
                full.main([option, "x"])

    def test_formal_entrypoint_uses_registered_shape(self):
        with patch.object(
            full, "_run_cells", return_value={"experiment_name": full.EXPERIMENT_NAME, "cells": [], "smoke": False}
        ) as run_cells:
            payload = full.run_formal_experiment(workers=2)

        self.assertFalse(payload["smoke"])
        kwargs = run_cells.call_args.kwargs
        self.assertEqual(kwargs["config_path"], DEFAULT_CONFIG_PATH)
        self.assertEqual(kwargs["output_path"], Path(full.DEFAULT_OUTPUT_PATH))
        self.assertEqual(kwargs["seeds"], full.FORMAL_SEEDS)
        self.assertEqual(kwargs["conditions"], full.CONDITIONS)
        self.assertEqual(kwargs["k"], full.DEFAULT_K)
        self.assertEqual(kwargs["random_draws"], full.RANDOM_DRAWS)
        self.assertFalse(kwargs["force"])

    def test_smoke_entrypoint_uses_fixed_spent_shape(self):
        with patch.object(
            full, "_run_cells", return_value={"experiment_name": full.SMOKE_EXPERIMENT_NAME, "cells": [], "smoke": True}
        ) as run_cells:
            payload = full.run_smoke_experiment(workers=2)

        self.assertTrue(payload["smoke"])
        self.assertEqual(payload["experiment_name"], full.SMOKE_EXPERIMENT_NAME)
        kwargs = run_cells.call_args.kwargs
        self.assertEqual(kwargs["config_path"], DEFAULT_CONFIG_PATH)
        self.assertEqual(kwargs["output_path"], Path(full.SMOKE_OUTPUT_PATH))
        self.assertEqual(kwargs["seeds"], full.SMOKE_SEEDS)
        self.assertEqual(kwargs["conditions"], ("reversed_a0",))
        self.assertEqual(kwargs["k"], 1)
        self.assertEqual(kwargs["random_draws"], 1)
        self.assertTrue(kwargs["force"])

    def test_smoke_does_not_claim_formal_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = SimpleNamespace(conditions=[SimpleNamespace(name="reversed_a0")])
            with patch.object(full, "REPO_ROOT", root), patch.object(
                full, "load_config", return_value=config
            ), patch.object(
                full,
                "run_cell",
                return_value=_fake_cell(experiment_name=full.SMOKE_EXPERIMENT_NAME),
            ):
                full.run_smoke_experiment(workers=1)

            self.assertTrue((root / full.SMOKE_OUTPUT_PATH).exists())
            self.assertFalse((root / full.DEFAULT_OUTPUT_PATH).with_suffix("").exists())

    def test_private_runner_rejects_custom_formal_seed_shape(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(full, "run_cell") as run_cell:
            with self.assertRaises(ValueError):
                full._run_cells(
                    config_path=DEFAULT_CONFIG_PATH,
                    output_path=Path(temp) / "custom.json",
                    seeds=(6481,),
                    conditions=("reversed_a0",),
                    k=1,
                    random_draws=1,
                    workers=1,
                    force=False,
                    smoke=False,
                )

        run_cell.assert_not_called()

    def test_fresh_look_guard_blocks_prior_formal_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "experiment/data/selection/full_selection_experiment.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(
                json.dumps(
                    {
                        "experiment_name": full.EXPERIMENT_NAME,
                        "registration": {"seeds": [6481]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_blocks_formal_named_artifact_without_seed_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "experiment/data/selection/renamed_prior.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(json.dumps({"experiment_name": full.EXPERIMENT_NAME}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_ignores_smoke_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / full.SMOKE_OUTPUT_PATH
            artifact.parent.mkdir(parents=True)
            artifact.write_text(
                json.dumps(
                    {
                        "experiment_name": full.SMOKE_EXPERIMENT_NAME,
                        "registration": {
                            "experiment_name": full.SMOKE_EXPERIMENT_NAME,
                            "seeds": [6460],
                        },
                    }
                ),
                encoding="utf-8",
            )
            _fresh_guard(root)

    def test_fresh_look_guard_blocks_corrupt_formal_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / full.DEFAULT_OUTPUT_PATH
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{not json", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_blocks_corrupt_primary_named_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / f"experiment/data/selection/{full.EXPERIMENT_NAME}.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{not json", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_blocks_corrupt_formal_cell_outside_current_cell_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "experiment/data/selection/other_cells/6481_reversed_a0.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{not json", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_blocks_parseable_junk_under_formal_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "experiment/data/selection/full_selection_experiment/cells/junk.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_blocks_non_json_artifact_under_formal_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "experiment/data/selection/full_selection_experiment/cells/junk.txt"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("junk", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_blocks_junk_under_formal_artifact_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "experiment/data/selection/full_selection_experiment/junk.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_blocks_parseable_junk_formal_seed_cell_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "experiment/data/selection/other_cells/6481_reversed_a0.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_blocks_matching_formal_cell_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "experiment/data/selection/full_selection_experiment/cells/6481_reversed_a0.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(
                json.dumps(_fake_cell(seed=6481, formal_seed=True, k=10, random_draws=20)),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_blocks_prior_formal_cell_artifact_outside_current_cell_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "experiment/data/selection/other_cells/6481_reversed_a0.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(
                json.dumps(
                    {
                        "experiment_name": full.EXPERIMENT_NAME,
                        "seed": 6481,
                        "condition": "reversed_a0",
                        "formal_seed": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _fresh_guard(root)

    def test_fresh_look_guard_blocks_numeric_string_formal_seed_metadata(self):
        payloads = (
            {"seed": "6481"},
            {"registration": {"seeds": ["6481"]}},
        )
        for index, payload in enumerate(payloads):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                artifact = root / f"experiment/data/selection/prior_{index}.json"
                artifact.parent.mkdir(parents=True)
                artifact.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(RuntimeError, "prior artifact"):
                    _fresh_guard(root)

    def test_atomic_formal_claim_allows_only_one_concurrent_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            cell_dir = (
                Path(temp)
                / "experiment/data/selection/full_selection_experiment/cells"
            )
            both_passed_guard = Barrier(2)

            def pass_guard(_cell_dir):
                both_passed_guard.wait()

            def claim():
                try:
                    full._claim_formal_run_directory(cell_dir)
                except RuntimeError:
                    return "blocked"
                return "won"

            with patch.object(full, "_fresh_look_guard", side_effect=pass_guard):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(lambda _: claim(), range(2)))

            self.assertEqual(sorted(results), ["blocked", "won"])
            self.assertTrue(cell_dir.is_dir())

    def test_methodology_records_selector_gate_passed(self):
        text = Path("experiment/docs/methodology.md").read_text(encoding="utf-8")

        self.assertIn("6477–6480 | selector-relevance gate | contaminated — gate passed", text)
        self.assertIn("Selector-relevance gate passed", text)
        self.assertIn("0.696", text)
        self.assertIn("intentionally one-shot", text)
        self.assertNotIn("6477–6480 | selector-relevance gate | reserved", text)

    def test_matched_starter_tasks_are_deterministic_and_not_prefix(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        world = make_world(config, 6460, config.conditions[0])

        first = full.matched_starter_tasks(world.tasks_start, 6460, 25)
        second = full.matched_starter_tasks(world.tasks_start, 6460, 25)

        self.assertEqual([task.id for task in first], [task.id for task in second])
        self.assertNotEqual(
            [task.id for task in first],
            [task.id for task in world.tasks_start[:25]],
        )

    def test_registration_contains_effect_definitions(self):
        registration = full._registration((6481,), ("reversed_a0",), 10, 20)

        self.assertEqual(registration["primary_metric"], "solved_count_delta")
        self.assertEqual(
            registration["k_sensitivity_status"],
            "registered_follow_up_not_run_in_this_command",
        )
        self.assertEqual(registration["secondary_metric"], "mean_search_cost_savings")
        self.assertIn("utility_on_validation", registration["arms"])
        self.assertIn("compression_on_validation_assisted", registration["arms"])
        self.assertIn("compression_on_validation_assisted", registration["primary_data_effect"])
        self.assertIn("utility_on_validation", registration["primary_scoring_effect"])
        self.assertIn("stale_reversed", registration)
        self.assertEqual(
            registration["assisted_validation_solve_config"]["node_budget"],
            full.ASSISTED_VALIDATION_SOLVE_CONFIG.node_budget,
        )
        self.assertEqual(
            registration["selection_cost_policy"]["validation_assisted_solution_search"],
            "charge assisted search only",
        )
        self.assertEqual(
            registration["break_even_policy"]["upfront_cost"],
            "utility selection cost minus assisted-compression selection cost",
        )

    def test_cell_schema_includes_required_arms_and_metrics(self):
        cell = _fake_cell()
        expected_arms = {
            "primitives_only",
            "random_k",
            "most_frequent_k",
            "compression_on_matched_25_starter",
            "utility_on_matched_25_starter",
            "compression_on_validation_skip",
            "compression_on_validation_assisted",
            "utility_on_validation",
            "compression_on_all_100_starter",
            "best_k_from_c_oracle",
            "hidden_motif_oracle",
        }
        self.assertEqual(set(cell["arms"]), expected_arms)
        self.assertIn("matched_starter_task_ids", cell)
        self.assertIn("matched_starter_solved_count", cell)
        self.assertIn("selected_set_overlap", cell)
        self.assertIn("wall_clock_seconds", cell)
        self.assertIn("experiment_name", cell)
        self.assertIn("formal_seed", cell)

    def test_validation_summaries_and_prediction_exist(self):
        cell = _fake_cell()
        aggregate = full._aggregates([cell])

        self.assertIn("validation_summary", cell["arms"]["utility_on_validation"])
        self.assertIn("validation_test_prediction", aggregate)
        self.assertIsNotNone(aggregate["validation_test_prediction"]["spearman_rho"])

    def test_formal_cell_payload_accepts_exact_registered_shape(self):
        cell = _fake_cell(seed=6481, formal_seed=True, k=10, random_draws=20)

        full._validate_formal_cell_payload(cell, _formal_job())

    def test_formal_cell_payload_rejects_wrong_metadata(self):
        cell = _fake_cell(seed=6481, formal_seed=True, k=10, random_draws=20)
        cell["seed"] = 6482

        with self.assertRaisesRegex(RuntimeError, "formal cell invariant failed"):
            full._validate_formal_cell_payload(cell, _formal_job())

    def test_formal_cell_payload_rejects_undersized_real_arm(self):
        cell = _fake_cell(seed=6481, formal_seed=True, k=10, random_draws=20)
        cell["arms"]["utility_on_validation"]["selected_programs"].pop()

        with self.assertRaisesRegex(RuntimeError, "formal cell invariant failed"):
            full._validate_formal_cell_payload(cell, _formal_job())

    def test_formal_cell_payload_rejects_undersized_random_draw(self):
        cell = _fake_cell(seed=6481, formal_seed=True, k=10, random_draws=20)
        cell["arms"]["random_k"]["draws"][0]["selected_programs"].pop()

        with self.assertRaisesRegex(RuntimeError, "formal cell invariant failed"):
            full._validate_formal_cell_payload(cell, _formal_job())

    def test_formal_cell_payload_rejects_missing_random_draw(self):
        cell = _fake_cell(seed=6481, formal_seed=True, k=10, random_draws=20)
        cell["arms"]["random_k"]["draws"].pop()

        with self.assertRaisesRegex(RuntimeError, "formal cell invariant failed"):
            full._validate_formal_cell_payload(cell, _formal_job())

    def test_formal_cell_payload_rejects_missing_assisted_search_cost(self):
        cell = _fake_cell(seed=6481, formal_seed=True, k=10, random_draws=20)
        del cell["arms"]["compression_on_validation_assisted"]["selection_cost"][
            "input_solution_search_candidate_programs_tried"
        ]

        with self.assertRaisesRegex(RuntimeError, "assisted solution-search cost"):
            full._validate_formal_cell_payload(cell, _formal_job())

    def test_effect_rows_report_cost_savings(self):
        cell = _fake_cell()
        rows = full._effect_rows(
            [cell],
            "utility_on_validation",
            "compression_on_validation_assisted",
        )

        self.assertEqual(rows[0]["mean_search_cost_delta"], -10)
        self.assertEqual(rows[0]["mean_search_cost_savings"], 10)

    def test_motif_recovery_exact_output_match(self):
        from experiment.frontier_promotion import FrontierCandidate
        from experiment.dsl import execute, primitive
        from experiment.generator import Motif

        motif = Motif("M00", primitive("square"), execute(primitive("square")))
        selected = (
            FrontierCandidate(
                primitive("square"),
                "square",
                execute(primitive("square")),
                0,
                (),
                0,
            ),
        )

        recovery = full._motif_recovery(selected, (motif,))

        self.assertEqual(recovery["motif_match_count"], 1)
        self.assertEqual(recovery["matched_motif_ids"], ["M00"])

    def test_utility_selection_cost_is_nonzero_when_trials_run(self):
        cell = _fake_cell()

        self.assertGreater(
            cell["arms"]["utility_on_validation"]["selection_cost"][
                "frontier_candidates_tried_total"
            ],
            0,
        )

    def test_random_and_most_frequent_selection_cost_zero(self):
        cell = _fake_cell()

        self.assertEqual(
            cell["arms"]["most_frequent_k"]["selection_cost"][
                "selection_cost_candidate_programs_tried"
            ],
            0,
        )
        self.assertEqual(
            cell["arms"]["random_k"]["draws"][0]["selection_cost"][
                "selection_cost_candidate_programs_tried"
            ],
            0,
        )

    def test_break_even_positive_savings_is_ceiling(self):
        cell = _fake_cell()
        rows = full._break_even_rows([cell])

        self.assertEqual(rows[0]["utility_selection_cost"], 250)
        self.assertEqual(rows[0]["compression_selection_cost"], 100)
        self.assertEqual(rows[0]["incremental_selection_cost"], 150)
        self.assertEqual(rows[0]["break_even_future_tasks"], 15)

    def test_break_even_is_zero_when_utility_upfront_cost_is_not_greater(self):
        cell = _fake_cell()
        cell["arms"]["compression_on_validation_assisted"]["selection_cost"][
            "selection_cost_candidate_programs_tried"
        ] = 300

        row = full._break_even_rows([cell])[0]

        self.assertEqual(row["incremental_selection_cost"], 0)
        self.assertEqual(row["break_even_future_tasks"], 0)

    def test_break_even_none_when_no_savings(self):
        cell = _fake_cell()
        cell["arms"]["utility_on_validation"]["summary"]["mean_search_cost"] = 30
        rows = full._break_even_rows([cell])

        self.assertIsNone(rows[0]["break_even_future_tasks"])

    def test_canonical_solutions_use_one_solution_per_solved_task(self):
        solved = [object(), object()]
        results = [
            type(
                "Result",
                (),
                {"solutions": (solved[0], object()), "solved": True, "candidates_tried_total": 7},
            )(),
            type(
                "Result", (), {"solutions": (), "solved": False, "candidates_tried_total": 11}
            )(),
            type(
                "Result", (), {"solutions": (solved[1],), "solved": True, "candidates_tried_total": 13}
            )(),
        ]

        self.assertEqual(full._canonical_solutions_from_results(results), solved)
        self.assertEqual(
            full._solution_input_diagnostics(results),
            {
                "solved_task_count": 2,
                "solution_program_count_before_canonicalization": 3,
                "canonical_solution_count": 2,
                "candidate_programs_tried_total": 31,
            },
        )

    def test_solution_search_cost_is_added_to_compression_selection_cost(self):
        selection = full._static_selection(())
        results = [
            SimpleNamespace(candidates_tried_total=40),
            SimpleNamespace(candidates_tried_total=60),
        ]

        costed = full._selection_with_input_search_cost(selection, results)

        self.assertEqual(
            costed.cost["input_solution_search_candidate_programs_tried"], 100
        )
        self.assertEqual(costed.cost["selection_cost_candidate_programs_tried"], 100)

    def test_compression_inputs_report_canonical_counts(self):
        cell = _fake_cell()

        self.assertEqual(
            cell["compression_input_diagnostics"]["matched_25_starter"]["canonical_solution_count"],
            16,
        )
        self.assertEqual(
            cell["compression_input_diagnostics"]["matched_25_starter"][
                "solution_program_count_before_canonicalization"
            ],
            44,
        )
        self.assertEqual(
            cell["compression_input_diagnostics"]["validation_skip"]["canonical_solution_count"],
            13,
        )

    def test_random_draw_rows_include_validation_summaries(self):
        cell = _fake_cell()

        random_row = cell["arms"]["random_k"]["draws"][0]

        self.assertIsNotNone(random_row["validation_summary"])
        self.assertEqual(random_row["validation_summary"]["solved_count"], 1)

    def test_random_rows_are_in_validation_test_pairs(self):
        cell = _fake_cell()
        pairs = full._validation_test_prediction([cell])["pairs"]

        self.assertIn("random_00", [pair["arm"] for pair in pairs])

    def test_validation_test_pairs_use_gains_over_primitives(self):
        cell = _fake_cell()
        pairs = full._validation_test_pairs(cell["arms"], cell["arms"]["random_k"]["draws"])
        random_pair = next(pair for pair in pairs if pair["arm"] == "random_00")
        utility_pair = next(pair for pair in pairs if pair["arm"] == "utility_on_validation")

        self.assertEqual(random_pair["validation_solved_gain"], 0)
        self.assertEqual(random_pair["test_solved_gain"], 0)
        self.assertEqual(utility_pair["validation_solved_gain"], 1)
        self.assertEqual(utility_pair["test_solved_gain"], 2)

    def test_selector_module_does_not_import_hidden_motifs(self):
        source = Path("experiment/selection.py").read_text(encoding="utf-8")

        self.assertNotIn("Motif", source)
        self.assertNotIn("hidden_program", source)

    def test_full_runner_does_not_use_hidden_program_as_selector_input(self):
        source = Path("experiment/commands/run_full_selection_experiment.py").read_text(encoding="utf-8")

        self.assertNotIn("hidden_program", source)

    def test_nonformal_cache_skips_existing_cell_unless_force(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cell.json"
            path.write_text(
                json.dumps({"seed": 6460, "condition": "reversed_a0"}),
                encoding="utf-8",
            )
            job = {
                "cell_path": str(path),
                "force": False,
                "config_path": DEFAULT_CONFIG_PATH,
                "seed": 6460,
                "condition": "reversed_a0",
                "k": 1,
                "random_draws": 1,
                "selector_workers": 1,
            }

            self.assertEqual(full._run_or_load_cell(job)["condition"], "reversed_a0")

    def test_run_or_load_cell_rejects_custom_formal_seed_job_before_run_cell(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(full, "run_cell") as run_cell:
            with self.assertRaises(RuntimeError):
                full._run_or_load_cell(
                    {
                        "cell_path": str(Path(temp) / "6481_reversed_a0.json"),
                        "force": False,
                        "config_path": str(Path(temp) / "generator.yaml"),
                        "seed": 6481,
                        "condition": "reversed_a0",
                        "k": 1,
                        "random_draws": 1,
                        "selector_workers": 1,
                    }
                )

        run_cell.assert_not_called()

    def test_run_or_load_cell_rejects_numeric_string_seed_before_run_cell(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
            full, "run_cell"
        ) as run_cell, patch.object(full, "_write_json"):
            with self.assertRaisesRegex(TypeError, "must be an exact integer"):
                full._run_or_load_cell(
                    {
                        "cell_path": str(Path(temp) / "cell.json"),
                        "force": False,
                        "config_path": DEFAULT_CONFIG_PATH,
                        "seed": "6481",
                        "condition": "reversed_a0",
                        "k": 1,
                        "random_draws": 1,
                        "selector_workers": 1,
                        "experiment_name": full.SMOKE_EXPERIMENT_NAME,
                    }
                )

        run_cell.assert_not_called()

    def test_numeric_string_seed_rejected_by_cell_compute_boundaries(self):
        with patch.object(full, "_compute_cell") as compute_cell:
            with self.assertRaisesRegex(TypeError, "must be an exact integer"):
                full.run_cell(
                    config_path=DEFAULT_CONFIG_PATH,
                    seed="6481",
                    condition_name="reversed_a0",
                    k=1,
                    random_draws=1,
                    selector_workers=1,
                )
        compute_cell.assert_not_called()

        with patch.object(full, "_compute_cell_body") as compute_cell_body:
            with self.assertRaisesRegex(TypeError, "must be an exact integer"):
                full._compute_cell(
                    config_path=DEFAULT_CONFIG_PATH,
                    seed="6481",
                    condition_name="reversed_a0",
                    k=1,
                    random_draws=1,
                    selector_workers=1,
                )
        compute_cell_body.assert_not_called()

        with patch.object(full, "load_config") as load_config:
            with self.assertRaisesRegex(TypeError, "must be an exact integer"):
                full._compute_cell_body(
                    config_path=DEFAULT_CONFIG_PATH,
                    seed="6481",
                    condition_name="reversed_a0",
                    k=1,
                    random_draws=1,
                    selector_workers=1,
                    experiment_name=full.SMOKE_EXPERIMENT_NAME,
                )
        load_config.assert_not_called()

    def test_compute_cell_from_world_rejects_numeric_string_world_seed(self):
        world = SimpleNamespace(world_seed="6481", tasks_start=())
        with patch.object(
            full, "frontier_promotion_menu", side_effect=TypeError("frontier called")
        ) as frontier_promotion_menu:
            with self.assertRaisesRegex(TypeError, "must be an exact integer"):
                full._compute_cell_from_world(
                    world=world,
                    seed=6460,
                    condition_name="reversed_a0",
                    k=1,
                    random_draws=1,
                    selector_workers=1,
                    experiment_name=full.SMOKE_EXPERIMENT_NAME,
                )
        frontier_promotion_menu.assert_not_called()

    def test_numeric_string_seed_rejected_by_run_and_job_builders(self):
        with patch.object(full, "load_config") as load_config:
            with self.assertRaises(Exception) as error:
                full._run_cells(
                    config_path=DEFAULT_CONFIG_PATH,
                    output_path=Path(full.DEFAULT_OUTPUT_PATH),
                    seeds=("6481",),
                    conditions=full.CONDITIONS,
                    k=full.DEFAULT_K,
                    random_draws=full.RANDOM_DRAWS,
                    workers=1,
                    force=False,
                    smoke=False,
                )
            self.assertIs(type(error.exception), TypeError)
            self.assertRegex(str(error.exception), "must be an exact integer")
        load_config.assert_not_called()

        with self.assertRaisesRegex(TypeError, "must be an exact integer"):
            full.build_cell_jobs(
                config_path=DEFAULT_CONFIG_PATH,
                cell_dir=Path("cells"),
                seeds=("6481",),
                conditions=("reversed_a0",),
                k=1,
                random_draws=1,
                workers=1,
                force=False,
            )

        with self.assertRaisesRegex(TypeError, "must be an exact integer"):
            full.matched_starter_tasks((), "6481", 0)

        with self.assertRaisesRegex(TypeError, "must be an exact integer"):
            full._validate_formal_cell_job({"seed": "6481"})

        with self.assertRaisesRegex(TypeError, "must be an exact integer"):
            full._cell_path(Path("cells"), "6481", "reversed_a0")

        with self.assertRaisesRegex(TypeError, "must be an exact integer"):
            full._registration(("6481",), ("reversed_a0",), 1, 1)

    def test_run_cell_rejects_direct_formal_seed_before_loading_config(self):
        with patch.object(full, "load_config") as load_config:
            with self.assertRaises(RuntimeError):
                full.run_cell(
                    config_path="custom.yaml",
                    seed=6481,
                    condition_name="reversed_a0",
                    k=1,
                    random_draws=1,
                    selector_workers=1,
                    experiment_name=full.EXPERIMENT_NAME,
                )

        load_config.assert_not_called()

    def test_run_cell_rejects_tokened_custom_formal_seed_shape_before_loading_config(self):
        with patch.object(full, "load_config") as load_config:
            with self.assertRaises(TypeError):
                full.run_cell(
                    config_path="custom.yaml",
                    seed=6481,
                    condition_name="reversed_a0",
                    k=1,
                    random_draws=1,
                    selector_workers=1,
                    _formal_seed_token=object(),
                )

        load_config.assert_not_called()

    def test_compute_cell_rejects_direct_formal_seed_before_loading_config(self):
        with patch.object(full, "load_config") as load_config:
            with self.assertRaises(RuntimeError):
                full._compute_cell(
                    config_path=DEFAULT_CONFIG_PATH,
                    seed=6481,
                    condition_name="reversed_a0",
                    k=10,
                    random_draws=20,
                    selector_workers=1,
                )

        load_config.assert_not_called()

    def test_compute_cell_body_rejects_direct_formal_seed_before_loading_config(self):
        with patch.object(full, "load_config") as load_config:
            with self.assertRaises(RuntimeError):
                full._compute_cell_body(
                    config_path="custom.yaml",
                    seed=6481,
                    condition_name="reversed_a0",
                    k=1,
                    random_draws=1,
                    selector_workers=1,
                    experiment_name=full.EXPERIMENT_NAME,
                )

        load_config.assert_not_called()

    def test_compute_cell_from_world_rejects_direct_formal_seed(self):
        with patch.object(full, "frontier_promotion_menu") as frontier_promotion_menu:
            with self.assertRaises(RuntimeError):
                full._compute_cell_from_world(
                    world=object(),
                    seed=6481,
                    condition_name="reversed_a0",
                    k=1,
                    random_draws=1,
                    selector_workers=1,
                    experiment_name=full.EXPERIMENT_NAME,
                )

        frontier_promotion_menu.assert_not_called()

    def test_compute_cell_from_world_rejects_mislabeled_formal_world(self):
        world = SimpleNamespace(world_seed=6481)
        with patch.object(full, "frontier_promotion_menu") as frontier_promotion_menu:
            with self.assertRaises(RuntimeError):
                full._compute_cell_from_world(
                    world=world,
                    seed=6460,
                    condition_name="reversed_a0",
                    k=1,
                    random_draws=1,
                    selector_workers=1,
                    experiment_name=full.SMOKE_EXPERIMENT_NAME,
                )

        frontier_promotion_menu.assert_not_called()

    def test_progress_flushes_immediately(self):
        with patch("builtins.print") as print_line:
            full._progress("CELL START")

        print_line.assert_called_once_with("CELL START", flush=True)

    def test_progress_duration_uses_hours_minutes_seconds(self):
        self.assertEqual(full._format_duration(3661.4), "01:01:01")

    def test_registered_formal_path_uses_exact_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            other_cwd = root / "other_cwd"
            other_cwd.mkdir()
            config = SimpleNamespace(
                conditions=[SimpleNamespace(name=name) for name in full.CONDITIONS]
            )
            tasks_start = [
                SimpleNamespace(id=f"start_{index:03d}", target=f"start_grid_{index}")
                for index in range(100)
            ]
            fake_world = SimpleNamespace(
                tasks_start=tasks_start,
                tasks_val=[SimpleNamespace(id="val_000", target="val_grid")],
                tasks_test=[SimpleNamespace(id="test_000", target="test_grid")],
                motifs=(),
                metadata={
                    "realized_rho": {"realized_start_test": 0.0},
                    "density_summary": {},
                    "expected_motif_length": 0,
                },
            )
            selected_candidates = tuple(
                SimpleNamespace(
                    program=None,
                    program_string=f"candidate_{index:02d}",
                    output=f"candidate_grid_{index:02d}",
                )
                for index in range(full.DEFAULT_K)
            )
            exact_selection = full._static_selection(selected_candidates)
            empty_results = [
                SimpleNamespace(solved=False, solutions=(), candidates_tried_total=0)
            ]
            summary = {
                "solved_count": 0,
                "task_count": 1,
                "solve_rate": 0,
                "mean_search_cost": 1,
                "mean_first_solution_cost": None,
                "frontier_candidates_tried_total": 1,
                "hit_budget": False,
                "unique_outputs": 1,
            }

            def make_world_after_claim(*_args, **_kwargs):
                self.assertTrue((root / full.DEFAULT_OUTPUT_PATH).with_suffix("").is_dir())
                return fake_world

            cwd = Path.cwd()
            try:
                import os

                os.chdir(other_cwd)
                with patch.object(full, "REPO_ROOT", root), patch.object(
                    full, "load_config", return_value=config
                ) as load_config, patch.object(
                    full, "make_world", side_effect=make_world_after_claim
                ) as make_world, patch.object(
                    full, "_compute_cell_from_world", side_effect=AssertionError("formal bypass")
                ), patch.object(
                    full,
                    "frontier_promotion_menu",
                    return_value=SimpleNamespace(candidates=selected_candidates),
                ), patch.object(full, "menu_diagnostics", return_value={}), patch.object(
                    full, "solve_tasks", return_value=empty_results
                ), patch.object(
                    full, "select_most_frequent_k", return_value=selected_candidates
                ), patch.object(
                    full, "select_random_k", return_value=selected_candidates
                ), patch.object(
                    full, "select_compression_k_with_cost", return_value=exact_selection
                ), patch.object(
                    full, "greedy_by_frontier_score_with_cost", return_value=exact_selection
                ), patch.object(
                    full, "greedy_by_solved_count_with_cost", return_value=exact_selection
                ), patch.object(
                    full, "solve_library_summary", return_value=summary
                ), patch.object(full, "_progress") as progress:
                    payload = full.run_formal_experiment(workers=999)
            finally:
                os.chdir(cwd)

            self.assertFalse(payload["smoke"])
            self.assertEqual(payload["experiment_name"], full.EXPERIMENT_NAME)
            self.assertEqual(load_config.call_args.args[0], str(root / DEFAULT_CONFIG_PATH))
            self.assertEqual(make_world.call_count, len(full.FORMAL_SEEDS) * len(full.CONDITIONS))
            for cell in payload["cells"]:
                self.assertEqual(cell["k"], full.DEFAULT_K)
                self.assertEqual(cell["random_draws"], full.RANDOM_DRAWS)
                self.assertEqual(cell["experiment_name"], full.EXPERIMENT_NAME)
                self.assertTrue(cell["formal_seed"])
                for name, row in cell["arms"].items():
                    if name == "primitives_only":
                        self.assertEqual(row["selected_programs"], [])
                    elif name == "random_k":
                        self.assertTrue(
                            all(
                                len(draw["selected_programs"]) == full.DEFAULT_K
                                for draw in row["draws"]
                            )
                        )
                    else:
                        self.assertEqual(len(row["selected_programs"]), full.DEFAULT_K)
            self.assertTrue((root / full.DEFAULT_OUTPUT_PATH).exists())
            self.assertFalse((other_cwd / full.DEFAULT_OUTPUT_PATH).exists())
            messages = [call.args[0] for call in progress.call_args_list]
            self.assertTrue(messages[0].startswith("FORMAL START cells=210 workers=999"))
            self.assertTrue(
                messages[1].startswith(
                    "CELL START 001/210 seed=6481 condition=reversed_a0"
                )
            )
            self.assertTrue(
                messages[2].startswith(
                    "CELL DONE 001/210 seed=6481 condition=reversed_a0"
                )
            )
            self.assertTrue(messages[-1].startswith("AGGREGATE START cells=210"))
            self.assertFalse(
                any(
                    "solved_count" in message or "realized_rho" in message
                    for message in messages
                )
            )

    def test_force_rejects_formal_seeds(self):
        with self.assertRaises(RuntimeError):
            full._run_or_load_cell(
                {
                    "cell_path": str(Path(full.DEFAULT_OUTPUT_PATH).with_suffix("") / "cells/6481_reversed_a0.json"),
                    "force": True,
                    "config_path": DEFAULT_CONFIG_PATH,
                    "seed": 6481,
                    "condition": "reversed_a0",
                    "k": 10,
                    "random_draws": 20,
                    "selector_workers": 1,
                }
            )

    def test_same_cell_is_deterministic_except_wall_clock(self):
        first = _fake_cell()
        second = _fake_cell()
        first.pop("wall_clock_seconds")
        second.pop("wall_clock_seconds")

        self.assertEqual(first, second)

    def test_worker_policy_disables_selector_workers_when_parallel(self):
        jobs = full.build_cell_jobs(
            config_path=DEFAULT_CONFIG_PATH,
            cell_dir=Path("cells"),
            seeds=(6460,),
            conditions=("reversed_a0",),
            k=1,
            random_draws=1,
            workers=6,
            force=True,
        )

        self.assertEqual(jobs[0]["selector_workers"], 1)

    def test_command_smoke_writes_json(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "smoke.json"
            with patch.object(full, "SMOKE_OUTPUT_PATH", str(output)), patch.object(
                full, "run_cell", return_value=_fake_cell(experiment_name=full.SMOKE_EXPERIMENT_NAME)
            ):
                full.main(["--smoke", "--workers", "1"])

            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(data["smoke"])
            self.assertEqual(data["experiment_name"], full.SMOKE_EXPERIMENT_NAME)
            self.assertEqual(data["registration"]["experiment_name"], full.SMOKE_EXPERIMENT_NAME)
            self.assertEqual(data["registration"]["seeds"], [6460])

    def test_real_smoke_cell_uses_spent_seed(self):
        cell = full.run_cell(
            config_path=DEFAULT_CONFIG_PATH,
            seed=6460,
            condition_name="reversed_a0",
            k=1,
            random_draws=1,
            selector_workers=1,
        )

        self.assertEqual(cell["seed"], 6460)
        self.assertIn("utility_on_validation", cell["arms"])
        self.assertGreater(
            cell["arms"]["utility_on_validation"]["selection_cost"][
                "frontier_candidates_tried_total"
            ],
            0,
        )
        assisted_cost = cell["arms"]["compression_on_validation_assisted"][
            "selection_cost"
        ]
        assisted_diagnostic = cell["compression_input_diagnostics"][
            "validation_assisted"
        ]["candidate_programs_tried_total"]
        self.assertGreater(assisted_diagnostic, 0)
        self.assertEqual(
            assisted_cost["input_solution_search_candidate_programs_tried"],
            assisted_diagnostic,
        )
        self.assertEqual(
            assisted_cost["selection_cost_candidate_programs_tried"],
            assisted_diagnostic,
        )


def _fake_cell(seed=6460, formal_seed=False, k=1, random_draws=1, experiment_name=None):
    def summary(solved=1, mean=20):
        return {
            "solved_count": solved,
            "task_count": 2,
            "solve_rate": solved / 2,
            "mean_search_cost": mean,
            "mean_first_solution_cost": mean if solved else None,
            "frontier_candidates_tried_total": mean * 2,
            "hit_budget": False,
            "unique_outputs": 10,
        }

    arms = {
        name: {
            "arm": name,
            "summary": summary(1, 20),
            "validation_summary": summary(1, 21),
            "selected_programs": (
                []
                if name == "primitives_only"
                else [f"{name}_program_{index:02d}" for index in range(k)]
            ),
            "selection_cost": full._zero_selection_cost(),
            "motif_recovery": {
                "selected_count": 0 if name == "primitives_only" else 1,
                "motif_match_count": 0,
                "precision": 0,
                "recall": 0,
                "matched_motif_ids": [],
            },
        }
        for name in (
            "primitives_only",
            "most_frequent_k",
            "compression_on_matched_25_starter",
            "utility_on_matched_25_starter",
            "compression_on_validation_skip",
            "compression_on_validation_assisted",
            "utility_on_validation",
            "compression_on_all_100_starter",
            "best_k_from_c_oracle",
            "hidden_motif_oracle",
        )
    }
    arms["utility_on_validation"]["summary"]["mean_search_cost"] = 10
    arms["utility_on_validation"]["summary"]["solved_count"] = 3
    arms["utility_on_validation"]["validation_summary"]["solved_count"] = 2
    arms["utility_on_validation"]["selection_cost"] = {
        **full._zero_selection_cost(),
        "selection_cost_candidate_programs_tried": 250,
        "trial_libraries_evaluated": 5,
        "frontier_candidates_tried_total": 250,
    }
    arms["compression_on_validation_assisted"]["selection_cost"] = {
        **full._zero_selection_cost(),
        "input_solution_search_candidate_programs_tried": 100,
        "selection_cost_candidate_programs_tried": 100,
    }
    arms["random_k"] = {
        "draws": [
            {
                "arm": f"random_{index:02d}",
                "summary": summary(1, 20),
                "validation_summary": summary(1, 21),
                "selected_programs": [
                    f"random_{index:02d}_program_{candidate_index:02d}"
                    for candidate_index in range(k)
                ],
                "selection_cost": full._zero_selection_cost(),
                "motif_recovery": {
                    "selected_count": 1,
                    "motif_match_count": 0,
                    "precision": 0,
                    "recall": 0,
                    "matched_motif_ids": [],
                },
            }
            for index in range(random_draws)
        ],
        "median_solved_count": 1,
    }
    return {
        "experiment_name": experiment_name or full.EXPERIMENT_NAME,
        "seed": seed,
        "condition": "reversed_a0",
        "formal_seed": formal_seed,
        "k": k,
        "random_draws": random_draws,
        "world_metadata": {"realized_rho": {"realized_start_test": 0.0}},
        "menu": {"kept_candidate_count": 1},
        "matched_starter_task_ids": ["start_000"],
        "matched_starter_solved_count": 1,
        "validation_skip_solved_count": 1,
        "compression_input_diagnostics": {
            "matched_25_starter": {
                "solved_task_count": 16,
                "solution_program_count_before_canonicalization": 44,
                "canonical_solution_count": 16,
                "candidate_programs_tried_total": 400,
            },
            "validation_skip": {
                "solved_task_count": 13,
                "solution_program_count_before_canonicalization": 25,
                "canonical_solution_count": 13,
                "candidate_programs_tried_total": 80,
            },
            "validation_assisted": {
                "solved_task_count": 25,
                "solution_program_count_before_canonicalization": 25,
                "canonical_solution_count": 25,
                "candidate_programs_tried_total": 100,
            },
            "all_100_starter": {
                "solved_task_count": 60,
                "solution_program_count_before_canonicalization": 120,
                "canonical_solution_count": 60,
                "candidate_programs_tried_total": 1200,
            },
        },
        "arms": arms,
        "validation_test_pairs": [
            {
                "arm": "random_00",
                "validation_solved_count": 1,
                "test_solved_count": 1,
                "validation_solved_gain": 0,
                "test_solved_gain": 0,
            },
            {
                "arm": "utility_on_validation",
                "validation_solved_count": 2,
                "test_solved_count": 3,
                "validation_solved_gain": 1,
                "test_solved_gain": 2,
            },
            {
                "arm": "compression_on_validation_assisted",
                "validation_solved_count": 0,
                "test_solved_count": 1,
                "validation_solved_gain": -1,
                "test_solved_gain": 0,
            },
        ],
        "selected_set_overlap": {},
        "capture_ratio": 0.5,
        "c_delta": 1,
        "motif_oracle_delta": 2,
        "wall_clock_seconds": 0.1,
    }


def _formal_job():
    return {
        "seed": 6481,
        "condition": "reversed_a0",
        "k": 10,
        "random_draws": 20,
        "experiment_name": full.EXPERIMENT_NAME,
    }


def _fresh_guard(root: Path) -> None:
    with patch.object(full, "REPO_ROOT", root):
        full._fresh_look_guard(root / "experiment/data/selection/full_selection_experiment/cells")


if __name__ == "__main__":
    unittest.main()
