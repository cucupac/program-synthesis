import unittest
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiment.commands import run_k_sweep_experiment as sweep
from experiment.commands import run_full_selection_experiment as full


class KSweepExperimentTests(unittest.TestCase):
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
                sweep.main([option, "x"])

    def test_formal_jobs_pair_every_world_across_registered_k_values(self):
        jobs = sweep.build_jobs(Path("cells"), workers=1)

        self.assertEqual(len(jobs), 540)
        observed = defaultdict(set)
        for job in jobs:
            observed[(job["seed"], job["condition"])].add(job["k"])
            self.assertIs(job["formal_seed"], True)
            self.assertEqual(
                Path(job["cell_path"]),
                Path("cells") / f"k{job['k']}" / f"{job['seed']}_{job['condition']}.json",
            )

        self.assertEqual(len(observed), 180)
        self.assertTrue(all(values == {2, 5, 10} for values in observed.values()))
        self.assertNotIn("stale_reversed", {job["condition"] for job in jobs})

    def test_smoke_jobs_use_one_spent_world_at_all_registered_k_values(self):
        jobs = sweep.build_jobs(Path("cells"), workers=2, smoke=True)

        self.assertEqual(len(jobs), 3)
        self.assertEqual({job["seed"] for job in jobs}, {6460})
        self.assertEqual({job["condition"] for job in jobs}, {"reversed_a0"})
        self.assertEqual({job["k"] for job in jobs}, {2, 5, 10})
        self.assertEqual({job["random_draws"] for job in jobs}, {1})
        self.assertEqual({job["formal_seed"] for job in jobs}, {False})
        self.assertEqual({job["selector_workers"] for job in jobs}, {1})
        self.assertEqual(
            {job["experiment_name"] for job in jobs},
            {sweep.SMOKE_EXPERIMENT_NAME},
        )

    def test_smoke_registration_describes_its_single_random_draw(self):
        registration = sweep._registration(smoke=True)

        self.assertEqual(registration["random_draws"], 1)
        self.assertEqual(
            registration["arms"]["random_k"],
            "1 deterministic random K draw from C",
        )

    def test_validate_cell_accepts_exact_registered_shape(self):
        job = sweep.build_jobs(Path("cells"), workers=1)[0]
        cell = _fake_cell(job)

        sweep.validate_cell(cell, job)

    def test_validate_cell_rejects_non_integer_seed_and_incorrect_metadata(self):
        job = sweep.build_jobs(Path("cells"), workers=1)[0]
        for field, value in (
            ("seed", str(job["seed"])),
            ("condition", "stale_reversed"),
            ("k", 5),
            ("random_draws", job["random_draws"] - 1),
            ("experiment_name", "other"),
        ):
            with self.subTest(field=field):
                cell = _fake_cell(job)
                cell[field] = value
                with self.assertRaisesRegex(RuntimeError, field):
                    sweep.validate_cell(cell, job)

    def test_validate_cell_rejects_incorrect_formal_seed_metadata(self):
        job = sweep.build_jobs(Path("cells"), workers=1)[0]
        job["formal_seed"] = True
        cell = _fake_cell(job)
        cell["formal_seed"] = False

        with self.assertRaisesRegex(RuntimeError, "formal_seed"):
            sweep.validate_cell(cell, job)

    def test_validate_cell_requires_exact_k_for_real_and_random_libraries(self):
        job = sweep.build_jobs(Path("cells"), workers=1)[0]
        real_cell = _fake_cell(job)
        real_cell["arms"]["compression_on_matched_25_starter"]["selected_programs"].pop()
        with self.assertRaisesRegex(RuntimeError, "exactly K"):
            sweep.validate_cell(real_cell, job)

        random_cell = _fake_cell(job)
        random_cell["arms"]["random_k"]["draws"][0]["selected_programs"].pop()
        with self.assertRaisesRegex(RuntimeError, "exactly K"):
            sweep.validate_cell(random_cell, job)

    def test_validate_cell_rejects_self_reported_hidden_motif_capacity(self):
        job = sweep.build_jobs(Path("cells"), workers=1)[0]
        cell = _fake_cell(job)
        cell["arms"]["hidden_motif_oracle"]["diagnostic_candidate_count"] = 12

        with self.assertRaisesRegex(RuntimeError, "hidden-motif capacity metadata"):
            sweep.validate_cell(cell, job)

    def test_atomic_json_write_leaves_only_complete_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "cell.json"

            sweep.write_json_atomic(path, {"complete": True})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"complete": True})
            self.assertFalse(path.with_name(f".{path.name}.tmp").exists())

    def test_formal_output_directory_can_be_claimed_only_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "selection" / "k_sweep"

            sweep.claim_formal_output(root)

            self.assertTrue(root.is_dir())
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                sweep.claim_formal_output(root)

    def test_formal_claim_rejects_malformed_artifact_named_for_a_sweep_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            selection = Path(temp) / "selection"
            artifact = selection / "other" / "6511_reversed_a0.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{broken", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "fresh-look"):
                sweep.claim_formal_output(selection / "k_sweep")

    def test_completed_smoke_artifacts_do_not_block_the_formal_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            selection = Path(temp) / "selection"
            smoke = selection / "k_sweep_smoke" / "registration.json"
            smoke.parent.mkdir(parents=True)
            smoke.write_text(
                json.dumps(
                    {
                        "experiment_name": sweep.SMOKE_EXPERIMENT_NAME,
                        "seeds": [6460],
                        "k_values": [2, 5, 10],
                    }
                ),
                encoding="utf-8",
            )
            (smoke.parent / f"{sweep.SMOKE_EXPERIMENT_NAME}.json").write_text(
                json.dumps(
                    {
                        "experiment_name": sweep.SMOKE_EXPERIMENT_NAME,
                        "registration": {
                            "experiment_name": sweep.SMOKE_EXPERIMENT_NAME,
                            "seeds": [6460],
                        },
                        "cells": [],
                    }
                ),
                encoding="utf-8",
            )

            sweep.claim_formal_output(selection / "k_sweep")

            self.assertTrue((selection / "k_sweep").is_dir())

    def test_formal_claim_rejects_non_object_artifact_in_formal_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            selection = Path(temp) / "selection"
            artifact = selection / "k_sweep" / "registration.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "fresh-look"):
                sweep.claim_formal_output(selection / "new_k_sweep")

    def test_run_job_uses_existing_cell_engine_and_writes_validated_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            job = sweep.build_jobs(Path(temp) / "cells", workers=2, smoke=True)[0]
            cell = _fake_cell(job)
            output = io.StringIO()

            with patch.object(sweep, "run_cell", return_value=cell) as run_cell, redirect_stdout(output):
                completed, elapsed = sweep.run_job(job)

            self.assertEqual(completed, cell)
            self.assertGreaterEqual(elapsed, 0)
            self.assertIn(
                "CELL START 001/003 k=2 seed=6460 condition=reversed_a0",
                output.getvalue(),
            )
            self.assertEqual(
                json.loads(Path(job["cell_path"]).read_text(encoding="utf-8")),
                cell,
            )
            run_cell.assert_called_once_with(
                config_path=full.DEFAULT_CONFIG_PATH,
                seed=job["seed"],
                condition_name=job["condition"],
                k=job["k"],
                random_draws=job["random_draws"],
                selector_workers=job["selector_workers"],
                experiment_name=job["experiment_name"],
            )

    def test_run_job_records_registered_sweep_seeds_as_formal(self):
        with tempfile.TemporaryDirectory() as temp:
            job = sweep.build_jobs(Path(temp) / "cells", workers=1)[0]
            cell = _fake_cell(job)
            cell["formal_seed"] = False

            with patch.object(sweep, "run_cell", return_value=cell), redirect_stdout(
                io.StringIO()
            ):
                completed, _ = sweep.run_job(job)

            self.assertIs(completed["formal_seed"], True)
            written = json.loads(Path(job["cell_path"]).read_text(encoding="utf-8"))
            self.assertIs(written["formal_seed"], True)

    def test_formal_run_writes_registration_and_540_raw_cells(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(
                sweep, "REPO_ROOT", Path(temp)
            ), patch.object(
                sweep,
                "run_job",
                side_effect=lambda job: (_minimal_cell(job), 0.0),
            ), redirect_stdout(output):
                payload = sweep.run_formal_experiment(workers=1)

            self.assertEqual(len(payload["cells"]), 540)
            self.assertNotIn("aggregates", payload)
            self.assertEqual(payload["registration"]["seeds"], list(range(6511, 6541)))
            self.assertEqual(payload["registration"]["k_values"], [2, 5, 10])
            self.assertEqual(len(payload["registration"]["conditions"]), 6)
            self.assertEqual(payload["registration"]["validation_task_count"], 25)
            self.assertEqual(payload["registration"]["random_draws"], 20)
            self.assertEqual(
                payload["registration"]["assisted_validation_solve_config"]["node_budget"],
                90_000,
            )
            self.assertEqual(payload["registration"]["arms"], full._arm_definitions())
            output_path = Path(temp) / sweep.OUTPUT_PATH
            registration = Path(temp) / sweep.OUTPUT_ROOT / "registration.json"
            self.assertTrue(output_path.exists())
            self.assertTrue(registration.exists())
            self.assertEqual(
                [(cell["k"], cell["seed"], cell["condition"]) for cell in payload["cells"]],
                sorted(
                    (cell["k"], cell["seed"], cell["condition"])
                    for cell in payload["cells"]
                ),
            )
        self.assertIn("K SWEEP START cells=540 workers=1", output.getvalue())
        self.assertIn("AGGREGATE START cells=540", output.getvalue())

    def test_smoke_run_writes_three_raw_cells_without_claiming_formal_output(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
            sweep, "REPO_ROOT", Path(temp)
        ), patch.object(
            sweep,
            "run_job",
            side_effect=lambda job: (_minimal_cell(job), 0.0),
        ), redirect_stdout(io.StringIO()):
            payload = sweep.run_smoke_experiment(workers=1)

            self.assertTrue(payload["smoke"])
            self.assertEqual(len(payload["cells"]), 3)
            self.assertTrue((Path(temp) / sweep.SMOKE_OUTPUT_PATH).exists())
            self.assertFalse((Path(temp) / sweep.OUTPUT_ROOT).exists())

    def test_multiple_workers_use_one_process_per_cell(self):
        job = sweep.build_jobs(Path("cells"), workers=2, smoke=True)[0]
        cell = _minimal_cell(job)
        with patch.object(sweep, "ProcessPoolExecutor") as executor, redirect_stdout(
            io.StringIO()
        ):
            executor.return_value.__enter__.return_value.map.return_value = [(cell, 0.0)]

            cells = sweep.execute_jobs([job], workers=2)

        self.assertEqual(cells, [cell])
        executor.assert_called_once_with(max_workers=2)
        executor.return_value.__enter__.return_value.map.assert_called_once_with(
            sweep.run_job,
            [job],
        )

    def test_job_progress_reports_identity_elapsed_time_and_eta(self):
        job = sweep.build_jobs(Path("cells"), workers=1, smoke=True)[0]
        cell = _minimal_cell(job)
        output = io.StringIO()
        with patch.object(sweep, "run_job", return_value=(cell, 1.0)), redirect_stdout(output):
            sweep.execute_jobs([job], workers=1)

        message = output.getvalue()
        self.assertIn("CELL DONE 001/001", message)
        self.assertIn("k=2 seed=6460 condition=reversed_a0", message)
        self.assertIn("elapsed=00:00:01", message)
        self.assertIn("eta=00:00:00", message)


def _fake_cell(job):
    selected = [f"program_{index}" for index in range(job["k"])]
    arms = {
        name: {"selected_programs": list(selected)}
        for name in full._arm_definitions()
        if name not in {"primitives_only", "random_k"}
    }
    arms["primitives_only"] = {"selected_programs": []}
    arms["random_k"] = {
        "draws": [
            {"selected_programs": list(selected)}
            for _ in range(job["random_draws"])
        ]
    }
    return {
        "experiment_name": job["experiment_name"],
        "seed": job["seed"],
        "condition": job["condition"],
        "formal_seed": job["formal_seed"],
        "k": job["k"],
        "random_draws": job["random_draws"],
        "arms": arms,
    }


def _minimal_cell(job):
    return {
        "experiment_name": job["experiment_name"],
        "seed": job["seed"],
        "condition": job["condition"],
        "formal_seed": job["formal_seed"],
        "k": job["k"],
        "random_draws": job["random_draws"],
    }


if __name__ == "__main__":
    unittest.main()
