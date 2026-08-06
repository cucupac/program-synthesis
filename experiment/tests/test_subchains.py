import json
import tempfile
import unittest
from pathlib import Path

from experiment.commands.extract_subchains import main as extract_main
from experiment.dsl import call, primitive, program_to_string
from experiment.subchains import extract_candidate_menu, program_subtrees


def supported_program():
    return call(
        "invert",
        call("add", primitive("line_horizontal"), primitive("line_vertical")),
    )


class SubchainTests(unittest.TestCase):
    def test_extracts_subtrees_from_composed_program(self):
        program = call(
            "add",
            call("invert", primitive("square")),
            call("overlap", primitive("line_horizontal"), primitive("line_vertical")),
        )

        texts = [program_to_string(subtree) for subtree in program_subtrees(program)]

        self.assertEqual(texts[0], program_to_string(program))
        self.assertIn("invert(square)", texts)
        self.assertIn("overlap(line_horizontal,line_vertical)", texts)
        self.assertIn("line_horizontal", texts)

    def test_excludes_candidates_below_min_op_count(self):
        menu = extract_candidate_menu(
            [("s1", call("invert", primitive("square")))],
            min_op_count=2,
            min_support=1,
        )

        self.assertEqual(menu.raw_candidate_count, 0)
        self.assertEqual(menu.candidates, ())

    def test_deduplicates_candidates_by_program_string(self):
        program = supported_program()

        menu = extract_candidate_menu(
            [("s1", program), ("s2", program)],
            min_op_count=2,
            min_support=1,
        )

        self.assertEqual(len(menu.candidates), 1)
        self.assertEqual(menu.candidates[0].program_string, program_to_string(program))
        self.assertEqual(menu.candidates[0].support_count, 2)
        self.assertEqual(menu.dedupe_by, "program_string")

    def test_counts_distinct_solution_support(self):
        candidate = supported_program()
        duplicated = call("add", candidate, candidate)

        menu = extract_candidate_menu(
            [("s1", duplicated), ("s2", candidate)],
            min_op_count=2,
            min_support=1,
        )

        by_string = {candidate.program_string: candidate for candidate in menu.candidates}
        supported = by_string[program_to_string(candidate)]
        self.assertEqual(supported.support_count, 2)
        self.assertEqual(supported.solution_ids, ("s1", "s2"))

    def test_filters_candidates_by_minimum_support(self):
        shared = supported_program()
        rare = call(
            "invert",
            call("subtract", primitive("square"), primitive("triangle")),
        )

        menu = extract_candidate_menu(
            [("s1", shared), ("s2", shared), ("s3", rare)],
            min_op_count=2,
            min_support=2,
        )

        strings = {candidate.program_string for candidate in menu.candidates}
        self.assertIn(program_to_string(shared), strings)
        self.assertNotIn(program_to_string(rare), strings)

    def test_command_runs_and_writes_json(self):
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "generator.yaml"
            output_path = Path(temp) / "candidates.json"
            config_path.write_text(
                """
output_dir: experiment/data/problem_sets/test_subchains
world_seeds: [6460]

sizes:
  n_start: 2
  n_val: 1
  n_test: 1

motifs:
  count: 12
  min_ops: 2
  max_ops: 4
  rarity_floor: 0.02

tasks:
  motifs_per_task_pattern: [2, 3]
  glue_ops_per_task_pattern: [0, 1, 2]
  sample_motifs_with_replacement: false
  min_filled_cells: 3
  max_filled_cells: 85
  max_rejection_attempts: 1000

conditions:
  - name: reversed_a0
    alt_kind: reversed
    alpha_val: 0.0
    alpha_test: 0.0
""",
                encoding="utf-8",
            )

            extract_main(["--config", str(config_path), "--output", str(output_path)])

            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(data["world_seed"], 6460)
            self.assertEqual(data["condition"], "reversed_a0")
            self.assertEqual(data["starter_task_count"], 2)
            self.assertEqual(data["extraction"]["dedupe_by"], "program_string")
            self.assertIn("candidate_size_distribution", data)
            self.assertIn("motif_coverage", data)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
