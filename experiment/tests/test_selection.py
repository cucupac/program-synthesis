import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment.commands.run_selection import _fresh_look_guard, main as selection_main
from experiment.dsl import call, execute, primitive
from experiment.frontier_promotion import FrontierCandidate, frontier_promotion_menu
from experiment.generator import Task
from experiment.selection import (
    compression_score,
    greedy_by_frontier_score,
    greedy_by_frontier_score_with_cost,
    greedy_by_solved_count_with_cost,
    select_compression_k_with_cost,
    select_most_frequent_k,
    select_random_k,
    select_utility_k,
    solve_library_summaries,
)
from experiment.solver import (
    LibraryItem,
    SolveConfig,
    build_frontier_index,
    primitive_library,
    solve_task,
)


def candidate(program, support=("start_000",), first_hit_cost=1):
    return FrontierCandidate(
        program=program,
        program_string=f"C:{program}",
        output=execute(program),
        op_count=1,
        support_task_ids=tuple(support),
        first_hit_cost=first_hit_cost,
    )


class SelectionTests(unittest.TestCase):
    def test_multiple_summary_groups_share_one_frontier_index(self):
        library = primitive_library()
        targets = (
            [execute(primitive("square"))],
            [execute(primitive("triangle"))],
        )

        with patch(
            "experiment.selection.build_frontier_index",
            wraps=build_frontier_index,
        ) as build:
            summaries = solve_library_summaries(targets, library)

        self.assertEqual(build.call_count, 1)
        self.assertEqual([row["task_count"] for row in summaries], [1, 1])

    def test_frontier_index_matches_solve_task_with_helpers(self):
        helper_program = call("add", primitive("line_horizontal"), primitive("line_vertical"))
        library = primitive_library() + (
            LibraryItem("H_cross", execute(helper_program), helper_program),
        )
        targets = [
            execute(helper_program),
            execute(call("invert", helper_program)),
            execute(call("subtract", primitive("square"), primitive("line_horizontal"))),
        ]
        index = build_frontier_index(library, SolveConfig(max_program_size=2, max_solutions=1))

        for target in targets:
            indexed = index.score(target)
            solved = solve_task(
                target,
                library=library,
                config=SolveConfig(max_program_size=2, max_solutions=1),
            )
            self.assertEqual(indexed.solved, solved.solved)
            self.assertEqual(
                indexed.candidates_tried_at_first_solution,
                solved.candidates_tried_at_first_solution,
            )

    def test_random_selection_is_deterministic_per_draw_seed(self):
        candidates = [candidate(primitive(name)) for name in ("blank", "square", "triangle")]

        first = select_random_k(candidates, 2, "draw")
        second = select_random_k(candidates, 2, "draw")
        other = select_random_k(candidates, 2, "other")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_all_selectors_reject_negative_k(self):
        candidates = [candidate(primitive("square"))]
        selectors = {
            "random": lambda: select_random_k(candidates, -1, "draw"),
            "most_frequent": lambda: select_most_frequent_k(candidates, -1),
            "compression": lambda: select_compression_k_with_cost(candidates, (), -1),
            "utility": lambda: select_utility_k(candidates, (), -1, workers=1),
            "oracle": lambda: greedy_by_solved_count_with_cost(
                candidates, (), -1, workers=1
            ),
        }

        for name, select in selectors.items():
            with self.subTest(selector=name), self.assertRaisesRegex(
                ValueError, "K cannot be negative"
            ):
                select()

    def test_all_selectors_reject_insufficient_candidates_before_scoring(self):
        candidates = [candidate(primitive("square"))]
        selectors = {
            "random": lambda: select_random_k(candidates, 2, "draw"),
            "most_frequent": lambda: select_most_frequent_k(candidates, 2),
            "compression": lambda: select_compression_k_with_cost(candidates, (), 2),
            "utility": lambda: select_utility_k(candidates, (), 2, workers=1),
            "oracle": lambda: greedy_by_solved_count_with_cost(
                candidates, (), 2, workers=1
            ),
        }

        with patch("experiment.selection._score_trial") as score_trial:
            for name, select in selectors.items():
                with self.subTest(selector=name), self.assertRaisesRegex(
                    ValueError, "requested K=2, available=1"
                ):
                    select()

        score_trial.assert_not_called()

    def test_all_selectors_return_exact_k_for_valid_inputs(self):
        candidates = [candidate(primitive(name)) for name in ("blank", "square", "triangle")]
        summary = {
            "solved_count": 0,
            "mean_search_cost": 1,
            "frontier_candidates_tried_total": 1,
        }

        with patch("experiment.selection._solve_summary_for_candidates", return_value=summary):
            selected = (
                select_random_k(candidates, 2, "draw"),
                select_most_frequent_k(candidates, 2),
                select_compression_k_with_cost(candidates, (), 2).candidates,
                select_utility_k(candidates, (), 2, workers=1),
                greedy_by_solved_count_with_cost(candidates, (), 2, workers=1).candidates,
            )

        self.assertTrue(all(len(items) == 2 for items in selected))

    def test_all_selectors_allow_zero_k(self):
        selected = (
            select_random_k((), 0, "draw"),
            select_most_frequent_k((), 0),
            select_compression_k_with_cost((), (), 0).candidates,
            select_utility_k((), (), 0, workers=1),
            greedy_by_solved_count_with_cost((), (), 0, workers=1).candidates,
        )

        self.assertTrue(all(items == () for items in selected))

    def test_most_frequent_sorts_by_support_then_program(self):
        low = candidate(primitive("square"), ("start_001",))
        high_b = candidate(primitive("triangle"), ("start_001", "start_002"))
        high_a = candidate(primitive("line_horizontal"), ("start_001", "start_003"))

        selected = select_most_frequent_k([low, high_b, high_a], 2)

        self.assertEqual([item.program_string for item in selected], sorted([
            high_a.program_string,
            high_b.program_string,
        ]))

    def test_compression_score_rewards_matching_subtree_output(self):
        solution = call("invert", call("add", primitive("line_horizontal"), primitive("line_vertical")))
        matching = candidate(call("add", primitive("line_horizontal"), primitive("line_vertical")))
        non_matching = candidate(call("subtract", primitive("square"), primitive("line_horizontal")))

        self.assertGreater(
            compression_score(matching, [solution]),
            compression_score(non_matching, [solution]),
        )

    def test_compression_helper_cost_zero_and_op_cost_one(self):
        solution = call("invert", primitive("square"))
        matching = candidate(solution)

        self.assertEqual(compression_score(matching, [solution]), 1)

    def test_compression_reports_segmentation_work_outside_candidate_program_cost(self):
        solution = call("invert", primitive("square"))
        result = select_compression_k_with_cost([candidate(solution)], [solution], 1)

        self.assertEqual(result.cost["selection_cost_candidate_programs_tried"], 0)
        self.assertEqual(result.cost["input_solution_search_candidate_programs_tried"], 0)
        self.assertEqual(result.cost["trial_libraries_evaluated"], 1)
        self.assertEqual(result.cost["solution_segmentations_evaluated"], 1)

    def test_compression_trace_reports_cumulative_cost_and_round_direction(self):
        solution = call("invert", primitive("square"))
        matching = candidate(solution)
        irrelevant = candidate(primitive("triangle"))

        result = select_compression_k_with_cost(
            [irrelevant, matching], [solution], 2, trace=True
        )

        self.assertEqual(result.prefix_costs[-1], result.cost)
        self.assertEqual(
            [row["direction"] for row in result.round_diagnostics],
            ["positive", "zero"],
        )
        self.assertEqual(
            [row["selected_program"] for row in result.round_diagnostics],
            [matching.program_string, irrelevant.program_string],
        )
        self.assertEqual(
            [row["trial_libraries_evaluated"] for row in result.prefix_costs],
            [2, 3],
        )

    def test_utility_trace_uses_cost_first_lexicographic_direction(self):
        first = candidate(primitive("square"))
        second = candidate(primitive("triangle"))

        def fake_summary(selected, targets, config):
            names = tuple(item.program_string for item in selected)
            summaries = {
                (): (10, 1),
                (first.program_string,): (5, 1),
                (second.program_string,): (20, 10),
                (first.program_string, second.program_string): (6, 2),
            }
            cost, solved = summaries[names]
            return {
                "solved_count": solved,
                "mean_search_cost": cost,
                "frontier_candidates_tried_total": cost,
            }

        with patch(
            "experiment.selection._solve_summary_for_candidates",
            side_effect=fake_summary,
        ):
            result = greedy_by_frontier_score_with_cost(
                [second, first], (), 2, workers=1, trace=True
            )

        self.assertEqual(result.prefix_costs[-1], result.cost)
        self.assertEqual(
            [row["direction"] for row in result.round_diagnostics],
            ["positive", "negative"],
        )
        self.assertEqual(
            result.round_diagnostics[1]["marginal_objective_change"],
            {"mean_search_cost_reduction": -1, "solved_count_change": 1},
        )

    def test_oracle_trace_uses_solved_count_first_and_reports_ties(self):
        first = candidate(primitive("square"))
        second = candidate(primitive("triangle"))

        def fake_summary(selected, targets, config):
            names = tuple(item.program_string for item in selected)
            summaries = {
                (): (10, 1),
                (first.program_string,): (20, 2),
                (second.program_string,): (20, 2),
                tuple(sorted((first.program_string, second.program_string))): (25, 2),
            }
            lookup = names if names in summaries else tuple(sorted(names))
            cost, solved = summaries[lookup]
            return {
                "solved_count": solved,
                "mean_search_cost": cost,
                "frontier_candidates_tried_total": cost,
            }

        with patch(
            "experiment.selection._solve_summary_for_candidates",
            side_effect=fake_summary,
        ):
            result = greedy_by_solved_count_with_cost(
                [second, first], (), 2, workers=1, trace=True
            )

        self.assertEqual(result.round_diagnostics[0]["best_tie_count"], 2)
        self.assertEqual(
            [row["direction"] for row in result.round_diagnostics],
            ["positive", "negative"],
        )

    def test_parallel_and_serial_greedy_paths_match(self):
        candidates = [
            candidate(primitive("square")),
            candidate(primitive("triangle")),
        ]
        config = SolveConfig(node_budget=10, max_program_size=1, max_solutions=1)

        serial = greedy_by_frontier_score_with_cost(
            candidates, (), 2, config=config, workers=1, trace=True
        )
        parallel = greedy_by_frontier_score_with_cost(
            candidates, (), 2, config=config, workers=2, trace=True
        )

        self.assertEqual(parallel, serial)

    def test_utility_uses_cost_first_greedy_scoring(self):
        target = execute(call("add", primitive("line_horizontal"), primitive("line_vertical")))
        good = candidate(call("add", primitive("line_horizontal"), primitive("line_vertical")))
        bad = candidate(call("subtract", primitive("square"), primitive("line_horizontal")))

        self.assertEqual(
            select_utility_k([bad, good], [target], 1, workers=1),
            greedy_by_frontier_score([bad, good], [target], 1, workers=1),
        )

    def test_utility_prefers_lower_search_cost_over_more_solved_targets(self):
        low_cost = FrontierCandidate(
            program=primitive("square"),
            program_string="low_cost",
            output=execute(primitive("square")),
            op_count=1,
            support_task_ids=("start_000",),
            first_hit_cost=1,
        )
        high_solved = FrontierCandidate(
            program=primitive("triangle"),
            program_string="high_solved",
            output=execute(primitive("triangle")),
            op_count=1,
            support_task_ids=("start_001",),
            first_hit_cost=1,
        )

        def fake_summary(selected, targets, config):
            candidate = selected[-1]
            if candidate.program_string == "low_cost":
                return {
                    "solved_count": 0,
                    "mean_search_cost": 5,
                    "frontier_candidates_tried_total": 5,
                }
            return {
                "solved_count": 10,
                "mean_search_cost": 50,
                "frontier_candidates_tried_total": 50,
            }

        with patch("experiment.selection._solve_summary_for_candidates", side_effect=fake_summary):
            selected = greedy_by_frontier_score_with_cost(
                [high_solved, low_cost],
                [execute(primitive("square"))],
                1,
                workers=1,
            ).candidates

        self.assertEqual(selected, (low_cost,))

    def test_solved_count_oracle_prefers_more_solved_over_lower_cost(self):
        low_cost = FrontierCandidate(
            program=primitive("square"),
            program_string="low_cost",
            output=execute(primitive("square")),
            op_count=1,
            support_task_ids=("start_000",),
            first_hit_cost=1,
        )
        high_solved = FrontierCandidate(
            program=primitive("triangle"),
            program_string="high_solved",
            output=execute(primitive("triangle")),
            op_count=1,
            support_task_ids=("start_001",),
            first_hit_cost=1,
        )

        def fake_summary(selected, targets, config):
            candidate = selected[-1]
            if candidate.program_string == "low_cost":
                return {
                    "solved_count": 0,
                    "mean_search_cost": 5,
                    "frontier_candidates_tried_total": 5,
                }
            return {
                "solved_count": 10,
                "mean_search_cost": 50,
                "frontier_candidates_tried_total": 50,
            }

        with patch("experiment.selection._solve_summary_for_candidates", side_effect=fake_summary):
            selected = greedy_by_solved_count_with_cost(
                [low_cost, high_solved],
                [execute(primitive("square"))],
                1,
                workers=1,
            ).candidates

        self.assertEqual(selected, (high_solved,))

    def test_extraction_rejects_non_starter_tasks(self):
        task = Task(
            id="val_000",
            split="val",
            target=execute(primitive("square")),
            hidden_program=primitive("square"),
            motif_ids=(),
            combine_ops=(),
            glue_ops=(),
        )

        with self.assertRaises(ValueError):
            frontier_promotion_menu([task])

    def test_fresh_look_guard_rejects_prior_gate_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "experiment/data/selection/prior.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(
                json.dumps(
                    {
                        "gate_name": "selector_relevance_gate",
                        "gate_seeds": [6477],
                    }
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                with self.assertRaises(RuntimeError):
                    _fresh_look_guard([6477], artifact)
            finally:
                os.chdir(cwd)

    def test_command_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "selection.json"
            payload = {
                "gate_name": "selector_relevance_gate",
                "overall_pass": False,
                "selector_ready": False,
            }
            with patch(
                "experiment.commands.run_selection._fresh_look_guard"
            ), patch(
                "experiment.commands.run_selection.run_selector_gate",
                return_value=payload,
            ):
                selection_main(["--output", str(output), "--seeds", "6477"])

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
