import unittest

from experiment.commands.solve_problem_sets import _smoke_ok
from experiment.dsl import call, execute, primitive, program_to_string
from experiment.generator import DEFAULT_CONFIG_PATH, load_config, make_world
from experiment.solver import (
    LibraryItem,
    SolveConfig,
    build_frontier_index,
    primitive_library,
    solve_task,
    solve_tasks,
)


def tiny_library():
    return (
        LibraryItem("line_horizontal", execute(primitive("line_horizontal"))),
        LibraryItem("line_vertical", execute(primitive("line_vertical"))),
    )


class SolverTests(unittest.TestCase):
    def test_solves_primitive_target(self):
        target = execute(primitive("line_horizontal"))

        result = solve_task(target, config=SolveConfig(max_solutions=1))

        self.assertTrue(result.solved)
        self.assertEqual(program_to_string(result.solutions[0]), "line_horizontal")
        self.assertEqual(result.candidates_tried_at_first_solution, 2)

    def test_solves_simple_composed_target(self):
        target = execute(call("add", primitive("line_horizontal"), primitive("line_vertical")))

        result = solve_task(
            target,
            library=tiny_library(),
            config=SolveConfig(max_program_size=1, max_solutions=1),
        )

        self.assertTrue(result.solved)
        self.assertEqual(
            program_to_string(result.solutions[0]),
            "add(line_horizontal,line_vertical)",
        )

    def test_helper_solves_as_one_step_leaf(self):
        helper_program = call("add", primitive("line_horizontal"), primitive("line_vertical"))
        target = execute(helper_program)
        library = primitive_library() + (
            LibraryItem("H_cross", target, program=helper_program),
        )

        result = solve_task(target, library=library, config=SolveConfig(max_solutions=1))

        self.assertTrue(result.solved)
        self.assertEqual(program_to_string(result.solutions[0]), "H_cross")
        self.assertEqual(result.candidates_tried_at_first_solution, 7)

    def test_node_budget_is_respected(self):
        target = execute(primitive("line_vertical"))

        result = solve_task(
            target,
            library=(LibraryItem("line_horizontal", execute(primitive("line_horizontal"))),),
            config=SolveConfig(node_budget=1, max_program_size=1),
        )

        self.assertFalse(result.solved)
        self.assertTrue(result.hit_budget)
        self.assertEqual(result.candidates_tried_total, 1)

    def test_first_solution_cost_and_total_cost_can_differ(self):
        target = execute(primitive("blank"))

        result = solve_task(
            target,
            library=(LibraryItem("blank", target),),
            config=SolveConfig(max_program_size=1, max_solutions=2),
        )

        self.assertTrue(result.solved)
        self.assertEqual(len(result.solutions), 2)
        self.assertEqual(result.candidates_tried_at_first_solution, 1)
        self.assertGreater(result.candidates_tried_total, 1)

    def test_duplicate_outputs_are_counted_then_pruned(self):
        target = execute(call("add", primitive("line_horizontal"), primitive("line_vertical")))

        result = solve_task(
            target,
            library=tiny_library(),
            config=SolveConfig(max_program_size=1, max_solutions=1),
        )

        self.assertGreater(result.candidates_tried_total, result.unique_outputs)

    def test_frontier_records_attempts_and_first_hit_size(self):
        index = build_frontier_index(
            tiny_library(),
            SolveConfig(node_budget=50, max_program_size=1, max_solutions=1),
        )
        target = execute(
            call("add", primitive("line_horizontal"), primitive("line_vertical"))
        )

        self.assertEqual(len(index.candidates_tried_by_size), 2)
        self.assertEqual(sum(index.candidates_tried_by_size), index.candidates_tried_total)
        self.assertEqual(index.candidates_tried_by_size[0], 2)
        self.assertGreater(index.candidates_tried_total, index.unique_outputs)
        self.assertEqual(index.entries[target].abstract_search_size, 1)

    def test_enumeration_order_is_pinned(self):
        target = execute(call("add", primitive("line_horizontal"), primitive("line_vertical")))

        result = solve_task(
            target,
            library=tiny_library(),
            config=SolveConfig(max_program_size=1, max_solutions=1),
        )

        self.assertEqual(result.candidates_tried_at_first_solution, 12)

    def test_repeated_runs_are_identical(self):
        target = execute(call("add", primitive("line_horizontal"), primitive("line_vertical")))

        first = solve_task(target, library=tiny_library(), config=SolveConfig(max_program_size=2))
        second = solve_task(target, library=tiny_library(), config=SolveConfig(max_program_size=2))

        self.assertEqual(
            [program_to_string(program) for program in first.solutions],
            [program_to_string(program) for program in second.solutions],
        )
        self.assertEqual(first.candidates_tried_total, second.candidates_tried_total)

    def test_solve_tasks_returns_one_result_per_target(self):
        targets = [execute(primitive("line_horizontal")), execute(primitive("line_vertical"))]

        results = solve_tasks(targets, config=SolveConfig(max_solutions=1))

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.solved for result in results))

    def test_generated_tasks_can_be_searched(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        world = make_world(config, config.world_seeds[0], config.conditions[0])

        results = solve_tasks(
            [task.target for task in world.tasks_start[:2]],
            config=SolveConfig(node_budget=50, max_program_size=1, max_solutions=1),
        )

        self.assertEqual(len(results), 2)

    def test_smoke_gate_requires_rare_motif_coverage(self):
        self.assertFalse(_smoke_ok(0.70, 0))
        self.assertTrue(_smoke_ok(0.70, 1))


if __name__ == "__main__":
    unittest.main()
