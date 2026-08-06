import copy
import unittest

from experiment import capacity_curve as capacity


def _summary(task_count, solved=0, cost=100.0):
    frontier_cost = 30_000
    first_cost = cost if solved else None
    return {
        "solved_count": solved,
        "failure_count": task_count - solved,
        "task_count": task_count,
        "solve_rate": solved / task_count,
        "mean_search_cost": (
            solved * cost + (task_count - solved) * frontier_cost
        )
        / task_count,
        "mean_first_solution_cost": first_cost,
        "frontier_candidates_tried_total": frontier_cost,
        "hit_budget": True,
        "unique_outputs": 100,
    }


def _cost(name, k):
    if k == 0 or name == capacity.RANDOM_ARM:
        return capacity.zero_cost()
    trials = k * capacity.K_MAX - k * (k - 1) // 2
    if name in capacity.COMPRESSION_ARMS:
        acquisition = (
            capacity.VALIDATION_TASK_COUNT
            if name == capacity.COMPRESSION_VALIDATION_ARM
            else 0
        )
        return {
            "selection_cost_candidate_programs_tried": acquisition,
            "input_solution_search_candidate_programs_tried": acquisition,
            "trial_libraries_evaluated": trials,
            "segmentation_evaluations": trials,
            "solution_segmentations_evaluated": trials,
            "frontier_candidates_tried_total": 0,
        }
    frontier_work = sum(
        (capacity.K_MAX - round_number + 1) * (6 + round_number)
        for round_number in range(1, k + 1)
    )
    return {
        "selection_cost_candidate_programs_tried": frontier_work,
        "input_solution_search_candidate_programs_tried": 0,
        "trial_libraries_evaluated": trials,
        "segmentation_evaluations": 0,
        "solution_segmentations_evaluated": 0,
        "frontier_candidates_tried_total": frontier_work,
    }


def _prefixes(validation, test, name):
    rows = []
    for k in range(capacity.K_MAX + 1):
        validation_summary = copy.deepcopy(validation)
        test_summary = copy.deepcopy(test)
        if name == capacity.UTILITY_ARM:
            validation_summary = _summary(
                capacity.VALIDATION_TASK_COUNT, solved=k, cost=100 - k
            )
        elif name == capacity.TEST_PEEK_ARM:
            test_summary = _summary(capacity.TEST_TASK_COUNT, solved=k, cost=100 - k)
        rows.append(
            {
                "k": k,
                "validation_summary": validation_summary,
                "test_summary": test_summary,
                "selection_cost": _cost(name, k),
            }
        )
    return rows


def _diagnostics(programs, objective, prefixes):
    rows = []
    for index, program in enumerate(programs, start=1):
        if objective == "compression":
            before = 21 - index
            after = before - 1
            marginal = 1
        else:
            summary_name = (
                "validation_summary"
                if objective == capacity.UTILITY_ARM
                else "test_summary"
            )
            before_summary = prefixes[index - 1][summary_name]
            after_summary = prefixes[index][summary_name]
            before = {
                "mean_search_cost": before_summary["mean_search_cost"],
                "solved_count": before_summary["solved_count"],
            }
            after = {
                "mean_search_cost": after_summary["mean_search_cost"],
                "solved_count": after_summary["solved_count"],
            }
            marginal = {
                "mean_search_cost_reduction": (
                    before["mean_search_cost"] - after["mean_search_cost"]
                ),
                "solved_count_change": after["solved_count"] - before["solved_count"],
            }
        rows.append(
            {
                "round": index,
                "selected_program": program,
                "objective_before": before,
                "objective_after": after,
                "marginal_objective_change": marginal,
                "direction": "positive",
                "best_tie_count": 1,
            }
        )
    return rows


def synthetic_cell(seed=6541, condition="reversed_a0", formal=True):
    programs = [f"program_{index:02d}" for index in range(capacity.K_MAX)]
    validation = _summary(capacity.VALIDATION_TASK_COUNT)
    test = _summary(capacity.TEST_TASK_COUNT)
    arms = {}
    for name in capacity.GREEDY_ARMS:
        objective = "compression" if name in capacity.COMPRESSION_ARMS else name
        prefixes = _prefixes(validation, test, name)
        arms[name] = {
            "selected_programs": list(programs),
            "prefixes": prefixes,
            "round_diagnostics": _diagnostics(programs, objective, prefixes),
        }
    arms[capacity.RANDOM_ARM] = {
        "draws": [
            {
                "draw": draw,
                "selected_programs": list(programs),
                "prefixes": _prefixes(validation, test, capacity.RANDOM_ARM),
            }
            for draw in range(capacity.RANDOM_DRAWS)
        ]
    }
    return {
        "experiment_name": (
            capacity.EXPERIMENT_NAME if formal else capacity.SMOKE_EXPERIMENT_NAME
        ),
        "seed": seed,
        "world_seed": seed,
        "condition": condition,
        "formal_seed": formal,
        "motif_count": capacity.MOTIF_COUNT,
        "starter_task_count": 100,
        "k_max": capacity.K_MAX,
        "random_draws": capacity.RANDOM_DRAWS,
        "validation_task_count": capacity.VALIDATION_TASK_COUNT,
        "test_task_count": capacity.TEST_TASK_COUNT,
        "world_metadata": {
            "condition": {
                "name": condition,
                "alt_kind": "reversed" if condition.startswith("reversed") else "permuted",
                "alpha_val": {"reversed_a0": 0.0, "reversed_a05": 0.5, "reversed_a1": 1.0,
                    "permuted_a0": 0.0, "permuted_a05": 0.5, "permuted_a1": 1.0}[condition],
                "alpha_test": {"reversed_a0": 0.0, "reversed_a05": 0.5, "reversed_a1": 1.0,
                    "permuted_a0": 0.0, "permuted_a05": 0.5, "permuted_a1": 1.0}[condition],
            },
            "realized_rho": {
                "realized_start_test": 0.5,
                "realized_start_val": 0.5,
                "realized_val_test": 0.5,
            },
            "density_summary": {},
            "expected_motif_length": {},
        },
        "shared_hashes": {
            "hidden_motifs": "0" * 64,
            "p_start": "1" * 64,
            "starter_tasks": "2" * 64,
            "candidate_menu": capacity.canonical_hash(programs),
        },
        "menu": {
            "menu_size": len(programs),
            "raw_candidate_count": len(programs),
            "op_count_distribution": {"1": len(programs)},
            "support_distribution": {"2": len(programs)},
            "frontier_unique_outputs": 100,
            "frontier_candidates_tried_total": 30_000,
            "frontier_hit_budget": True,
            "cap": 50,
        },
        "candidate_menu_programs": list(programs),
        "input_solution_diagnostics": {
            "all_100_starter": {
                "candidate_programs_tried_total": 100,
                "canonical_solution_count": 1,
                "solution_program_count_before_canonicalization": 1,
                "solved_task_count": 1,
            },
            "validation_assisted": {
                "candidate_programs_tried_total": capacity.VALIDATION_TASK_COUNT,
                "canonical_solution_count": 1,
                "solution_program_count_before_canonicalization": 1,
                "solved_task_count": 1,
                "solve_config": {
                    "node_budget": 90_000,
                    "max_program_size": 7,
                    "max_solutions": 1,
                },
            },
        },
        "primitive": {
            "validation_summary": validation,
            "test_summary": test,
        },
        "arms": arms,
        "timings": {
            "selection_seconds": 1.0,
            "real_prefix_seconds": 1.0,
            "random_prefix_seconds": 1.0,
        },
        "wall_clock_seconds": 1.0,
    }


class CapacityCurveValidationTests(unittest.TestCase):
    def test_validates_complete_cell(self):
        capacity.validate_cell(synthetic_cell(), formal=True)

    def test_rejects_bad_menu_prefix_and_random_metadata(self):
        cases = []
        short_menu = synthetic_cell()
        short_menu["candidate_menu_programs"].pop()
        short_menu["menu"]["menu_size"] -= 1
        cases.append((short_menu, "menu size"))

        bad_prefix = synthetic_cell()
        bad_prefix["arms"][capacity.UTILITY_ARM]["prefixes"][2]["k"] = 9
        cases.append((bad_prefix, "prefix"))

        bad_draws = synthetic_cell()
        bad_draws["random_draws"] = 1
        cases.append((bad_draws, "random_draws"))

        for cell, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                capacity.validate_cell(cell, formal=True)

    def test_rejects_trace_misalignment_and_invalid_direction(self):
        cell = synthetic_cell()
        diagnostics = cell["arms"][capacity.UTILITY_ARM]["round_diagnostics"]
        diagnostics[0]["selected_program"] = "wrong"
        with self.assertRaisesRegex(ValueError, "diagnostic selection"):
            capacity.validate_cell(cell, formal=True)

    def test_rejects_inconsistent_failure_count(self):
        cell = synthetic_cell()
        cell["arms"][capacity.UTILITY_ARM]["prefixes"][1]["test_summary"][
            "failure_count"
        ] = 99

        with self.assertRaisesRegex(ValueError, "failure_count"):
            capacity.validate_cell(cell, formal=True)

    def test_rejects_candidate_menu_hash_that_does_not_match_programs(self):
        cell = synthetic_cell()
        cell["shared_hashes"]["candidate_menu"] = "f" * 64

        with self.assertRaisesRegex(ValueError, "candidate-menu hash"):
            capacity.validate_cell(cell, formal=True)

    def test_rejects_inconsistent_world_and_input_metadata(self):
        cell = synthetic_cell()
        cell["world_metadata"]["condition"]["name"] = "permuted_a1"
        with self.assertRaisesRegex(ValueError, "world condition"):
            capacity.validate_cell(cell, formal=True)

        cell = synthetic_cell()
        cell["input_solution_diagnostics"]["validation_assisted"]["solve_config"][
            "node_budget"
        ] = 1
        with self.assertRaisesRegex(ValueError, "assisted solve config"):
            capacity.validate_cell(cell, formal=True)

        cell = synthetic_cell()
        cell["arms"][capacity.UTILITY_ARM]["round_diagnostics"][0][
            "direction"
        ] = "negative"
        with self.assertRaisesRegex(ValueError, "diagnostic direction"):
            capacity.validate_cell(cell, formal=True)

    def test_rejects_contradictory_menu_and_input_counts(self):
        cases = []

        raw_too_small = synthetic_cell()
        raw_too_small["menu"]["raw_candidate_count"] = 19
        cases.append((raw_too_small, "raw candidate"))

        bad_op_distribution = synthetic_cell()
        bad_op_distribution["menu"]["op_count_distribution"] = {"1": 19}
        cases.append((bad_op_distribution, "op-count distribution"))

        bad_support_distribution = synthetic_cell()
        bad_support_distribution["menu"]["support_distribution"] = {"2": 21}
        cases.append((bad_support_distribution, "support distribution"))

        too_many_unique_outputs = synthetic_cell()
        too_many_unique_outputs["menu"]["frontier_unique_outputs"] = 30_001
        cases.append((too_many_unique_outputs, "unique outputs"))

        solved_without_canonical_solution = synthetic_cell()
        solved_without_canonical_solution["input_solution_diagnostics"][
            "all_100_starter"
        ]["solved_task_count"] = 2
        cases.append((solved_without_canonical_solution, "canonical solutions"))

        impossible_search_cost = synthetic_cell()
        impossible_search_cost["input_solution_diagnostics"]["all_100_starter"][
            "candidate_programs_tried_total"
        ] = 3_000_001
        cases.append((impossible_search_cost, "search budget"))

        for cell, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                capacity.validate_cell(cell, formal=True)

    def test_rejects_inconsistent_summaries(self):
        cases = []

        wrong_rate = synthetic_cell()
        wrong_rate["primitive"]["test_summary"]["solve_rate"] = 0.5
        for name, arm in wrong_rate["arms"].items():
            paths = (
                [draw["prefixes"] for draw in arm["draws"]]
                if name == capacity.RANDOM_ARM
                else [arm["prefixes"]]
            )
            for rows in paths:
                rows[0]["test_summary"]["solve_rate"] = 0.5
        cases.append((wrong_rate, "solve_rate"))

        nan_cost = synthetic_cell()
        nan_cost["arms"][capacity.UTILITY_ARM]["prefixes"][1][
            "validation_summary"
        ]["mean_search_cost"] = float("nan")
        cases.append((nan_cost, "finite"))

        bad_budget_flag = synthetic_cell()
        bad_budget_flag["arms"][capacity.COMPRESSION_ALL_ARM]["prefixes"][1][
            "test_summary"
        ]["hit_budget"] = "yes"
        cases.append((bad_budget_flag, "hit_budget"))

        too_many_outputs = synthetic_cell()
        too_many_outputs["arms"][capacity.COMPRESSION_ALL_ARM]["prefixes"][1][
            "test_summary"
        ]["unique_outputs"] = 30_001
        cases.append((too_many_outputs, "unique_outputs"))

        mean_cost_beyond_frontier = synthetic_cell()
        mean_cost_beyond_frontier["arms"][capacity.COMPRESSION_ALL_ARM][
            "prefixes"
        ][1]["test_summary"][
            "mean_search_cost"
        ] = 30_001
        cases.append((mean_cost_beyond_frontier, "mean_search_cost"))

        first_cost_beyond_frontier = synthetic_cell()
        summary = first_cost_beyond_frontier["arms"][capacity.UTILITY_ARM][
            "prefixes"
        ][1]["validation_summary"]
        summary["mean_first_solution_cost"] = 30_001
        cases.append((first_cost_beyond_frontier, "mean_first_solution_cost"))

        impossible_weighted_cost = synthetic_cell()
        summary = impossible_weighted_cost["arms"][capacity.COMPRESSION_ALL_ARM][
            "prefixes"
        ][1]["test_summary"]
        _set_solved(summary, 1)
        summary["mean_search_cost"] += 1
        cases.append((impossible_weighted_cost, "weighted cost identity"))

        zero_first_hit_cost = synthetic_cell()
        zero_first_hit_cost["arms"][capacity.UTILITY_ARM]["prefixes"][1][
            "validation_summary"
        ]["mean_first_solution_cost"] = 0
        cases.append((zero_first_hit_cost, "positive"))

        unsolved_cost_below_frontier = synthetic_cell()
        unsolved_cost_below_frontier["arms"][capacity.COMPRESSION_ALL_ARM][
            "prefixes"
        ][1]["test_summary"]["mean_search_cost"] -= 1
        cases.append((unsolved_cost_below_frontier, "weighted cost identity"))

        for cell, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                capacity.validate_cell(cell, formal=True)

    def test_rejects_too_few_frontier_outputs_for_library_or_solved_targets(self):
        too_few_for_leaves = synthetic_cell()
        prefix = too_few_for_leaves["arms"][capacity.COMPRESSION_ALL_ARM][
            "prefixes"
        ][1]
        for summary_name in ("validation_summary", "test_summary"):
            prefix[summary_name]["unique_outputs"] = 6

        solves_more_than_outputs = synthetic_cell()
        prefix = solves_more_than_outputs["arms"][capacity.COMPRESSION_ALL_ARM][
            "prefixes"
        ][20]
        _set_solved(prefix["test_summary"], 30)
        for summary_name in ("validation_summary", "test_summary"):
            prefix[summary_name]["unique_outputs"] = 29

        for cell in (too_few_for_leaves, solves_more_than_outputs):
            with self.subTest(), self.assertRaisesRegex(ValueError, "unique_outputs"):
                capacity.validate_cell(cell, formal=True)

    def test_rejects_validation_test_frontier_mismatch_for_one_library(self):
        cell = synthetic_cell()
        cell["arms"][capacity.COMPRESSION_ALL_ARM]["prefixes"][1][
            "test_summary"
        ]["unique_outputs"] = 99

        with self.assertRaisesRegex(ValueError, "frontier diagnostics"):
            capacity.validate_cell(cell, formal=True)

    def test_rejects_wrong_arm_specific_costs(self):
        cases = []

        wrong_trials = synthetic_cell()
        wrong_trials["arms"][capacity.COMPRESSION_ALL_ARM]["prefixes"][1][
            "selection_cost"
        ]["trial_libraries_evaluated"] = 1
        cases.append((wrong_trials, "trial-library count"))

        compression_frontier_cost = synthetic_cell()
        for row in compression_frontier_cost["arms"][
            capacity.COMPRESSION_ALL_ARM
        ]["prefixes"][1:]:
            row["selection_cost"]["frontier_candidates_tried_total"] = 1
        cases.append((compression_frontier_cost, "compression cost units"))

        utility_segmentation_cost = synthetic_cell()
        for row in utility_segmentation_cost["arms"][capacity.UTILITY_ARM][
            "prefixes"
        ][1:]:
            row["selection_cost"]["segmentation_evaluations"] = 1
        cases.append((utility_segmentation_cost, "trial-selector cost units"))

        disconnected_trial_cost = synthetic_cell()
        disconnected_trial_cost["arms"][capacity.UTILITY_ARM]["prefixes"][1][
            "selection_cost"
        ]["selection_cost_candidate_programs_tried"] = 2
        cases.append((disconnected_trial_cost, "frontier cost"))

        impossible_zero_frontier_work = synthetic_cell()
        for row in impossible_zero_frontier_work["arms"][capacity.UTILITY_ARM][
            "prefixes"
        ][1:]:
            row["selection_cost"]["selection_cost_candidate_programs_tried"] = 0
            row["selection_cost"]["frontier_candidates_tried_total"] = 0
        cases.append((impossible_zero_frontier_work, "frontier work"))

        for cell, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                capacity.validate_cell(cell, formal=True)

    def test_rejects_input_search_work_below_source_task_count(self):
        cell = synthetic_cell()
        cell["input_solution_diagnostics"]["all_100_starter"][
            "candidate_programs_tried_total"
        ] = 0

        with self.assertRaisesRegex(ValueError, "source task count"):
            capacity.validate_cell(cell, formal=True)

    def test_rejects_disconnected_or_impossible_round_diagnostics(self):
        cases = []

        impossible_tie = synthetic_cell()
        impossible_tie["arms"][capacity.UTILITY_ARM]["round_diagnostics"][0][
            "best_tie_count"
        ] = 21
        cases.append((impossible_tie, "tie count"))

        disconnected = synthetic_cell()
        diagnostic = disconnected["arms"][capacity.COMPRESSION_ALL_ARM][
            "round_diagnostics"
        ][1]
        diagnostic["objective_before"] = 20
        diagnostic["marginal_objective_change"] = 2
        cases.append((disconnected, "trace continuity"))

        objective_mismatch = synthetic_cell()
        diagnostic = objective_mismatch["arms"][capacity.UTILITY_ARM][
            "round_diagnostics"
        ][0]
        diagnostic["objective_after"]["mean_search_cost"] = 98
        diagnostic["marginal_objective_change"]["mean_search_cost_reduction"] = (
            diagnostic["objective_before"]["mean_search_cost"] - 98
        )
        cases.append((objective_mismatch, "prefix summary"))

        for cell, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                capacity.validate_cell(cell, formal=True)

    def test_rejects_assisted_cost_that_disagrees_with_input_diagnostic(self):
        cell = synthetic_cell()
        cell["input_solution_diagnostics"]["validation_assisted"][
            "candidate_programs_tried_total"
        ] = capacity.VALIDATION_TASK_COUNT + 1

        with self.assertRaisesRegex(ValueError, "assisted acquisition"):
            capacity.validate_cell(cell, formal=True)

    def test_rejects_unregistered_generator_counts_and_input_overflow(self):
        cases = []

        wrong_motifs = synthetic_cell()
        wrong_motifs["motif_count"] = 11
        cases.append((wrong_motifs, "motif count"))

        wrong_starter_count = synthetic_cell()
        wrong_starter_count["starter_task_count"] = 99
        cases.append((wrong_starter_count, "starter task count"))

        too_many_starter_solutions = synthetic_cell()
        too_many_starter_solutions["input_solution_diagnostics"]["all_100_starter"][
            "solved_task_count"
        ] = 101
        too_many_starter_solutions["input_solution_diagnostics"]["all_100_starter"][
            "canonical_solution_count"
        ] = 101
        too_many_starter_solutions["input_solution_diagnostics"]["all_100_starter"][
            "solution_program_count_before_canonicalization"
        ] = 101
        cases.append((too_many_starter_solutions, "source task count"))

        too_many_validation_solutions = synthetic_cell()
        too_many_validation_solutions["input_solution_diagnostics"][
            "validation_assisted"
        ]["solved_task_count"] = 26
        too_many_validation_solutions["input_solution_diagnostics"][
            "validation_assisted"
        ]["canonical_solution_count"] = 26
        too_many_validation_solutions["input_solution_diagnostics"][
            "validation_assisted"
        ]["solution_program_count_before_canonicalization"] = 26
        cases.append((too_many_validation_solutions, "source task count"))

        fractional_cost = synthetic_cell()
        fractional_cost["arms"][capacity.UTILITY_ARM]["prefixes"][1][
            "selection_cost"
        ]["frontier_candidates_tried_total"] = 1.5
        cases.append((fractional_cost, "integer counts"))

        for cell, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                capacity.validate_cell(cell, formal=True)

    def test_validates_complete_aggregate_and_shared_seed_invariants(self):
        cells = [
            synthetic_cell(seed, condition)
            for seed in capacity.FORMAL_SEEDS
            for condition in capacity.CONDITIONS
        ]
        payload = {
            "experiment_name": capacity.EXPERIMENT_NAME,
            "smoke": False,
            "registration": capacity.registration(smoke=False),
            "cells": cells,
        }

        capacity.validate_aggregate(payload, formal=True)

        changed = copy.deepcopy(payload)
        changed["cells"][1]["shared_hashes"]["starter_tasks"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "shared starter"):
            capacity.validate_aggregate(changed, formal=True)

        changed = copy.deepcopy(payload)
        changed["cells"][1]["arms"][capacity.RANDOM_ARM]["draws"][0][
            "selected_programs"
        ].reverse()
        with self.assertRaisesRegex(ValueError, "random paths"):
            capacity.validate_aggregate(changed, formal=True)

        changed = copy.deepcopy(payload)
        cell = changed["cells"][1]
        for summary_name in ("validation_summary", "test_summary"):
            cell["primitive"][summary_name]["unique_outputs"] = 99
        for name, arm in cell["arms"].items():
            paths = (
                [draw["prefixes"] for draw in arm["draws"]]
                if name == capacity.RANDOM_ARM
                else [arm["prefixes"]]
            )
            for prefixes in paths:
                for summary_name in ("validation_summary", "test_summary"):
                    prefixes[0][summary_name]["unique_outputs"] = 99
        with self.assertRaisesRegex(ValueError, "shared primitive"):
            capacity.validate_aggregate(changed, formal=True)

        changed = copy.deepcopy(payload)
        c100 = changed["cells"][1]["arms"][capacity.COMPRESSION_ALL_ARM]
        c100["selected_programs"][0], c100["selected_programs"][1] = (
            c100["selected_programs"][1],
            c100["selected_programs"][0],
        )
        c100["round_diagnostics"][0]["selected_program"], c100[
            "round_diagnostics"
        ][1]["selected_program"] = (
            c100["round_diagnostics"][1]["selected_program"],
            c100["round_diagnostics"][0]["selected_program"],
        )
        with self.assertRaisesRegex(ValueError, "shared C100"):
            capacity.validate_aggregate(changed, formal=True)

        changed = copy.deepcopy(payload)
        changed["cells"][1]["menu"]["raw_candidate_count"] = 21
        with self.assertRaisesRegex(ValueError, "shared menu diagnostics"):
            capacity.validate_aggregate(changed, formal=True)

        changed = copy.deepcopy(payload)
        changed["cells"][1]["input_solution_diagnostics"]["all_100_starter"][
            "candidate_programs_tried_total"
        ] = 101
        with self.assertRaisesRegex(ValueError, "shared starter diagnostics"):
            capacity.validate_aggregate(changed, formal=True)

        changed = copy.deepcopy(payload)
        c100_prefix = changed["cells"][1]["arms"][
            capacity.COMPRESSION_ALL_ARM
        ]["prefixes"][1]
        for summary_name in ("validation_summary", "test_summary"):
            c100_prefix[summary_name]["unique_outputs"] = 99
        with self.assertRaisesRegex(ValueError, "shared C100"):
            capacity.validate_aggregate(changed, formal=True)

        changed = copy.deepcopy(payload)
        random_prefix = changed["cells"][1]["arms"][capacity.RANDOM_ARM][
            "draws"
        ][0]["prefixes"][1]
        for summary_name in ("validation_summary", "test_summary"):
            random_prefix[summary_name]["unique_outputs"] = 99
        with self.assertRaisesRegex(ValueError, "random paths"):
            capacity.validate_aggregate(changed, formal=True)

    def test_smoke_shape_uses_spent_seed_and_full_workload(self):
        cell = synthetic_cell(seed=6511, formal=False)
        capacity.validate_cell(cell, formal=False)
        registration = capacity.registration(smoke=True)

        self.assertEqual(registration["seeds"], [6511])
        self.assertEqual(registration["conditions"], ["reversed_a0"])
        self.assertEqual(registration["random_draws"], 20)


class CapacityCurveAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cells = [
            synthetic_cell(seed, condition)
            for seed in capacity.FORMAL_SEEDS
            for condition in capacity.CONDITIONS
        ]
        for cell in cells:
            primitive = cell["primitive"]["test_summary"]
            _set_solved(primitive, 50)
            for name, slope in (
                (capacity.COMPRESSION_ALL_ARM, 1),
                (capacity.COMPRESSION_VALIDATION_ARM, 1),
                (capacity.UTILITY_ARM, 2),
                (capacity.TEST_PEEK_ARM, 2),
            ):
                for row in cell["arms"][name]["prefixes"]:
                    _set_solved(row["test_summary"], 50 + slope * row["k"])
            diagnostics = cell["arms"][capacity.TEST_PEEK_ARM][
                "round_diagnostics"
            ]
            prefixes = cell["arms"][capacity.TEST_PEEK_ARM]["prefixes"]
            for k, diagnostic in enumerate(diagnostics, start=1):
                before = prefixes[k - 1]["test_summary"]
                after = prefixes[k]["test_summary"]
                diagnostic["objective_before"] = {
                    "mean_search_cost": before["mean_search_cost"],
                    "solved_count": before["solved_count"],
                }
                diagnostic["objective_after"] = {
                    "mean_search_cost": after["mean_search_cost"],
                    "solved_count": after["solved_count"],
                }
                diagnostic["marginal_objective_change"] = {
                    "mean_search_cost_reduction": (
                        before["mean_search_cost"] - after["mean_search_cost"]
                    ),
                    "solved_count_change": (
                        after["solved_count"] - before["solved_count"]
                    ),
                }
                diagnostic["direction"] = "positive"
            for draw in cell["arms"][capacity.RANDOM_ARM]["draws"]:
                for row in draw["prefixes"]:
                    _set_solved(row["test_summary"], 50 + row["k"] // 2)
        cls.payload = {
            "experiment_name": capacity.EXPERIMENT_NAME,
            "smoke": False,
            "registration": capacity.registration(smoke=False),
            "cells": cells,
        }

    def test_primary_estimands_match_registered_definitions(self):
        analysis = capacity.analyze(self.payload)

        self.assertEqual(
            {name: row["estimate"] for name, row in analysis["primary"].items()},
            {
                "utility20_minus_assisted_compression20": 20,
                "scoring_effect20_minus_scoring_effect10": 10,
                "utility20_minus_past_compression20": 20,
                "utility20_minus_utility10": 20,
                "past_compression20_minus_past_compression10": 10,
            },
        )
        self.assertTrue(
            all(row["interval"] == [row["estimate"], row["estimate"]]
                for row in analysis["primary"].values())
        )

    def test_curves_include_assisted_compression_and_mean_random(self):
        analysis = capacity.analyze(self.payload)

        self.assertEqual(
            set(analysis["curves"]),
            {
                "past_compression_gain",
                "assisted_compression_gain",
                "utility_gain",
                "mean_random_gain",
                "test_peeking_gain",
                "utility_minus_assisted_compression",
                "utility_minus_past_compression",
            },
        )
        self.assertEqual(
            analysis["curves"]["mean_random_gain"][20]["estimate"], 10
        )

    def test_post_k10_changes_have_direct_simultaneous_bands(self):
        analysis = capacity.analyze(self.payload)

        self.assertEqual(
            set(analysis["post_k10_changes"]),
            {
                "past_compression",
                "assisted_compression",
                "utility",
                "mean_random",
                "test_peeking",
            },
        )
        self.assertEqual(
            set(analysis["post_k10_changes"]["utility"]), set(range(11, 21))
        )
        self.assertEqual(
            analysis["post_k10_changes"]["utility"][20]["estimate"], 20
        )
        self.assertEqual(
            analysis["post_k10_changes"]["mean_random"][20]["estimate"], 5
        )

    def test_bootstrap_samples_and_nearest_rank_are_frozen(self):
        first = capacity.bootstrap_samples()
        second = capacity.bootstrap_samples()

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10_000)
        self.assertEqual(len(first[0]), 30)
        self.assertEqual(capacity.nearest_rank_95(list(range(10_000))), 9499)


def _set_solved(summary, solved):
    summary["solved_count"] = solved
    summary["failure_count"] = summary["task_count"] - solved
    summary["solve_rate"] = solved / summary["task_count"]
    first_cost = 100.0 if solved else None
    summary["mean_first_solution_cost"] = first_cost
    summary["mean_search_cost"] = (
        solved * (first_cost or 0)
        + summary["failure_count"] * summary["frontier_candidates_tried_total"]
    ) / summary["task_count"]


if __name__ == "__main__":
    unittest.main()
