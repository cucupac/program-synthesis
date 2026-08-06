"""Paper figures for the formal selection experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import json
import math
from pathlib import Path
import random
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from experiment import budget_intervention as budget_intervention
from experiment import capacity_curve as capacity
from experiment.commands.run_budget_intervention import _source_anchor, _task_hash_rows
from experiment.commands.run_full_selection_experiment import (
    ASSISTED_VALIDATION_SOLVE_CONFIG,
    FORMAL_SEEDS,
    MATCHED_STARTER_COUNT,
    matched_starter_tasks,
)
from experiment.dsl import program_to_string
from experiment.generator import DEFAULT_CONFIG_PATH, load_config, make_world
from experiment.selection import _segmentation_cost
from experiment.solver import SolveConfig, build_frontier_index, primitive_library


BLUE = "#4477AA"
GRAY = "#333333"
LIGHT_GRAY = "#C9C9C9"
BOOTSTRAP_SEED = 20260710
K_SWEEP_NAME = "full_selection_experiment_k_sweep"
K_SWEEP_VALUES = (2, 5, 10)
K_SWEEP_SEEDS = tuple(range(6511, 6541))
K_SWEEP_CONDITIONS = (
    "reversed_a0",
    "reversed_a05",
    "reversed_a1",
    "permuted_a0",
    "permuted_a05",
    "permuted_a1",
)


def load_formal_results(path: Path) -> dict:
    results = json.loads(path.read_text(encoding="utf-8"))
    if results.get("experiment_name") != "full_selection_experiment_primary_k10":
        raise ValueError("expected the formal K=10 selection experiment")
    if results.get("smoke") is not False or len(results.get("cells", ())) != 210:
        raise ValueError("expected the complete 210-cell formal result")
    return results


def load_k_sweep_results(path: Path) -> dict:
    results = json.loads(path.read_text(encoding="utf-8"))
    registration = results.get("registration", {})
    cells = results.get("cells", ())
    expected = {
        (k, seed, condition)
        for k in K_SWEEP_VALUES
        for seed in K_SWEEP_SEEDS
        for condition in K_SWEEP_CONDITIONS
    }
    actual = {
        (cell.get("k"), cell.get("seed"), cell.get("condition")) for cell in cells
    }
    if (
        results.get("experiment_name") != K_SWEEP_NAME
        or results.get("smoke") is not False
        or registration.get("experiment_name") != K_SWEEP_NAME
        or registration.get("k_values") != list(K_SWEEP_VALUES)
        or registration.get("seeds") != list(K_SWEEP_SEEDS)
        or registration.get("conditions") != list(K_SWEEP_CONDITIONS)
        or registration.get("cell_count") != len(expected)
        or registration.get("random_draws") != 20
        or len(cells) != len(expected)
        or actual != expected
        or any(not _valid_k_sweep_cell(cell, registration) for cell in cells)
    ):
        raise ValueError("expected the complete registered 540-cell K sweep")
    return results


def load_capacity_curve_results(path: Path) -> dict:
    results = json.loads(path.read_text(encoding="utf-8"))
    capacity.validate_aggregate(results, formal=True)
    return results


def load_budget_intervention_results(path: Path) -> dict:
    results = json.loads(path.read_text(encoding="utf-8"))
    if results.get("experiment_name") != budget_intervention.EXPERIMENT_NAME:
        raise ValueError("expected the formal budget intervention")

    root = Path(__file__).resolve().parents[1]
    source_path = root / budget_intervention.FORMAL_SOURCE
    if budget_intervention.file_sha256(source_path) != budget_intervention.FORMAL_SOURCE_SHA256:
        raise ValueError("registered capacity source hash does not match")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    capacity.validate_aggregate(source, formal=True)

    config = load_config(str(root / DEFAULT_CONFIG_PATH))
    conditions = {condition.name: condition for condition in config.conditions}
    anchors = []
    for cell in source["cells"]:
        world = make_world(config, cell["seed"], conditions[cell["condition"]])
        anchors.append(
            {
                **_source_anchor(cell),
                "task_hashes": {
                    "validation": _task_hash_rows(world.tasks_val),
                    "test": _task_hash_rows(world.tasks_test),
                },
            }
        )
    budget_intervention.validate_aggregate(
        results,
        formal=True,
        source_anchors=anchors,
    )
    return results


def _valid_k_sweep_cell(cell: dict, registration: dict) -> bool:
    arms = cell.get("arms")
    k = cell.get("k")
    if (
        type(k) is not int
        or cell.get("random_draws") != 20
        or not isinstance(arms, dict)
        or set(arms) != set(registration.get("arms", {}))
        or "diagnostic_candidate_count" in arms.get("hidden_motif_oracle", {})
    ):
        return False
    for name, arm in arms.items():
        if name == "random_k":
            draws = arm.get("draws") if isinstance(arm, dict) else None
            if (
                not isinstance(draws, list)
                or len(draws) != 20
                or any(len(row.get("selected_programs", ())) != k for row in draws)
            ):
                return False
        else:
            expected = 0 if name == "primitives_only" else k
            if not isinstance(arm, dict) or len(arm.get("selected_programs", ())) != expected:
                return False
    return True


def _solved(cell: dict, arm: str) -> float:
    return float(cell["arms"][arm]["summary"]["solved_count"])


def _cluster_ci(
    cells: Sequence[dict],
    value: Callable[[dict], float],
    *,
    draws: int = 5_000,
) -> tuple[float, float, float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for cell in cells:
        by_seed[cell["seed"]].append(float(value(cell)))
    seeds = sorted(by_seed)
    point = mean(item for seed in seeds for item in by_seed[seed])
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(draws):
        sample = [rng.choice(seeds) for _ in seeds]
        estimates.append(mean(item for seed in sample for item in by_seed[seed]))
    estimates.sort()
    return point, estimates[int(0.025 * draws)], estimates[int(0.975 * draws) - 1]


def _comparison(cells: Sequence[dict], label: str) -> dict:
    primitive = _cluster_ci(cells, lambda cell: _solved(cell, "primitives_only"))
    past = _cluster_ci(cells, lambda cell: _solved(cell, "compression_on_all_100_starter"))
    utility = _cluster_ci(cells, lambda cell: _solved(cell, "utility_on_validation"))
    compression_gain = _cluster_ci(
        cells,
        lambda cell: _solved(cell, "compression_on_all_100_starter")
        - _solved(cell, "primitives_only"),
    )
    utility_gain = _cluster_ci(
        cells,
        lambda cell: _solved(cell, "utility_on_validation")
        - _solved(cell, "primitives_only"),
    )
    advantage = _cluster_ci(
        cells,
        lambda cell: _solved(cell, "utility_on_validation")
        - _solved(cell, "compression_on_all_100_starter"),
    )
    return {
        "label": label,
        "n": len(cells),
        "primitive": primitive[0],
        "past": past[0],
        "past_ci": past[1:],
        "utility": utility[0],
        "utility_ci": utility[1:],
        "compression_gain": compression_gain[0],
        "compression_gain_ci": compression_gain[1:],
        "utility_gain": utility_gain[0],
        "utility_gain_ci": utility_gain[1:],
        "advantage": advantage[0],
        "advantage_ci": advantage[1:],
    }


def figure_summaries(results: dict) -> dict:
    main = [cell for cell in results["cells"] if cell["condition"] != "stale_reversed"]

    def rho(cell: dict) -> float:
        return cell["world_metadata"]["realized_rho"]["realized_start_test"]

    similarity = [
        _comparison([cell for cell in main if rho(cell) < 0], "Different\nρ < 0"),
        _comparison([cell for cell in main if 0 <= rho(cell) < 0.5], "Moderate\n0 ≤ ρ < 0.5"),
        _comparison([cell for cell in main if rho(cell) >= 0.5], "Similar\nρ ≥ 0.5"),
    ]

    arms = {
        "past_compression": "compression_on_matched_25_starter",
        "past_utility": "utility_on_matched_25_starter",
        "future_compression": "compression_on_validation_assisted",
        "future_utility": "utility_on_validation",
    }
    mechanism = {
        name: _cluster_ci(main, lambda cell, arm=arm: _solved(cell, arm))
        for name, arm in arms.items()
    }
    data_effect = _cluster_ci(
        main,
        lambda cell: _solved(cell, arms["future_compression"])
        - _solved(cell, arms["past_compression"]),
    )
    scoring_effect = _cluster_ci(
        main,
        lambda cell: _solved(cell, arms["future_utility"])
        - _solved(cell, arms["future_compression"]),
    )
    mechanism["data_effect"] = data_effect[0]
    mechanism["data_effect_ci"] = data_effect[1:]
    mechanism["scoring_effect"] = scoring_effect[0]
    mechanism["scoring_effect_ci"] = scoring_effect[1:]

    staleness = [
        _comparison(
            [cell for cell in results["cells"] if cell["condition"] == "reversed_a1"],
            "Fresh validation\naligned with test",
        ),
        _comparison(
            [cell for cell in results["cells"] if cell["condition"] == "stale_reversed"],
            "Stale validation\nlags test",
        ),
    ]

    practical = []
    for cell in main:
        utility = cell["arms"]["utility_on_validation"]
        compression = cell["arms"]["compression_on_all_100_starter"]
        savings = compression["summary"]["mean_search_cost"] - utility["summary"]["mean_search_cost"]
        if savings > 0:
            practical.append(
                math.ceil(
                    utility["selection_cost"]["selection_cost_candidate_programs_tried"]
                    / savings
                )
            )
    registered = [
        row["break_even_future_tasks"]
        for row in results["aggregates"]["break_even"]
        if row["condition"] != "stale_reversed"
        and row["break_even_future_tasks"] is not None
    ]
    cost = {
        "practical": {
            "finite": len(practical),
            "never": len(main) - len(practical),
            "finite_median": median(practical),
        },
        "registered": {
            "finite": len(registered),
            "never": len(main) - len(registered),
            "finite_median": median(registered),
        },
    }
    return {
        "similarity": similarity,
        "mechanism": mechanism,
        "staleness": staleness,
        "cost": cost,
    }


def compression_tradeoff_summaries(results: dict) -> dict:
    """Reconstruct the post-hoc compression comparison from stored selections."""
    cells = {
        (cell["seed"], cell["condition"]): cell
        for cell in results["cells"]
        if cell["condition"] in K_SWEEP_CONDITIONS
    }
    expected = {
        (seed, condition)
        for seed in FORMAL_SEEDS
        for condition in K_SWEEP_CONDITIONS
    }
    if set(cells) != expected:
        raise ValueError("expected all 180 primary seed-condition cells")

    config = load_config(DEFAULT_CONFIG_PATH)
    conditions = {condition.name: condition for condition in config.conditions}
    frontier = build_frontier_index(
        primitive_library(),
        ASSISTED_VALIDATION_SOLVE_CONFIG,
    )
    output_by_program = {
        program_to_string(entry.program): output
        for output, entry in frontier.entries.items()
    }
    starter_budget = SolveConfig().node_budget

    def solutions(tasks, budget):
        return [
            entry.program
            for task in tasks
            if (entry := frontier.entries.get(task.target)) is not None
            and entry.candidates_tried_at_first_solution <= budget
        ]

    def compression_counts(solution_programs, selected_programs):
        if len(selected_programs) != 10 or len(set(selected_programs)) != 10:
            raise ValueError("expected exactly 10 unique selected programs")
        try:
            helper_outputs = {
                output_by_program[program] for program in selected_programs
            }
        except KeyError as error:
            raise ValueError(f"selected program missing from frontier: {error.args[0]}") from error
        baseline = sum(_segmentation_cost(program, set()) for program in solution_programs)
        if baseline <= 0:
            raise ValueError("expected positive baseline segmentation cost")
        remaining = sum(
            _segmentation_cost(program, helper_outputs)
            for program in solution_programs
        )
        return baseline, baseline - remaining

    def removed_pct(counts):
        baseline, removed = counts
        return 100 * removed / baseline

    seed_rows = []
    for seed in FORMAL_SEEDS:
        seed_cells = [cells[seed, condition] for condition in K_SWEEP_CONDITIONS]
        anchor = seed_cells[0]
        for cell in seed_cells[1:]:
            if (
                cell["matched_starter_task_ids"]
                != anchor["matched_starter_task_ids"]
                or cell["matched_starter_solved_count"]
                != anchor["matched_starter_solved_count"]
                or cell["arms"]["compression_on_matched_25_starter"][
                    "selected_programs"
                ]
                != anchor["arms"]["compression_on_matched_25_starter"][
                    "selected_programs"
                ]
                or cell["arms"]["utility_on_matched_25_starter"][
                    "selected_programs"
                ]
                != anchor["arms"]["utility_on_matched_25_starter"][
                    "selected_programs"
                ]
            ):
                raise ValueError("starter selection inputs changed across conditions")

        starter_world = make_world(
            config,
            seed,
            conditions[K_SWEEP_CONDITIONS[0]],
        )
        starter_tasks = matched_starter_tasks(
            starter_world.tasks_start,
            seed,
            MATCHED_STARTER_COUNT,
        )
        if [task.id for task in starter_tasks] != anchor["matched_starter_task_ids"]:
            raise ValueError("regenerated starter task IDs do not match")
        starter_solutions = solutions(starter_tasks, starter_budget)
        if len(starter_solutions) != anchor["matched_starter_solved_count"]:
            raise ValueError("regenerated starter solution count does not match")

        starter_compression = removed_pct(
            compression_counts(
                starter_solutions,
                anchor["arms"]["compression_on_matched_25_starter"][
                    "selected_programs"
                ],
            )
        )
        starter_utility = removed_pct(
            compression_counts(
                starter_solutions,
                anchor["arms"]["utility_on_matched_25_starter"][
                    "selected_programs"
                ],
            )
        )
        validation_compression = []
        validation_utility = []
        validation_utility_on_starter = []
        for condition in K_SWEEP_CONDITIONS:
            cell = cells[seed, condition]
            world = make_world(config, seed, conditions[condition])
            if len(world.tasks_val) != MATCHED_STARTER_COUNT:
                raise ValueError("expected 25 validation tasks")
            validation_solutions = solutions(
                world.tasks_val,
                ASSISTED_VALIDATION_SOLVE_CONFIG.node_budget,
            )
            expected_solved = cell["compression_input_diagnostics"][
                "validation_assisted"
            ]["solved_task_count"]
            if len(validation_solutions) != expected_solved:
                raise ValueError("regenerated validation solution count does not match")
            compression_programs = cell["arms"][
                "compression_on_validation_assisted"
            ]["selected_programs"]
            utility_programs = cell["arms"]["utility_on_validation"][
                "selected_programs"
            ]
            validation_compression.append(
                compression_counts(validation_solutions, compression_programs)
            )
            validation_utility.append(
                compression_counts(validation_solutions, utility_programs)
            )
            validation_utility_on_starter.append(
                removed_pct(compression_counts(starter_solutions, utility_programs))
            )

        def pooled_pct(rows):
            return 100 * sum(removed for _, removed in rows) / sum(
                baseline for baseline, _ in rows
            )

        seed_rows.append(
            {
                "seed": seed,
                "starter_compression": starter_compression,
                "starter_utility": starter_utility,
                "validation_compression": pooled_pct(validation_compression),
                "validation_utility": pooled_pct(validation_utility),
                "validation_utility_on_starter": mean(
                    validation_utility_on_starter
                ),
            }
        )

    def estimate(key):
        point, low, high = _cluster_ci(seed_rows, lambda row: row[key])
        return {
            "mean": point,
            "median": median(row[key] for row in seed_rows),
            "ci": (low, high),
        }

    rows = []
    for label, prefix in (
        ("Starter solution programs", "starter"),
        ("Validation solution programs", "validation"),
    ):
        difference_key = f"{prefix}_difference"
        for row in seed_rows:
            row[difference_key] = row[f"{prefix}_utility"] - row[
                f"{prefix}_compression"
            ]
        rows.append(
            {
                "label": label,
                "compression": estimate(f"{prefix}_compression"),
                "utility": estimate(f"{prefix}_utility"),
                "difference": estimate(difference_key),
                "compression_wins": sum(
                    row[f"{prefix}_compression"] > row[f"{prefix}_utility"]
                    for row in seed_rows
                ),
            }
        )
    return {
        "rows": rows,
        "validation_utility_on_starter": estimate(
            "validation_utility_on_starter"
        ),
    }


def k_sweep_summaries(results: dict) -> list[dict]:
    summaries = []
    comparisons = {
        "compression_gain": lambda cell: _solved(
            cell, "compression_on_all_100_starter"
        )
        - _solved(cell, "primitives_only"),
        "utility_gain": lambda cell: _solved(cell, "utility_on_validation")
        - _solved(cell, "primitives_only"),
        "scoring_effect": lambda cell: _solved(cell, "utility_on_validation")
        - _solved(cell, "compression_on_validation_assisted"),
        "data_effect": lambda cell: _solved(
            cell, "compression_on_validation_assisted"
        )
        - _solved(cell, "compression_on_matched_25_starter"),
    }
    for k in K_SWEEP_VALUES:
        cells = [cell for cell in results["cells"] if cell["k"] == k]
        row = {"k": k}
        for name, comparison in comparisons.items():
            estimate = _cluster_ci(cells, comparison)
            row[name], row[f"{name}_ci"] = estimate[0], estimate[1:]
        summaries.append(row)
    return summaries


def capacity_curve_summaries(results: dict) -> dict:
    summary = capacity.analyze(results)
    by_seed: dict[int, list[float]] = defaultdict(list)
    for cell in results["cells"]:
        by_seed[cell["seed"]].append(
            float(cell["primitive"]["test_summary"]["solved_count"])
        )
    summary["primitive_mean"] = mean(mean(values) for values in by_seed.values())
    return summary


def capacity_diagnostic_summaries(results: dict) -> list[dict]:
    """Return seed-weighted, post-hoc summaries for the capacity diagnostics."""
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for cell in results["cells"]:
        by_seed[cell["seed"]].append(cell)

    rows = []
    for k in range(capacity.K_MAX + 1):
        seed_rows = []
        for cells in by_seed.values():
            compression = cells[0]["arms"][capacity.COMPRESSION_ALL_ARM]
            baseline_nodes = float(
                compression["round_diagnostics"][0]["objective_before"]
            )
            remaining_nodes = (
                baseline_nodes
                if k == 0
                else float(
                    compression["round_diagnostics"][k - 1]["objective_after"]
                )
            )
            seed_rows.append(
                {
                    "past_unique_outputs": mean(
                        float(
                            cell["arms"][capacity.COMPRESSION_ALL_ARM]["prefixes"][
                                k
                            ]["test_summary"]["unique_outputs"]
                        )
                        for cell in cells
                    ),
                    "utility_unique_outputs": mean(
                        float(
                            cell["arms"][capacity.UTILITY_ARM]["prefixes"][k][
                                "test_summary"
                            ]["unique_outputs"]
                        )
                        for cell in cells
                    ),
                    "compression_removed_pct": 100
                    * (baseline_nodes - remaining_nodes)
                    / baseline_nodes,
                    "past_test_solved": mean(
                        float(
                            cell["arms"][capacity.COMPRESSION_ALL_ARM]["prefixes"][
                                k
                            ]["test_summary"]["solved_count"]
                        )
                        for cell in cells
                    ),
                }
            )
        rows.append(
            {
                "k": k,
                **{
                    name: mean(seed_row[name] for seed_row in seed_rows)
                    for name in seed_rows[0]
                },
            }
        )
    return rows


def _budget_solved(cell: dict, arm: str, k: int, budget: int, split: str) -> float:
    rows = cell["arms"][arm]["prefixes"][k - 1]["budgets"]
    row = next(row for row in rows if row["budget"] == budget)
    return float(row[f"{split}_summary"]["solved_count"])


def _observed_sustained_catch_up(
    k1_ranks: Sequence[int], k2_ranks: Sequence[int]
) -> tuple[int | None, bool]:
    start, stop = budget_intervention.BUDGETS[0], budget_intervention.BUDGETS[-1]
    events = sorted(
        {start, stop}
        | {rank for rank in (*k1_ranks, *k2_ranks) if start < rank <= stop}
    )
    differences = [
        sum(rank <= budget for rank in k2_ranks)
        - sum(rank <= budget for rank in k1_ranks)
        for budget in events
    ]
    first_nonnegative = next(
        (index for index, difference in enumerate(differences) if difference >= 0),
        None,
    )
    reversed_after_crossing = first_nonnegative is not None and any(
        difference < 0 for difference in differences[first_nonnegative + 1 :]
    )
    if differences[-1] < 0:
        return None, reversed_after_crossing
    last_negative = max(
        (index for index, difference in enumerate(differences) if difference < 0),
        default=-1,
    )
    return events[last_negative + 1], reversed_after_crossing


def budget_intervention_summaries(results: dict) -> dict:
    cells = results["cells"]
    budgets = list(budget_intervention.BUDGETS)
    methods = {
        "past_compression": capacity.COMPRESSION_ALL_ARM,
        "future_utility": capacity.UTILITY_ARM,
    }
    seed_interactions = {
        name: capacity._seed_values(
            cells,
            lambda cell, arm=arm: (
                _budget_solved(cell, arm, 2, budgets[-1], "test")
                - _budget_solved(cell, arm, 1, budgets[-1], "test")
                - _budget_solved(cell, arm, 2, budgets[0], "test")
                + _budget_solved(cell, arm, 1, budgets[0], "test")
            ),
        )
        for name, arm in methods.items()
    }
    seeds = tuple(sorted(capacity.FORMAL_SEEDS))
    rng = random.Random(20260714)
    samples = tuple(
        tuple(rng.choices(seeds, k=len(seeds))) for _ in range(10_000)
    )
    interactions = capacity._simultaneous_intervals(seed_interactions, samples)

    comparison_arms = {
        "matched_compression": capacity.COMPRESSION_VALIDATION_ARM,
        "past_compression": capacity.COMPRESSION_ALL_ARM,
    }
    selector_values = {
        f"{comparison}_k{k}_{budget}": capacity._seed_values(
            cells,
            lambda cell, arm=arm, k=k, budget=budget: (
                _budget_solved(cell, capacity.UTILITY_ARM, k, budget, "test")
                - _budget_solved(cell, arm, k, budget, "test")
            ),
        )
        for comparison, arm in comparison_arms.items()
        for k in (1, 2)
        for budget in budgets
    }
    selector_intervals = capacity._simultaneous_intervals(selector_values, samples)
    selector_effects = {
        comparison: {
            k: [
                {
                    "budget": budget,
                    **selector_intervals[f"{comparison}_k{k}_{budget}"],
                }
                for budget in budgets
            ]
            for k in (1, 2)
        }
        for comparison in comparison_arms
    }

    summaries = {}
    for name, arm in methods.items():
        difference = []
        k1 = []
        k2 = []
        validation_difference = []
        for budget in budgets:
            k1_values = capacity._seed_values(
                cells,
                lambda cell, arm=arm, budget=budget: _budget_solved(
                    cell, arm, 1, budget, "test"
                ),
            )
            k2_values = capacity._seed_values(
                cells,
                lambda cell, arm=arm, budget=budget: _budget_solved(
                    cell, arm, 2, budget, "test"
                ),
            )
            validation_values = capacity._seed_values(
                cells,
                lambda cell, arm=arm, budget=budget: (
                    _budget_solved(cell, arm, 2, budget, "validation")
                    - _budget_solved(cell, arm, 1, budget, "validation")
                ),
            )
            k1.append(mean(k1_values.values()))
            k2.append(mean(k2_values.values()))
            difference.append(
                mean(k2_values[seed] - k1_values[seed] for seed in seeds)
            )
            validation_difference.append(mean(validation_values.values()))

        mechanism = [cell["mechanism"][arm] for cell in cells]
        lost_count = sum(rows[0]["lost_count"] for rows in mechanism)
        recovered_pct = [
            100 * sum(rows[index]["recovered_count"] for rows in mechanism) / lost_count
            for index in range(len(budgets))
        ]
        lost_at_size4 = 0
        for cell in cells:
            first = {
                row["task_id"]: row
                for row in cell["arms"][arm]["prefixes"][0]["test_targets"]
            }
            second = {
                row["task_id"]: row
                for row in cell["arms"][arm]["prefixes"][1]["test_targets"]
            }
            lost_at_size4 += sum(
                row["abstract_search_size"] == 4
                for task_id, row in first.items()
                if row["first_hit_rank"] is not None
                and row["first_hit_rank"] <= budgets[0]
                and not (
                    second[task_id]["first_hit_rank"] is not None
                    and second[task_id]["first_hit_rank"] <= budgets[0]
                )
            )
        summaries[name] = {
            "difference": difference,
            "k1": k1,
            "k2": k2,
            "validation_difference": validation_difference,
            "interaction": interactions[name],
            "positive_seed_interactions": sum(
                value > 0 for value in seed_interactions[name].values()
            ),
            "lost_count": lost_count,
            "lost_at_size4_pct": 100 * lost_at_size4 / lost_count,
            "recovered_pct": recovered_pct,
        }

    size4_access_pct = []
    primary_arms = tuple(methods.values())
    for budget in budgets:
        reached = sum(
            (
                rank := cell["arms"][arm]["prefixes"][1]["max_frontier"][
                    "first_size4_rank"
                ]
            )
            is not None
            and rank <= budget
            for cell in cells
            for arm in primary_arms
        )
        size4_access_pct.append(100 * reached / (len(cells) * len(primary_arms)))

    catch_up = {}
    for name, arm in methods.items():
        thresholds = []
        reversals = 0
        for cell in cells:
            prefixes = cell["arms"][arm]["prefixes"]
            ranks = [
                [
                    row["first_hit_rank"]
                    for row in prefix["test_targets"]
                    if row["first_hit_rank"] is not None
                ]
                for prefix in prefixes
            ]
            threshold, reversed_after_crossing = _observed_sustained_catch_up(
                ranks[0], ranks[1]
            )
            thresholds.append(threshold)
            reversals += reversed_after_crossing
        finite = sorted(value for value in thresholds if value is not None)
        median_budget = finite[len(thresholds) // 2 - 1] if len(finite) >= 90 else None
        curve_budgets = sorted({budgets[0], budgets[-1], *finite})
        catch_up[name] = {
            "thresholds": thresholds,
            "curve_budgets": curve_budgets,
            "curve_pct": [
                100 * sum(value <= budget for value in finite) / len(thresholds)
                for budget in curve_budgets
            ],
            "median_budget": median_budget,
            "caught_by": {
                budget: sum(value <= budget for value in finite)
                for budget in budgets
            },
            "right_censored": len(thresholds) - len(finite),
            "reversals": reversals,
        }
    return {
        "budgets": budgets,
        "methods": summaries,
        "size4_access_pct": size4_access_pct,
        "catch_up": catch_up,
        "selector_effects": selector_effects,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": GRAY,
            "axes.labelcolor": GRAY,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def _clean(axis, grid: str = "x") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid, color="#E8E8E8", linewidth=0.7)
    axis.set_axisbelow(True)


def _dumbbell(rows: Sequence[dict], title: str, *, height: float = 4.2):
    figure, axis = plt.subplots(figsize=(9.0, height), constrained_layout=True)
    positions = list(reversed(range(len(rows))))
    label_x = max(max(row["past"], row["utility"]) for row in rows) + 1.6
    for y, row in zip(positions, rows):
        axis.plot([row["past"], row["utility"]], [y, y], color=LIGHT_GRAY, linewidth=3)
        axis.scatter(row["past"], y, color=GRAY, s=45, zorder=3)
        axis.scatter(row["utility"], y, color=BLUE, s=45, zorder=3)
        low, high = row["advantage_ci"]
        axis.text(
            label_x,
            y,
            f"Utility {row['advantage']:+.1f}  [95% CI {low:+.1f}, {high:+.1f}]",
            va="center",
            color=BLUE if row["advantage"] > 0 else GRAY,
            fontsize=8,
        )
    first = rows[0]
    axis.text(first["past"], positions[0] + 0.28, "Past compression", ha="center", color=GRAY, fontsize=8)
    axis.text(first["utility"], positions[0] + 0.28, "Future utility", ha="center", color=BLUE, fontsize=8)
    axis.set_yticks(positions, [row["label"] for row in rows])
    left = min(min(row["past"], row["utility"]) for row in rows) - 1.4
    axis.set_xlim(left, label_x + 5.3)
    axis.set_xlabel("Mean test problems solved (of 100)")
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
    _clean(axis)
    return figure


def plot_similarity(summary: dict):
    rows = summary["similarity"]
    figure, (absolute, gains) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.6),
        sharey=True,
        gridspec_kw={"width_ratios": (1.35, 1)},
    )
    positions = list(reversed(range(len(rows))))

    difference_x = 68.3
    for y, row in zip(positions, rows):
        absolute.scatter(
            row["primitive"],
            y,
            marker="s",
            s=40,
            facecolor=LIGHT_GRAY,
            edgecolor="#777777",
            zorder=3,
        )
        absolute.plot(
            [row["past"], row["utility"]],
            [y, y],
            color=LIGHT_GRAY,
            linewidth=3,
        )
        absolute.scatter(row["past"], y, color=GRAY, s=45, zorder=3)
        absolute.scatter(row["utility"], y, color=BLUE, s=45, zorder=3)
        for value, color, shift, alignment in (
            (row["primitive"], "#777777", 0, "center"),
            (row["past"], GRAY, -0.12, "right"),
            (row["utility"], BLUE, 0.12, "left"),
        ):
            absolute.text(
                value + shift,
                y - 0.13,
                f"{value:.1f}",
                ha=alignment,
                va="top",
                color=color,
                fontsize=7,
            )
        low, high = row["advantage_ci"]
        absolute.text(
            difference_x,
            y,
            f"{row['advantage']:+.1f}  [95% CI {low:+.1f}, {high:+.1f}]",
            va="center",
            color=BLUE,
            fontsize=8,
        )

        for offset, key, ci_key, color, marker, name in (
            (0.13, "compression_gain", "compression_gain_ci", GRAY, "o", "Past compression"),
            (-0.13, "utility_gain", "utility_gain_ci", BLUE, "D", "Future utility"),
        ):
            value = row[key]
            ci_low, ci_high = row[ci_key]
            gains.errorbar(
                value,
                y + offset,
                xerr=[[value - ci_low], [ci_high - value]],
                color=color,
                marker=marker,
                capsize=3,
                markersize=5,
                linewidth=1.4,
            )
            label = f"{name} {value:+.1f}" if y == positions[0] else f"{value:+.1f}"
            gains.text(
                value,
                y + offset + (0.08 if offset > 0 else -0.08),
                label,
                ha="center",
                va="bottom" if offset > 0 else "top",
                color=color,
                fontsize=7,
            )

    first = rows[0]
    for value, label, color in (
        (first["primitive"], "Primitives\nonly", "#777777"),
        (first["past"], "Past\ncompression", GRAY),
        (first["utility"], "Future\nutility", BLUE),
    ):
        absolute.text(
            value,
            positions[0] + 0.31,
            label,
            ha="center",
            va="center",
            color=color,
            fontsize=8,
        )
    absolute.text(
        difference_x,
        positions[0] + 0.3,
        "Utility − compression",
        ha="left",
        color=BLUE,
        fontsize=8,
    )

    absolute.set_yticks(positions, [row["label"] for row in rows])
    absolute.set_xlim(55.2, 77.0)
    absolute.set_xlabel("Mean test problems solved (of 100)")
    absolute.set_title("A  Mean test problems solved", loc="left", fontsize=10, fontweight="bold")
    _clean(absolute)

    gains.axvline(0, color="#777777", linewidth=0.8)
    gains.set_yticks(positions, [row["label"] for row in rows])
    gains.tick_params(axis="y", left=True, labelleft=True, labelsize=8, pad=6)
    gains.set_xlim(-1.0, 12.5)
    gains.set_xlabel("Additional test problems solved (of 100)")
    gains.set_title("B  Gain over primitives", loc="left", fontsize=10, fontweight="bold")
    _clean(gains)

    absolute.set_ylim(-0.35, positions[0] + 0.48)
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.75, wspace=0.28)
    figure.suptitle(
        "At K=10, both abstraction methods beat primitives; utility had the highest mean solve rate.",
        x=0.12,
        y=0.96,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.12,
        0.04,
        "Similarity bands are descriptive; the registered analysis uses continuous realized ρ.",
        color=GRAY,
        fontsize=8,
    )
    return figure


def plot_k_sensitivity(summary: Sequence[dict]):
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    ks = [row["k"] for row in summary]
    panels = (
        (
            axes[0],
            "A  Gain over primitives",
            "Additional test problems solved (of 100)",
            (
                ("compression_gain", "compression_gain_ci", GRAY, "o", "Past compression"),
                ("utility_gain", "utility_gain_ci", BLUE, "D", "Future utility"),
            ),
        ),
        (
            axes[1],
            "B  Registered matched contrasts",
            "Paired change in test problems solved (of 100)",
            (
                (
                    "scoring_effect",
                    "scoring_effect_ci",
                    BLUE,
                    "D",
                    "Utility - matched compression",
                ),
                (
                    "data_effect",
                    "data_effect_ci",
                    GRAY,
                    "o",
                    "Compression: validation - starter",
                ),
            ),
        ),
    )
    for axis, title, ylabel, series in panels:
        axis.axhline(0, color="#777777", linewidth=0.8)
        for value_key, ci_key, color, marker, label in series:
            values = [row[value_key] for row in summary]
            intervals = [row[ci_key] for row in summary]
            axis.errorbar(
                ks,
                values,
                yerr=(
                    [value - low for value, (low, _) in zip(values, intervals)],
                    [high - value for value, (_, high) in zip(values, intervals)],
                ),
                color=color,
                marker=marker,
                capsize=3,
                linewidth=1.8,
                markersize=5,
            )
            for k, value in zip(ks, values):
                axis.annotate(
                    f"{value:+.1f}",
                    (k, value),
                    xytext=(0, 8 if value >= 0 else -13),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=7,
                )
            axis.text(10.25, values[-1], label, va="center", color=color, fontsize=8)
        axis.set_xticks(ks, [str(k) for k in ks])
        axis.set_xlim(1.2, 12.0)
        axis.set_xlabel("Selected subchains (K)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        _clean(axis, "y")
    axes[0].set_ylim(-9.0, 11.5)
    axes[1].set_ylim(-1.5, 7.0)
    figure.subplots_adjust(left=0.08, right=0.96, bottom=0.19, top=0.76, wspace=0.3)
    figure.suptitle(
        "Library size determined whether abstractions helped, while utility scoring remained advantageous.",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.04,
        "Intervals resample 30 seeds while keeping all six conditions within each seed.",
        color=GRAY,
        fontsize=8,
    )
    return figure


def plot_capacity_curve(summary: dict):
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    ks = list(range(capacity.K_MAX + 1))
    primitive = summary["primitive_mean"]

    performance = axes[0]
    performance.axhline(primitive, color="#777777", linewidth=1.1)
    performance.text(
        20.35,
        primitive,
        f"Primitives  {primitive:.2f}",
        va="center",
        color="#777777",
        fontsize=7.5,
    )
    performance_series = (
        ("past_compression_gain", GRAY, "Past compression"),
        ("utility_gain", BLUE, "Future utility"),
    )
    for name, color, label in performance_series:
        rows = summary["curves"][name]
        estimates = [primitive + rows[k]["estimate"] for k in ks]
        lows = [primitive + rows[k]["interval"][0] for k in ks]
        highs = [primitive + rows[k]["interval"][1] for k in ks]
        performance.fill_between(
            ks, lows, highs, color=color, alpha=0.11, linewidth=0
        )
        performance.plot(ks, estimates, color=color, linewidth=2.0)
        performance.text(
            20.35,
            estimates[-1],
            label,
            va="center",
            color=color,
            fontsize=7.5,
        )
        performance.scatter(11, estimates[11], color=color, s=24, zorder=3)
    performance.axvline(11, color=LIGHT_GRAY, linestyle=":", linewidth=1.0)
    performance.text(
        11,
        performance.get_ylim()[1],
        "Observed mean peaks\nat K=11",
        ha="center",
        va="top",
        color=GRAY,
        fontsize=7,
    )
    performance.set_xlim(0, 24.0)
    performance.set_xticks([0, 5, 10, 15, 20])
    performance.set_xlabel("Selected subchains (K)")
    performance.set_ylabel("Mean test problems solved (of 100)")
    performance.set_title(
        "A  Test performance", loc="left", fontsize=10, fontweight="bold"
    )
    _clean(performance, "y")

    comparisons = axes[1]
    comparisons.axhline(0, color="#777777", linewidth=0.8)
    comparison_series = (
        (
            "utility_minus_assisted_compression",
            BLUE,
            "-",
            "Same validation problems:\nutility - compression",
        ),
        (
            "utility_minus_past_compression",
            GRAY,
            "--",
            "Practical comparison:\nutility - past compression",
        ),
    )
    for name, color, linestyle, label in comparison_series:
        rows = summary["curves"][name]
        estimates = [rows[k]["estimate"] for k in ks]
        lows = [rows[k]["interval"][0] for k in ks]
        highs = [rows[k]["interval"][1] for k in ks]
        comparisons.fill_between(
            ks, lows, highs, color=color, alpha=0.11, linewidth=0
        )
        comparisons.plot(
            ks, estimates, color=color, linestyle=linestyle, linewidth=2.0
        )
        comparisons.text(
            20.35,
            estimates[-1],
            label,
            va="center",
            color=color,
            fontsize=7.3,
        )
    comparisons.set_xlim(0, 26.5)
    comparisons.set_xticks([0, 5, 10, 15, 20])
    comparisons.set_xlabel("Selected subchains (K)")
    comparisons.set_ylabel("Difference in test problems solved (of 100)")
    comparisons.set_title(
        "B  Utility beat matched compression from K=2 onward;\n"
        "its practical advantage was less certain.",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
    )
    _clean(comparisons, "y")

    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.19, top=0.76, wspace=0.3)
    figure.suptitle(
        "Test performance rose through about 11 abstractions, then fell sharply.",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.04,
        "Bands are simultaneous across K=1–20 within each trajectory; K=0 is the primitive baseline.",
        color=GRAY,
        fontsize=8,
    )
    return figure


def plot_capacity_diagnostics(summary: Sequence[dict]):
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    ks = [row["k"] for row in summary]

    breadth = axes[0]
    for name, color, label in (
        ("past_unique_outputs", GRAY, "Past compression"),
        ("utility_unique_outputs", BLUE, "Future utility"),
    ):
        values = [row[name] for row in summary]
        breadth.plot(ks, values, color=color, linewidth=2.0)
        breadth.text(
            20.35, values[-1], label, va="center", color=color, fontsize=8
        )
    breadth.axvline(2, color=LIGHT_GRAY, linestyle=":", linewidth=1.0)
    breadth.annotate(
        "K=2 drop",
        xy=(2, summary[2]["past_unique_outputs"]),
        xytext=(3.3, max(row["past_unique_outputs"] for row in summary) * 0.78),
        arrowprops={"arrowstyle": "->", "color": GRAY, "linewidth": 0.8},
        color=GRAY,
        fontsize=7.5,
    )
    breadth.annotate(
        "Later decline",
        xy=(18, summary[18]["utility_unique_outputs"]),
        xytext=(15.0, max(row["utility_unique_outputs"] for row in summary) * 0.80),
        arrowprops={"arrowstyle": "->", "color": BLUE, "linewidth": 0.8},
        color=BLUE,
        fontsize=7.5,
    )
    breadth.set_xlim(0, 24.0)
    breadth.set_xticks([0, 5, 10, 15, 20])
    breadth.set_xlabel("Selected subchains (K)")
    breadth.set_ylabel("Mean distinct grid outputs reached")
    breadth.set_title(
        "A  Search breadth (descriptive, post-hoc)",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    _clean(breadth, "y")

    compression = axes[1]
    removed = [row["compression_removed_pct"] for row in summary]
    solved = [row["past_test_solved"] for row in summary]
    compression.plot(
        removed, solved, color=GRAY, linewidth=1.8, marker="o", markersize=3
    )
    label_offsets = {0: (5, -12), 2: (5, -12), 11: (5, 5), 20: (-4, 6)}
    for k in (0, 2, 11, 20):
        compression.annotate(
            f"K={k}",
            (removed[k], solved[k]),
            xytext=label_offsets[k],
            textcoords="offset points",
            ha="right" if k == 20 else "left",
            color=GRAY,
            fontsize=7.5,
        )
    compression.set_xlabel("Starter-solution operation nodes removed (%)")
    compression.set_ylabel("Mean test problems solved (of 100)")
    compression.set_title(
        "B  Starter solutions kept getting shorter after\n"
        "test performance began to fall.",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
    )
    _clean(compression, "both")

    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.19, top=0.76, wspace=0.3)
    figure.suptitle(
        "Post-hoc diagnostics show where the capacity curve changed.",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.04,
        "Descriptive post-hoc summaries; no inferential bands are shown.",
        color=GRAY,
        fontsize=8,
    )
    return figure


def plot_budget_intervention(summary: dict):
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    budgets = summary["budgets"]

    effects = axes[0]
    effects.axhline(0, color="#777777", linewidth=0.9)
    effects.text(
        31_000,
        0.35,
        "K=1 and K=2 solve the same number",
        color="#777777",
        fontsize=7.5,
    )
    for name, color, label in (
        ("past_compression", GRAY, "Past compression"),
        ("future_utility", BLUE, "Future utility"),
    ):
        row = summary["methods"][name]
        effects.plot(
            budgets,
            row["difference"],
            color=color,
            marker="o",
            linewidth=2.0,
        )
        effects.text(
            91_500,
            row["difference"][-1] + (0.65 if name == "future_utility" else -0.65),
            label,
            color=color,
            va="center",
            fontsize=8,
        )
        low, high = row["interaction"]["interval"]
        effects.text(
            31_000,
            6.3 if name == "past_compression" else 5.1,
            f"{label}: 30k-to-90k change {row['interaction']['estimate']:+.2f} "
            f"[{low:.2f}, {high:.2f}]",
            color=color,
            fontsize=7.5,
        )
    effects.set_xlim(28_000, 112_000)
    effects.set_ylim(-10.0, 7.5)
    effects.set_xticks(budgets, ["30k", "45k", "60k", "90k"])
    effects.set_xlabel("Evaluation budget (candidate programs tried)")
    effects.set_ylabel("K=2 minus K=1 test problems solved (of 100)")
    effects.set_title(
        "A  K=2 changed from worse than K=1 to better.",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    _clean(effects, "y")

    mechanism = axes[1]
    mechanism.plot(
        budgets,
        summary["size4_access_pct"],
        color="#777777",
        linestyle=":",
        marker="o",
        linewidth=1.8,
    )
    for name, color, label in (
        ("past_compression", GRAY, "Past compression"),
        ("future_utility", BLUE, "Future utility"),
    ):
        mechanism.plot(
            budgets,
            summary["methods"][name]["recovered_pct"],
            color=color,
            marker="o",
            linewidth=2.0,
        )
        mechanism.text(
            47_500,
            summary["methods"][name]["recovered_pct"][1]
            + (2.0 if name == "past_compression" else -5.0),
            label,
            color=color,
            fontsize=8,
        )
    mechanism.text(
        47_500,
        104,
        "K=2 searches reaching size four",
        color="#777777",
        fontsize=8,
    )
    total_lost = sum(
        row["lost_count"] for row in summary["methods"].values()
    )
    mechanism.annotate(
        f"All {total_lost:,} lost cell-problem instances\nrecovered by 90k",
        xy=(90_000, 100),
        xytext=(67_000, 65),
        arrowprops={"arrowstyle": "->", "color": GRAY, "linewidth": 0.8},
        color=GRAY,
        fontsize=7.5,
    )
    mechanism.set_xlim(28_000, 112_000)
    mechanism.set_ylim(-5, 112)
    mechanism.set_xticks(budgets, ["30k", "45k", "60k", "90k"])
    mechanism.set_xlabel("Evaluation budget (candidate programs tried)")
    mechanism.set_ylabel("Searches or lost problems recovered (%)")
    mechanism.set_title(
        "B  Restoring size-four search recovered the exact\n"
        "problems lost at 30,000.",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
    )
    _clean(mechanism, "y")

    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.20, top=0.75, wspace=0.3)
    figure.suptitle(
        "More search budget reversed the K=2 loss and restored the missing solutions.",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.04,
        "Panel A annotations are simultaneous 95% intervals for the registered 30k-to-90k change; Panel B is descriptive.",
        color=GRAY,
        fontsize=8,
    )
    return figure


def plot_budget_followup(summary: dict):
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    budgets = summary["budgets"]

    catch_axis = axes[0]
    catch_rows = (
        ("past_compression", GRAY, "Past compression"),
        ("future_utility", BLUE, "Future utility"),
    )
    for name, color, label in catch_rows:
        row = summary["catch_up"][name]
        catch_axis.step(
            row["curve_budgets"],
            row["curve_pct"],
            where="post",
            color=color,
            linewidth=2.0,
        )
        anchor = 100 * row["caught_by"][60_000] / len(row["thresholds"])
        catch_axis.text(
            61_000,
            anchor + (3.0 if name == "past_compression" else -4.5),
            label,
            color=color,
            fontsize=8,
        )
    catch_axis.axvline(47_700, color="#999999", linestyle=":", linewidth=1.0)
    catch_axis.text(
        48_700,
        23,
        "50% by about 47.7k",
        color="#777777",
        fontsize=7.5,
        rotation=90,
        va="bottom",
    )
    catch_axis.text(
        88_500,
        8,
        "Still behind at 90k:\n21 compression cells\n19 utility cells",
        color="#777777",
        fontsize=7.5,
        ha="right",
    )
    catch_axis.set_xlim(28_000, 92_000)
    catch_axis.set_ylim(0, 100)
    catch_axis.set_xticks(budgets, ["30k", "45k", "60k", "90k"])
    catch_axis.set_xlabel("Evaluation budget (candidate programs tried)")
    catch_axis.set_ylabel("Cells with sustained K=2 catch-up through 90k (%)")
    catch_axis.set_title(
        "A  Half of cells sustained catch-up by about 47,700.",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
    )
    _clean(catch_axis, "y")

    comparison_panels = (
        (
            axes[1],
            "matched_compression",
            "B  Matched utility advantage was larger at K=2.",
            "Utility minus assisted compression",
        ),
        (
            axes[2],
            "past_compression",
            "C  Practical K=2 advantage was uncertain above 30k.",
            "Utility minus past compression",
        ),
    )
    for axis, comparison, title, ylabel in comparison_panels:
        axis.axhline(0, color="#777777", linewidth=0.9)
        for k, color, marker, offset in (
            (1, GRAY, "o", -0.45),
            (2, BLUE, "D", 0.35),
        ):
            rows = summary["selector_effects"][comparison][k]
            estimates = [row["estimate"] for row in rows]
            lows = [row["interval"][0] for row in rows]
            highs = [row["interval"][1] for row in rows]
            axis.errorbar(
                budgets,
                estimates,
                yerr=[
                    [estimate - low for estimate, low in zip(estimates, lows)],
                    [high - estimate for estimate, high in zip(estimates, highs)],
                ],
                color=color,
                marker=marker,
                linewidth=1.8,
                capsize=3,
            )
            axis.text(
                31_300,
                estimates[0] + offset,
                f"K={k}",
                color=color,
                fontsize=8,
            )
        axis.set_xlim(28_000, 92_000)
        axis.set_ylim(-1.7, 4.6)
        axis.set_xticks(budgets, ["30k", "45k", "60k", "90k"])
        axis.set_xlabel("Evaluation budget (candidate programs tried)")
        axis.set_ylabel(f"{ylabel}\n(test problems solved)")
        axis.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
        _clean(axis, "y")

    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.75, wspace=0.32)
    figure.suptitle(
        "Most K=2 libraries caught up by 60,000 candidates; utility's advantage depended on K and budget.",
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.07,
        0.04,
        "Panel A is descriptive over 180 condition-cells nested within 30 seeds. Panels B-C show exploratory simultaneous 95% intervals across all 16 contrasts.",
        color=GRAY,
        fontsize=8,
    )
    return figure


def plot_compression_tradeoff(summary: dict):
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    starter = summary["rows"][0]
    rows = (
        (
            "Greedy compression\nusing solved starter programs",
            starter["compression"],
            GRAY,
            "Greedy compression selection",
        ),
        (
            "Search utility\nusing the same 25 starter problems",
            starter["utility"],
            BLUE,
            "Search-utility selection",
        ),
        (
            "Search utility\nusing 25 validation problems",
            summary["validation_utility_on_starter"],
            BLUE,
            None,
        ),
    )
    positions = (2, 1, 0)
    for y, (label, values, color, legend_label) in zip(positions, rows):
        axis.barh(
            y,
            values["mean"],
            height=0.55,
            color=color,
            xerr=[
                [values["mean"] - values["ci"][0]],
                [values["ci"][1] - values["mean"]],
            ],
            error_kw={"ecolor": color, "capsize": 3, "linewidth": 1.2},
            label=legend_label,
        )
        axis.text(
            values["mean"] - 0.45,
            y,
            f"{values['mean']:.1f}%",
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="center",
        )

    axis.set_yticks(positions, [row[0] for row in rows])
    axis.set_xlim(0, 33)
    axis.set_ylim(-0.55, 2.55)
    axis.set_xlabel("Starter-solution operations removed (%)")
    figure.suptitle(
        "Utility-selected abstractions also shortened starter solutions,\n"
        "but less than greedy compression.",
        x=0.37,
        y=0.98,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    axis.legend(
        loc="center left",
        bbox_to_anchor=(0.37, 0.68),
        bbox_transform=figure.transFigure,
        ncol=2,
        frameon=False,
        fontsize=8,
        borderaxespad=0,
    )
    figure.text(
        0.37,
        0.79,
        "All three selected sets were applied back to the same available starter solution programs.",
        color="#666666",
        fontsize=8,
    )
    _clean(axis, "x")
    figure.subplots_adjust(left=0.37, right=0.98, bottom=0.15, top=0.58)
    return figure


def plot_mechanism(summary: dict):
    mechanism = summary["mechanism"]
    figure, axis = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    xs = [0, 1]
    for name, color, label in (
        ("compression", GRAY, "Compression procedure"),
        ("utility", BLUE, "Utility procedure"),
    ):
        points = [mechanism[f"past_{name}"], mechanism[f"future_{name}"]]
        axis.plot(xs, [point[0] for point in points], color=color, marker="o", linewidth=2)
        axis.text(1.03, points[1][0], label, va="center", color=color, fontsize=9)
    data_low, data_high = mechanism["data_effect_ci"]
    axis.text(
        0.5,
        mechanism["future_compression"][0] - 0.8,
        f"Compression source contrast: {mechanism['data_effect']:+.1f}  "
        f"[95% CI {data_low:+.1f}, {data_high:+.1f}]",
        ha="center",
        color=GRAY,
        fontsize=8,
    )
    compression = mechanism["future_compression"][0]
    utility = mechanism["future_utility"][0]
    midpoint = mean([compression, utility])
    axis.annotate(
        "",
        xy=(0.92, utility),
        xytext=(0.92, compression),
        arrowprops={"arrowstyle": "<->", "color": BLUE, "linewidth": 1.2},
    )
    scoring_low, scoring_high = mechanism["scoring_effect_ci"]
    axis.text(
        0.89,
        midpoint,
        f"Matched procedure contrast: {mechanism['scoring_effect']:+.1f}\n"
        f"[95% CI {scoring_low:+.1f}, {scoring_high:+.1f}]",
        ha="right",
        va="center",
        color=BLUE,
        fontsize=8,
    )
    axis.set_xticks(xs, ["Past problem data", "Future-like problem data"])
    axis.set_xlim(-0.12, 1.42)
    axis.set_ylim(59.5, 68.0)
    axis.set_ylabel("Mean test problems solved (of 100)")
    axis.set_title(
        "Utility performed better; changing compression's problem source had no detectable gain.",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    _clean(axis, "y")
    return figure


def plot_staleness(summary: dict):
    return _dumbbell(
        summary["staleness"],
        "The stale-validation comparison was inconclusive.",
        height=3.4,
    )


def plot_cost(summary: dict):
    figure, axis = plt.subplots(figsize=(9.0, 3.7), constrained_layout=True)
    rows = [
        ("Standard past compression\npractical comparison", summary["cost"]["practical"]),
        ("Assisted future compression\nregistered comparison", summary["cost"]["registered"]),
    ]
    for y, (label, row) in zip([1, 0], rows):
        finite_pct = 100 * row["finite"] / 180
        never_pct = 100 - finite_pct
        axis.barh(y, finite_pct, color=BLUE, height=0.45)
        axis.barh(y, never_pct, left=finite_pct, color=LIGHT_GRAY, height=0.45)
        axis.text(finite_pct / 2, y, f"{row['finite']}/180 break even", ha="center", va="center", color="white", fontsize=8)
        axis.text(finite_pct + never_pct / 2, y, f"{row['never']}/180 never", ha="center", va="center", color=GRAY, fontsize=8)
        axis.text(102, y, f"Finite median: {math.ceil(row['finite_median']):,} problems", va="center", fontsize=8, color=GRAY)
    axis.set_yticks([1, 0], [row[0] for row in rows])
    axis.set_xlim(0, 130)
    axis.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axis.set_xlabel("Share of 180 seed-condition cells")
    axis.set_title(
        "Utility is a long-horizon investment and does not pay back in every cell.",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(0, -0.32, "Break-even uses utility's upfront selection cost and its per-problem search savings.", transform=axis.transAxes, color=GRAY, fontsize=8)
    _clean(axis)
    return figure


def _save(figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths


def make_figures(
    results: dict,
    k_sweep_results: dict,
    output_dir: Path,
    capacity_curve_results: dict | None = None,
    budget_intervention_results: dict | None = None,
) -> list[Path]:
    _style()
    summary = figure_summaries(results)
    compression_tradeoff_summary = compression_tradeoff_summaries(results)
    k_sweep_summary = k_sweep_summaries(k_sweep_results)
    figures = [
        ("figure_1_similarity", plot_similarity(summary)),
        ("figure_2_mechanism", plot_mechanism(summary)),
        ("figure_3_staleness", plot_staleness(summary)),
        ("figure_4_cost", plot_cost(summary)),
        ("figure_5_k_sensitivity", plot_k_sensitivity(k_sweep_summary)),
        (
            "figure_10_compression_tradeoff",
            plot_compression_tradeoff(compression_tradeoff_summary),
        ),
    ]
    if capacity_curve_results is not None:
        capacity_summary = capacity_curve_summaries(capacity_curve_results)
        diagnostic_summary = capacity_diagnostic_summaries(capacity_curve_results)
        figures.extend(
            [
                ("figure_6_capacity_curve", plot_capacity_curve(capacity_summary)),
                (
                    "figure_7_capacity_diagnostics",
                    plot_capacity_diagnostics(diagnostic_summary),
                ),
            ]
        )
    if budget_intervention_results is not None:
        budget_summary = budget_intervention_summaries(budget_intervention_results)
        figures.append(
            (
                "figure_8_budget_intervention",
                plot_budget_intervention(budget_summary),
            )
        )
        figures.append(
            (
                "figure_9_budget_followup",
                plot_budget_followup(budget_summary),
            )
        )
    return [path for name, figure in figures for path in _save(figure, output_dir / name)]
