"""Frozen schema and analysis contract for the capacity-curve experiment."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
import random
import re
from statistics import mean, stdev


EXPERIMENT_NAME = "full_selection_experiment_capacity_curve"
SMOKE_EXPERIMENT_NAME = "full_selection_experiment_capacity_curve_smoke"
FORMAL_SEEDS = tuple(range(6541, 6571))
SMOKE_SEEDS = (6511,)
CONDITIONS = (
    "reversed_a0",
    "reversed_a05",
    "reversed_a1",
    "permuted_a0",
    "permuted_a05",
    "permuted_a1",
)
CONDITION_SPECS = {
    "reversed_a0": {"name": "reversed_a0", "alt_kind": "reversed", "alpha_val": 0.0, "alpha_test": 0.0},
    "reversed_a05": {"name": "reversed_a05", "alt_kind": "reversed", "alpha_val": 0.5, "alpha_test": 0.5},
    "reversed_a1": {"name": "reversed_a1", "alt_kind": "reversed", "alpha_val": 1.0, "alpha_test": 1.0},
    "permuted_a0": {"name": "permuted_a0", "alt_kind": "permuted", "alpha_val": 0.0, "alpha_test": 0.0},
    "permuted_a05": {"name": "permuted_a05", "alt_kind": "permuted", "alpha_val": 0.5, "alpha_test": 0.5},
    "permuted_a1": {"name": "permuted_a1", "alt_kind": "permuted", "alpha_val": 1.0, "alpha_test": 1.0},
}
SMOKE_CONDITIONS = ("reversed_a0",)
K_MAX = 20
MENU_CAP = 50
MOTIF_COUNT = 12
PRIMITIVE_LIBRARY_SIZE = 6
RANDOM_DRAWS = 20
VALIDATION_TASK_COUNT = 25
TEST_TASK_COUNT = 100

COMPRESSION_ALL_ARM = "compression_on_all_100_starter"
COMPRESSION_VALIDATION_ARM = "compression_on_validation_assisted"
UTILITY_ARM = "utility_on_validation"
TEST_PEEK_ARM = "best_k_from_c_oracle"
RANDOM_ARM = "random_k"
COMPRESSION_ARMS = (COMPRESSION_ALL_ARM, COMPRESSION_VALIDATION_ARM)
GREEDY_ARMS = COMPRESSION_ARMS + (UTILITY_ARM, TEST_PEEK_ARM)
COST_KEYS = (
    "selection_cost_candidate_programs_tried",
    "input_solution_search_candidate_programs_tried",
    "trial_libraries_evaluated",
    "segmentation_evaluations",
    "solution_segmentations_evaluated",
    "frontier_candidates_tried_total",
)
HASH_KEYS = ("hidden_motifs", "p_start", "starter_tasks", "candidate_menu")
BOOTSTRAP_SEED = 20260713
BOOTSTRAP_DRAWS = 10_000


def zero_cost() -> dict:
    return {key: 0 for key in COST_KEYS}


def canonical_hash(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def registration(*, smoke: bool) -> dict:
    seeds = SMOKE_SEEDS if smoke else FORMAL_SEEDS
    conditions = SMOKE_CONDITIONS if smoke else CONDITIONS
    return {
        "experiment_name": SMOKE_EXPERIMENT_NAME if smoke else EXPERIMENT_NAME,
        "registration": "R13",
        "registration_timing": "before_formal_results",
        "smoke": smoke,
        "seeds": list(seeds),
        "conditions": list(conditions),
        "cell_count": len(seeds) * len(conditions),
        "k_max": K_MAX,
        "prefixes": list(range(K_MAX + 1)),
        "candidate_menu_cap": MENU_CAP,
        "minimum_candidate_menu_size": K_MAX,
        "latent_motif_count": MOTIF_COUNT,
        "arms": ["primitives_only", *GREEDY_ARMS, RANDOM_ARM],
        "random_draws": RANDOM_DRAWS,
        "starter_task_count": 100,
        "validation_task_count": VALIDATION_TASK_COUNT,
        "test_task_count": TEST_TASK_COUNT,
        "evaluation_solve_config": {
            "node_budget": 30_000,
            "max_program_size": 7,
            "max_solutions": 1,
        },
        "assisted_validation_solve_config": {
            "node_budget": 90_000,
            "max_program_size": 7,
            "max_solutions": 1,
        },
        "random_seed_rule": "seed:draw",
        "no_early_stopping": True,
        "primary_contrasts": [
            "utility20_minus_assisted_compression20",
            "scoring_effect20_minus_scoring_effect10",
            "utility20_minus_past_compression20",
            "utility20_minus_utility10",
            "past_compression20_minus_past_compression10",
        ],
        "bootstrap": {
            "seed": 20260713,
            "draws": 10_000,
            "cluster": "seed",
            "conditions_per_seed": 6,
            "critical_value": "single_step_max_t_nearest_rank_element_9499",
        },
    }


def bootstrap_samples() -> tuple[tuple[int, ...], ...]:
    seeds = tuple(sorted(FORMAL_SEEDS))
    rng = random.Random(BOOTSTRAP_SEED)
    return tuple(tuple(rng.choices(seeds, k=len(seeds))) for _ in range(BOOTSTRAP_DRAWS))


def nearest_rank_95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("nearest-rank input cannot be empty")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def analyze(payload: dict) -> dict:
    validate_aggregate(payload, formal=True)
    cells = payload["cells"]
    samples = bootstrap_samples()

    primary_functions = {
        "utility20_minus_assisted_compression20": lambda cell: (
            _test_solved(cell, UTILITY_ARM, 20)
            - _test_solved(cell, COMPRESSION_VALIDATION_ARM, 20)
        ),
        "scoring_effect20_minus_scoring_effect10": lambda cell: (
            _test_solved(cell, UTILITY_ARM, 20)
            - _test_solved(cell, COMPRESSION_VALIDATION_ARM, 20)
            - _test_solved(cell, UTILITY_ARM, 10)
            + _test_solved(cell, COMPRESSION_VALIDATION_ARM, 10)
        ),
        "utility20_minus_past_compression20": lambda cell: (
            _test_solved(cell, UTILITY_ARM, 20)
            - _test_solved(cell, COMPRESSION_ALL_ARM, 20)
        ),
        "utility20_minus_utility10": lambda cell: (
            _test_solved(cell, UTILITY_ARM, 20)
            - _test_solved(cell, UTILITY_ARM, 10)
        ),
        "past_compression20_minus_past_compression10": lambda cell: (
            _test_solved(cell, COMPRESSION_ALL_ARM, 20)
            - _test_solved(cell, COMPRESSION_ALL_ARM, 10)
        ),
    }
    primary_values = {
        name: _seed_values(cells, function)
        for name, function in primary_functions.items()
    }
    primary = _simultaneous_intervals(primary_values, samples)

    curve_functions = {
        "past_compression_gain": lambda cell, k: (
            _test_solved(cell, COMPRESSION_ALL_ARM, k) - _primitive_solved(cell)
        ),
        "assisted_compression_gain": lambda cell, k: (
            _test_solved(cell, COMPRESSION_VALIDATION_ARM, k)
            - _primitive_solved(cell)
        ),
        "utility_gain": lambda cell, k: (
            _test_solved(cell, UTILITY_ARM, k) - _primitive_solved(cell)
        ),
        "mean_random_gain": lambda cell, k: (
            _random_test_solved(cell, k) - _primitive_solved(cell)
        ),
        "test_peeking_gain": lambda cell, k: (
            _test_solved(cell, TEST_PEEK_ARM, k) - _primitive_solved(cell)
        ),
        "utility_minus_assisted_compression": lambda cell, k: (
            _test_solved(cell, UTILITY_ARM, k)
            - _test_solved(cell, COMPRESSION_VALIDATION_ARM, k)
        ),
        "utility_minus_past_compression": lambda cell, k: (
            _test_solved(cell, UTILITY_ARM, k)
            - _test_solved(cell, COMPRESSION_ALL_ARM, k)
        ),
    }
    curves = {}
    for name, function in curve_functions.items():
        values = {
            str(k): _seed_values(cells, lambda cell, k=k: function(cell, k))
            for k in range(1, K_MAX + 1)
        }
        intervals = _simultaneous_intervals(values, samples)
        curves[name] = {
            0: {
                "estimate": 0.0,
                "standard_error": 0.0,
                "interval": [0.0, 0.0],
                "zero_standard_error": True,
            },
            **{int(k): row for k, row in intervals.items()},
        }

    post_k10_functions = {
        "past_compression": lambda cell, k: _test_solved(
            cell, COMPRESSION_ALL_ARM, k
        ),
        "assisted_compression": lambda cell, k: _test_solved(
            cell, COMPRESSION_VALIDATION_ARM, k
        ),
        "utility": lambda cell, k: _test_solved(cell, UTILITY_ARM, k),
        "mean_random": _random_test_solved,
        "test_peeking": lambda cell, k: _test_solved(cell, TEST_PEEK_ARM, k),
    }
    post_k10_changes = {}
    for name, function in post_k10_functions.items():
        values = {
            str(k): _seed_values(
                cells,
                lambda cell, k=k: function(cell, k) - function(cell, 10),
            )
            for k in range(11, K_MAX + 1)
        }
        post_k10_changes[name] = {
            int(k): row
            for k, row in _simultaneous_intervals(values, samples).items()
        }
    return {
        "primary": primary,
        "curves": curves,
        "post_k10_changes": post_k10_changes,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "nearest_rank_index": 9499,
        },
    }


def _simultaneous_intervals(
    values_by_name: dict[str, dict[int, float]],
    samples: Sequence[Sequence[int]],
) -> dict[str, dict]:
    estimates = {
        name: mean(by_seed[seed] for seed in sorted(by_seed))
        for name, by_seed in values_by_name.items()
    }
    standard_errors = {
        name: (
            stdev(by_seed[seed] for seed in sorted(by_seed))
            / math.sqrt(len(by_seed))
        )
        for name, by_seed in values_by_name.items()
    }
    active = [name for name, error in standard_errors.items() if error > 0]
    maxima = []
    if active:
        for sample in samples:
            maxima.append(
                max(
                    abs(
                        (
                            mean(values_by_name[name][seed] for seed in sample)
                            - estimates[name]
                        )
                        / standard_errors[name]
                    )
                    for name in active
                )
            )
        critical = nearest_rank_95(maxima)
    else:
        critical = 0.0
    return {
        name: {
            "estimate": estimates[name],
            "standard_error": standard_errors[name],
            "interval": (
                [estimates[name], estimates[name]]
                if standard_errors[name] == 0
                else [
                    estimates[name] - critical * standard_errors[name],
                    estimates[name] + critical * standard_errors[name],
                ]
            ),
            "zero_standard_error": standard_errors[name] == 0,
        }
        for name in values_by_name
    }


def _seed_values(cells: Sequence[dict], function) -> dict[int, float]:
    by_seed: dict[int, list[float]] = {}
    for cell in cells:
        by_seed.setdefault(cell["seed"], []).append(float(function(cell)))
    return {seed: mean(values) for seed, values in by_seed.items()}


def _test_solved(cell: dict, arm: str, k: int) -> float:
    return float(cell["arms"][arm]["prefixes"][k]["test_summary"]["solved_count"])


def _primitive_solved(cell: dict) -> float:
    return float(cell["primitive"]["test_summary"]["solved_count"])


def _random_test_solved(cell: dict, k: int) -> float:
    return mean(
        float(draw["prefixes"][k]["test_summary"]["solved_count"])
        for draw in cell["arms"][RANDOM_ARM]["draws"]
    )


def validate_cell(cell: dict, *, formal: bool) -> None:
    if not isinstance(cell, dict):
        _fail("cell must be an object")
    expected_name = EXPERIMENT_NAME if formal else SMOKE_EXPERIMENT_NAME
    expected_seeds = FORMAL_SEEDS if formal else SMOKE_SEEDS
    expected_conditions = CONDITIONS if formal else SMOKE_CONDITIONS
    seed = cell.get("seed")
    if cell.get("experiment_name") != expected_name:
        _fail("experiment_name does not match registration")
    if type(seed) is not int or seed not in expected_seeds:
        _fail("seed does not match registration")
    if type(cell.get("world_seed")) is not int or cell["world_seed"] != seed:
        _fail("world_seed must exactly match seed")
    if cell.get("condition") not in expected_conditions:
        _fail("condition does not match registration")
    if cell.get("formal_seed") is not formal:
        _fail("formal_seed provenance does not match run")
    if cell.get("motif_count") != MOTIF_COUNT:
        _fail("motif count must equal 12")
    if cell.get("starter_task_count") != 100:
        _fail("starter task count must equal 100")
    if cell.get("k_max") != K_MAX:
        _fail("k_max does not match registration")
    if cell.get("random_draws") != RANDOM_DRAWS:
        _fail("random_draws must equal 20")
    if cell.get("validation_task_count") != VALIDATION_TASK_COUNT:
        _fail("validation task count must equal 25")
    if cell.get("test_task_count") != TEST_TASK_COUNT:
        _fail("test task count must equal 100")
    _validate_world_metadata(cell.get("world_metadata"), cell["condition"])

    programs = cell.get("candidate_menu_programs")
    if (
        not isinstance(programs, list)
        or not all(isinstance(program, str) for program in programs)
        or not K_MAX <= len(programs) <= MENU_CAP
        or len(set(programs)) != len(programs)
    ):
        _fail("menu size must be 20--50 unique programs")
    menu = cell.get("menu")
    _validate_menu(menu, len(programs))
    _validate_hashes(cell.get("shared_hashes"))
    if cell["shared_hashes"]["candidate_menu"] != canonical_hash(programs):
        _fail("candidate-menu hash does not match ordered programs")
    input_diagnostics = cell.get("input_solution_diagnostics")
    _validate_input_diagnostics(input_diagnostics)
    _validate_timings(cell)

    primitive = cell.get("primitive")
    if not isinstance(primitive, dict):
        _fail("primitive summaries are required")
    _validate_summary_pair(
        primitive.get("validation_summary"),
        primitive.get("test_summary"),
        selected_count=0,
    )

    arms = cell.get("arms")
    if not isinstance(arms, dict) or set(arms) != {*GREEDY_ARMS, RANDOM_ARM}:
        _fail("arm set does not match registration")
    menu_members = set(programs)
    for name in GREEDY_ARMS:
        assisted_acquisition = (
            input_diagnostics["validation_assisted"][
                "candidate_programs_tried_total"
            ]
            if name == COMPRESSION_VALIDATION_ARM
            else None
        )
        _validate_greedy_arm(
            name,
            arms[name],
            menu_members,
            primitive,
            menu_size=len(programs),
            input_diagnostics=input_diagnostics,
            assisted_acquisition=assisted_acquisition,
        )
    _validate_random_arm(
        arms[RANDOM_ARM], menu_members, primitive, menu_size=len(programs)
    )


def validate_aggregate(payload: dict, *, formal: bool) -> None:
    expected_registration = registration(smoke=not formal)
    expected_name = EXPERIMENT_NAME if formal else SMOKE_EXPERIMENT_NAME
    expected_seeds = FORMAL_SEEDS if formal else SMOKE_SEEDS
    expected_conditions = CONDITIONS if formal else SMOKE_CONDITIONS
    if not isinstance(payload, dict) or payload.get("experiment_name") != expected_name:
        _fail("aggregate experiment_name does not match registration")
    if payload.get("smoke") is not (not formal):
        _fail("aggregate smoke flag does not match registration")
    if payload.get("registration") != expected_registration:
        _fail("aggregate registration does not match frozen values")
    if "aggregates" in payload:
        _fail("raw aggregate must not contain statistical aggregates")
    cells = payload.get("cells")
    expected = {
        (seed, condition) for seed in expected_seeds for condition in expected_conditions
    }
    if not isinstance(cells, list) or len(cells) != len(expected):
        _fail(f"aggregate must contain exactly {len(expected)} cells")
    actual = [(cell.get("seed"), cell.get("condition")) for cell in cells]
    if set(actual) != expected or len(set(actual)) != len(actual):
        _fail("aggregate cells must be unique and complete")
    expected_order = sorted(actual, key=lambda item: (item[0], CONDITIONS.index(item[1])))
    if actual != expected_order:
        _fail("aggregate cells must use deterministic seed-condition order")
    for cell in cells:
        validate_cell(cell, formal=formal)
    _validate_seed_pairing(cells)


def _validate_greedy_arm(
    name: str,
    arm: dict,
    menu: set[str],
    primitive: dict,
    *,
    menu_size: int,
    input_diagnostics: dict,
    assisted_acquisition: int | None,
) -> None:
    if not isinstance(arm, dict):
        _fail(f"{name} must be an object")
    selected = arm.get("selected_programs")
    if (
        not isinstance(selected, list)
        or len(selected) != K_MAX
        or len(set(selected)) != K_MAX
        or not set(selected) <= menu
    ):
        _fail(f"{name} selected path must contain 20 unique menu programs")
    _validate_prefixes(
        name,
        arm.get("prefixes"),
        primitive,
        random_path=False,
        menu_size=menu_size,
        input_diagnostics=input_diagnostics,
        assisted_acquisition=assisted_acquisition,
    )
    diagnostics = arm.get("round_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != K_MAX:
        _fail(f"{name} must contain 20 round diagnostics")
    prefixes = arm["prefixes"]
    for index, row in enumerate(diagnostics, start=1):
        _validate_diagnostic(
            name,
            row,
            selected[index - 1],
            index,
            remaining_count=menu_size - index + 1,
        )
        if index > 1 and row["objective_before"] != diagnostics[index - 2][
            "objective_after"
        ]:
            _fail(f"{name} round trace continuity is broken")
        if name in {UTILITY_ARM, TEST_PEEK_ARM}:
            summary_key = (
                "validation_summary" if name == UTILITY_ARM else "test_summary"
            )
            expected_before = _objective_from_summary(prefixes[index - 1][summary_key])
            expected_after = _objective_from_summary(prefixes[index][summary_key])
            if (
                row["objective_before"] != expected_before
                or row["objective_after"] != expected_after
            ):
                _fail(f"{name} objective must match its prefix summary")


def _validate_random_arm(
    arm: dict, menu: set[str], primitive: dict, *, menu_size: int
) -> None:
    draws = arm.get("draws") if isinstance(arm, dict) else None
    if not isinstance(draws, list) or len(draws) != RANDOM_DRAWS:
        _fail("random arm must contain exactly 20 draws")
    if [draw.get("draw") for draw in draws if isinstance(draw, dict)] != list(
        range(RANDOM_DRAWS)
    ):
        _fail("random draw identifiers must be exactly 0--19")
    for draw in draws:
        selected = draw.get("selected_programs")
        if (
            not isinstance(selected, list)
            or len(selected) != K_MAX
            or len(set(selected)) != K_MAX
            or not set(selected) <= menu
        ):
            _fail("random path must contain 20 unique menu programs")
        _validate_prefixes(
            RANDOM_ARM,
            draw.get("prefixes"),
            primitive,
            random_path=True,
            menu_size=menu_size,
            input_diagnostics=None,
            assisted_acquisition=None,
        )


def _validate_prefixes(
    name: str,
    prefixes,
    primitive: dict,
    *,
    random_path: bool,
    menu_size: int,
    input_diagnostics: dict | None,
    assisted_acquisition: int | None,
) -> None:
    if not isinstance(prefixes, list) or len(prefixes) != K_MAX + 1:
        _fail(f"{name} must contain exactly 21 prefix rows")
    if [row.get("k") for row in prefixes if isinstance(row, dict)] != list(
        range(K_MAX + 1)
    ):
        _fail(f"{name} prefix order must be exactly 0--20")
    previous = zero_cost()
    for k, row in enumerate(prefixes):
        if "selected_programs" in row:
            _fail(f"{name} prefix rows must not duplicate selected programs")
        _validate_summary_pair(
            row.get("validation_summary"),
            row.get("test_summary"),
            selected_count=k,
        )
        cost = row.get("selection_cost")
        _validate_cost(cost)
        _validate_arm_cost(
            name,
            cost,
            k,
            menu_size,
            input_diagnostics=input_diagnostics,
            assisted_acquisition=assisted_acquisition,
        )
        if k == 0:
            if row["validation_summary"] != primitive["validation_summary"]:
                _fail(f"{name} K=0 validation summary must equal primitives")
            if row["test_summary"] != primitive["test_summary"]:
                _fail(f"{name} K=0 test summary must equal primitives")
            if cost != zero_cost():
                _fail(f"{name} K=0 selection cost must be zero")
        if not random_path and any(cost[key] < previous[key] for key in COST_KEYS):
            _fail(f"{name} prefix selection costs must be cumulative")
        previous = cost


def _validate_diagnostic(
    name: str,
    row: dict,
    selected: str,
    round_number: int,
    *,
    remaining_count: int,
) -> None:
    if (
        not isinstance(row, dict)
        or set(row)
        != {
            "round",
            "selected_program",
            "objective_before",
            "objective_after",
            "marginal_objective_change",
            "direction",
            "best_tie_count",
        }
        or row.get("round") != round_number
    ):
        _fail(f"{name} diagnostic rounds must align with selection order")
    if row.get("selected_program") != selected:
        _fail(f"{name} diagnostic selection must align with selected path")
    if (
        type(row.get("best_tie_count")) is not int
        or not 1 <= row["best_tie_count"] <= remaining_count
    ):
        _fail(f"{name} diagnostic tie count exceeds remaining candidates")
    before = row.get("objective_before")
    after = row.get("objective_after")
    if name in COMPRESSION_ARMS:
        if (
            type(before) is not int
            or type(after) is not int
            or before < 0
            or after < 0
        ):
            _fail(f"{name} compression objectives must be nonnegative integers")
        marginal = before - after
        if row.get("marginal_objective_change") != marginal:
            _fail(f"{name} diagnostic marginal does not match objectives")
        direction = _direction(marginal)
    else:
        _validate_trial_objective(before, name)
        _validate_trial_objective(after, name)
        marginal = {
            "mean_search_cost_reduction": (
                before["mean_search_cost"] - after["mean_search_cost"]
            ),
            "solved_count_change": after["solved_count"] - before["solved_count"],
        }
        if row.get("marginal_objective_change") != marginal:
            _fail(f"{name} diagnostic marginal does not match objectives")
        objective = "utility" if name == UTILITY_ARM else "solved"
        before_rank = _trial_rank(before, objective)
        after_rank = _trial_rank(after, objective)
        direction = (
            "positive"
            if after_rank > before_rank
            else "zero" if after_rank == before_rank else "negative"
        )
    if row.get("direction") != direction:
        _fail(f"{name} diagnostic direction does not match objectives")


def _validate_seed_pairing(cells: Sequence[dict]) -> None:
    by_seed: dict[int, list[dict]] = {}
    for cell in cells:
        by_seed.setdefault(cell["seed"], []).append(cell)
    for seed_cells in by_seed.values():
        reference = seed_cells[0]
        for cell in seed_cells[1:]:
            if cell["shared_hashes"] != reference["shared_hashes"]:
                if (
                    cell["shared_hashes"]["starter_tasks"]
                    != reference["shared_hashes"]["starter_tasks"]
                ):
                    _fail("shared starter hash differs within seed")
                _fail("shared world hash differs within seed")
            if cell["candidate_menu_programs"] != reference["candidate_menu_programs"]:
                _fail("ordered candidate menu differs within seed")
            if cell["menu"] != reference["menu"]:
                _fail("shared menu diagnostics differ within seed")
            if (
                cell["input_solution_diagnostics"]["all_100_starter"]
                != reference["input_solution_diagnostics"]["all_100_starter"]
            ):
                _fail("shared starter diagnostics differ within seed")
            if _primitive_frontier(cell) != _primitive_frontier(reference):
                _fail("shared primitive frontier diagnostics differ within seed")
            if _shared_c100(cell) != _shared_c100(reference):
                _fail("shared C100 selection differs within seed")
            if _random_paths(cell) != _random_paths(reference):
                _fail("random paths differ within seed")


def _random_paths(cell: dict) -> list[dict]:
    return [
        {
            "selected_programs": draw["selected_programs"],
            "prefix_frontiers": _prefix_frontiers(draw["prefixes"]),
        }
        for draw in cell["arms"][RANDOM_ARM]["draws"]
    ]


def _shared_c100(cell: dict) -> dict:
    arm = cell["arms"][COMPRESSION_ALL_ARM]
    return {
        "selected_programs": arm["selected_programs"],
        "round_diagnostics": arm["round_diagnostics"],
        "prefix_costs": [row["selection_cost"] for row in arm["prefixes"]],
        "prefix_frontiers": _prefix_frontiers(arm["prefixes"]),
    }


def _primitive_frontier(cell: dict) -> tuple:
    return _frontier_diagnostics(cell["primitive"]["validation_summary"])


def _prefix_frontiers(prefixes: Sequence[dict]) -> list[tuple]:
    return [
        _frontier_diagnostics(row["validation_summary"])
        for row in prefixes
    ]


def _validate_menu(menu, menu_size: int) -> None:
    if (
        not isinstance(menu, dict)
        or set(menu)
        != {
            "menu_size",
            "raw_candidate_count",
            "op_count_distribution",
            "support_distribution",
            "frontier_unique_outputs",
            "frontier_candidates_tried_total",
            "frontier_hit_budget",
            "cap",
        }
        or menu.get("menu_size") != menu_size
        or menu.get("cap") != MENU_CAP
    ):
        _fail("menu metadata does not match candidate programs")
    for key in (
        "raw_candidate_count",
        "frontier_unique_outputs",
        "frontier_candidates_tried_total",
    ):
        if type(menu[key]) is not int or menu[key] < 0:
            _fail(f"menu {key} must be a nonnegative integer")
    if menu["raw_candidate_count"] < menu_size:
        _fail("menu raw candidate count cannot be smaller than the menu")
    if menu["frontier_unique_outputs"] < menu["raw_candidate_count"]:
        _fail("menu frontier outputs cannot be fewer than raw candidates")
    if menu["frontier_unique_outputs"] > menu["frontier_candidates_tried_total"]:
        _fail("menu unique outputs cannot exceed frontier candidate work")
    if (
        menu["frontier_candidates_tried_total"] > 30_000
        or type(menu["frontier_hit_budget"]) is not bool
        or (
            menu["frontier_hit_budget"]
            and menu["frontier_candidates_tried_total"] != 30_000
        )
    ):
        _fail("menu frontier diagnostics contradict the registered budget")
    _validate_distribution(
        menu["op_count_distribution"],
        "op-count distribution",
        menu_size,
        minimum=1,
        maximum=4,
    )
    _validate_distribution(
        menu["support_distribution"],
        "support distribution",
        menu_size,
        minimum=2,
        maximum=100,
    )


def _validate_distribution(
    distribution, name: str, total: int, *, minimum: int, maximum: int
) -> None:
    if (
        not isinstance(distribution, dict)
        or not distribution
        or any(
            not isinstance(key, str)
            or re.fullmatch(r"\d+", key) is None
            or not minimum <= int(key) <= maximum
            or type(count) is not int
            or count < 1
            for key, count in distribution.items()
        )
        or sum(distribution.values()) != total
    ):
        _fail(f"menu {name} is inconsistent with menu size")


def _validate_hashes(hashes) -> None:
    if not isinstance(hashes, dict) or set(hashes) != set(HASH_KEYS):
        _fail("shared hashes do not match schema")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in hashes.values()
    ):
        _fail("shared hashes must be lowercase SHA-256 values")


def _validate_world_metadata(metadata, condition: str) -> None:
    if (
        not isinstance(metadata, dict)
        or set(metadata)
        != {"condition", "realized_rho", "density_summary", "expected_motif_length"}
        or metadata.get("condition") != CONDITION_SPECS[condition]
    ):
        _fail("world condition metadata does not match registration")
    rho = metadata.get("realized_rho")
    if (
        not isinstance(rho, dict)
        or set(rho)
        != {"realized_start_test", "realized_start_val", "realized_val_test"}
        or any(not _number(value) or not -1 <= value <= 1 for value in rho.values())
    ):
        _fail("world realized-rho metadata is invalid")
    if not isinstance(metadata.get("density_summary"), dict) or not isinstance(
        metadata.get("expected_motif_length"), dict
    ):
        _fail("world distribution metadata is invalid")


def _validate_input_diagnostics(diagnostics) -> None:
    if not isinstance(diagnostics, dict) or set(diagnostics) != {
        "all_100_starter",
        "validation_assisted",
    }:
        _fail("input-solution diagnostics do not match schema")
    _validate_solution_input(
        diagnostics["all_100_starter"],
        assisted=False,
        source_task_count=100,
        max_solutions=3,
        node_budget=30_000,
    )
    _validate_solution_input(
        diagnostics["validation_assisted"],
        assisted=True,
        source_task_count=VALIDATION_TASK_COUNT,
        max_solutions=1,
        node_budget=90_000,
    )


def _validate_solution_input(
    row,
    *,
    assisted: bool,
    source_task_count: int,
    max_solutions: int,
    node_budget: int,
) -> None:
    base_keys = {
        "candidate_programs_tried_total",
        "canonical_solution_count",
        "solution_program_count_before_canonicalization",
        "solved_task_count",
    }
    expected = base_keys | ({"solve_config"} if assisted else set())
    if not isinstance(row, dict) or set(row) != expected:
        _fail("input-solution diagnostic fields do not match schema")
    if any(type(row[key]) is not int or row[key] < 0 for key in base_keys):
        _fail("input-solution diagnostic values must be nonnegative integers")
    if row["candidate_programs_tried_total"] < source_task_count:
        _fail("input-solution candidate work is below the source task count")
    if row["candidate_programs_tried_total"] > source_task_count * node_budget:
        _fail("input-solution candidate work exceeds the registered search budget")
    if row["solved_task_count"] > row["canonical_solution_count"]:
        _fail("input-solution solved count cannot exceed canonical solutions")
    if (
        row["solved_task_count"] > source_task_count
        or row["solution_program_count_before_canonicalization"]
        > source_task_count * max_solutions
        or row["canonical_solution_count"]
        > row["solution_program_count_before_canonicalization"]
    ):
        _fail("input-solution counts cannot exceed their source task count")
    if assisted and row["solve_config"] != {
        "node_budget": 90_000,
        "max_program_size": 7,
        "max_solutions": 1,
    }:
        _fail("assisted solve config does not match registration")


def _validate_timings(cell: dict) -> None:
    timings = cell.get("timings")
    if (
        not isinstance(timings, dict)
        or set(timings)
        != {"selection_seconds", "real_prefix_seconds", "random_prefix_seconds"}
        or any(not _finite_number(value) or value < 0 for value in timings.values())
        or not _finite_number(cell.get("wall_clock_seconds"))
        or cell["wall_clock_seconds"] < 0
    ):
        _fail("timing metadata is invalid")


def _validate_summary(summary, task_count: int, selected_count: int) -> None:
    if (
        not isinstance(summary, dict)
        or set(summary)
        != {
            "solved_count",
            "failure_count",
            "task_count",
            "solve_rate",
            "mean_search_cost",
            "mean_first_solution_cost",
            "frontier_candidates_tried_total",
            "hit_budget",
            "unique_outputs",
        }
        or summary.get("task_count") != task_count
    ):
        _fail(f"summary task_count must equal {task_count}")
    solved = summary.get("solved_count")
    if type(solved) is not int or not 0 <= solved <= task_count:
        _fail("summary solved_count is invalid")
    if summary.get("failure_count") != task_count - solved:
        _fail("summary failure_count must equal task_count minus solved_count")
    if (
        not _finite_number(summary["solve_rate"])
        or summary["solve_rate"] != solved / task_count
    ):
        _fail("summary solve_rate must equal solved_count / task_count")
    if (
        not _finite_number(summary["mean_search_cost"])
        or summary["mean_search_cost"] < 0
    ):
        _fail("summary mean_search_cost must be finite and nonnegative")
    first_cost = summary["mean_first_solution_cost"]
    if (solved == 0 and first_cost is not None) or (
        solved > 0 and (not _finite_number(first_cost) or first_cost <= 0)
    ):
        _fail(
            "summary mean_first_solution_cost must be positive when tasks are solved"
        )
    frontier_cost = summary["frontier_candidates_tried_total"]
    if type(frontier_cost) is not int or not 0 <= frontier_cost <= 30_000:
        _fail("summary frontier candidate count is invalid")
    if type(summary["hit_budget"]) is not bool:
        _fail("summary hit_budget must be boolean")
    if summary["hit_budget"] and frontier_cost != 30_000:
        _fail("summary hit_budget contradicts frontier candidate count")
    if summary["mean_search_cost"] > frontier_cost:
        _fail("summary mean_search_cost cannot exceed its frontier candidate count")
    if first_cost is not None and first_cost > frontier_cost:
        _fail(
            "summary mean_first_solution_cost cannot exceed its frontier candidate count"
        )
    expected_total_cost = (
        solved * (first_cost or 0) + summary["failure_count"] * frontier_cost
    )
    if (solved == 0 and summary["mean_search_cost"] != frontier_cost) or (
        solved > 0
        and not math.isclose(
            task_count * summary["mean_search_cost"],
            expected_total_cost,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        _fail("summary weighted cost identity is inconsistent")
    if (
        type(summary["unique_outputs"]) is not int
        or not 0 <= summary["unique_outputs"] <= frontier_cost
    ):
        _fail("summary unique_outputs must be a nonnegative integer")
    if summary["unique_outputs"] < max(
        PRIMITIVE_LIBRARY_SIZE + selected_count, solved
    ):
        _fail("summary unique_outputs cannot be fewer than library leaves or solved targets")


def _validate_summary_pair(
    validation_summary, test_summary, *, selected_count: int
) -> None:
    _validate_summary(validation_summary, VALIDATION_TASK_COUNT, selected_count)
    _validate_summary(test_summary, TEST_TASK_COUNT, selected_count)
    if _frontier_diagnostics(validation_summary) != _frontier_diagnostics(
        test_summary
    ):
        _fail("validation and test frontier diagnostics must match for one library")


def _frontier_diagnostics(summary: dict) -> tuple:
    return (
        summary["frontier_candidates_tried_total"],
        summary["hit_budget"],
        summary["unique_outputs"],
    )


def _validate_cost(cost) -> None:
    if not isinstance(cost, dict) or set(cost) != set(COST_KEYS):
        _fail("selection cost fields do not match schema")
    if any(type(value) is not int or value < 0 for value in cost.values()):
        _fail("selection cost components must be nonnegative integer counts")


def _validate_arm_cost(
    name: str,
    cost: dict,
    k: int,
    menu_size: int,
    *,
    input_diagnostics: dict | None,
    assisted_acquisition: int | None,
) -> None:
    if k == 0 or name == RANDOM_ARM:
        if cost != zero_cost():
            _fail(f"{name} K=0 or random selection cost must be zero")
        return
    trials = k * menu_size - k * (k - 1) // 2
    if cost["trial_libraries_evaluated"] != trials:
        _fail(f"{name} trial-library count does not match greedy rounds")
    if name in COMPRESSION_ARMS:
        diagnostic_name = (
            "all_100_starter"
            if name == COMPRESSION_ALL_ARM
            else "validation_assisted"
        )
        solution_count = input_diagnostics[diagnostic_name][
            "canonical_solution_count"
        ]
        acquisition = assisted_acquisition or 0
        expected = {
            "selection_cost_candidate_programs_tried": acquisition,
            "input_solution_search_candidate_programs_tried": acquisition,
            "trial_libraries_evaluated": trials,
            "segmentation_evaluations": trials,
            "solution_segmentations_evaluated": trials * solution_count,
            "frontier_candidates_tried_total": 0,
        }
        if cost != expected:
            if name == COMPRESSION_VALIDATION_ARM and (
                cost["selection_cost_candidate_programs_tried"] != acquisition
                or cost["input_solution_search_candidate_programs_tried"]
                != acquisition
            ):
                _fail("assisted acquisition cost must match its input diagnostic")
            _fail(f"{name} compression cost units are inconsistent")
        return
    if (
        cost["input_solution_search_candidate_programs_tried"] != 0
        or cost["segmentation_evaluations"] != 0
        or cost["solution_segmentations_evaluated"] != 0
    ):
        _fail(f"{name} trial-selector cost units must exclude compression work")
    frontier_cost = cost["frontier_candidates_tried_total"]
    minimum_frontier_work = sum(
        (menu_size - round_number + 1)
        * (PRIMITIVE_LIBRARY_SIZE + round_number)
        for round_number in range(1, k + 1)
    )
    if (
        cost["selection_cost_candidate_programs_tried"] != frontier_cost
        or frontier_cost < minimum_frontier_work
        or frontier_cost > trials * 30_000
    ):
        _fail(f"{name} frontier cost must equal bounded frontier work")


def _validate_trial_objective(objective, name: str) -> None:
    if (
        not isinstance(objective, dict)
        or set(objective) != {"mean_search_cost", "solved_count"}
        or not _finite_number(objective["mean_search_cost"])
        or objective["mean_search_cost"] < 0
        or type(objective["solved_count"]) is not int
    ):
        _fail(f"{name} trial objective is invalid")


def _trial_rank(objective: dict, kind: str) -> tuple:
    if kind == "utility":
        return (-objective["mean_search_cost"], objective["solved_count"])
    return (objective["solved_count"], -objective["mean_search_cost"])


def _objective_from_summary(summary: dict) -> dict:
    return {
        "mean_search_cost": summary["mean_search_cost"],
        "solved_count": summary["solved_count"],
    }


def _direction(value: int | float) -> str:
    return "positive" if value > 0 else "zero" if value == 0 else "negative"


def _number(value) -> bool:
    return type(value) in {int, float}


def _finite_number(value) -> bool:
    return _number(value) and math.isfinite(value)


def _fail(message: str) -> None:
    raise ValueError(f"capacity-curve invariant failed: {message}")
