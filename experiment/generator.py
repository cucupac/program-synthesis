"""Hidden motifs and generated problem sets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import random

from experiment.dsl import (
    GRID_SIZE,
    Grid,
    PRIMITIVES,
    Program,
    UNARY_OPS,
    call,
    execute,
    primitive,
    program_to_string,
    render_grid,
)

DEFAULT_CONFIG_PATH = "experiment/configs/generator.yaml"

NON_BLANK_PRIMITIVES = tuple(sorted(PRIMITIVES - {"blank"}))
UNARY_GLUE_OPS = ("reflect_horizontal", "reflect_vertical", "reflect_diag", "invert")
GENERATOR_BINARY_OPS = ("add", "subtract", "overlap")
GENERATOR_UNARY_OPS = tuple(sorted(UNARY_OPS))
PRIMITIVE_TARGETS = frozenset(execute(primitive(name)) for name in NON_BLANK_PRIMITIVES)


@dataclass(frozen=True)
class Condition:
    name: str
    alt_kind: str
    alpha_val: float
    alpha_test: float


@dataclass(frozen=True)
class GeneratorConfig:
    output_dir: str
    world_seeds: tuple[int, ...]
    n_start: int
    n_val: int
    n_test: int
    motif_count: int
    motif_min_ops: int
    motif_max_ops: int
    rarity_floor: float
    motifs_per_task_pattern: tuple[int, ...]
    glue_ops_per_task_pattern: tuple[int, ...]
    sample_motifs_with_replacement: bool
    min_filled_cells: int
    max_filled_cells: int
    max_rejection_attempts: int
    conditions: tuple[Condition, ...]


@dataclass(frozen=True)
class Motif:
    id: str
    program: Program
    target: Grid


@dataclass(frozen=True)
class Task:
    id: str
    split: str
    target: Grid
    hidden_program: Program
    motif_ids: tuple[str, ...]
    combine_ops: tuple[str, ...]
    glue_ops: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedWorld:
    config: GeneratorConfig
    condition: Condition
    world_seed: int
    motifs: tuple[Motif, ...]
    p_start: tuple[float, ...]
    p_alt: tuple[float, ...]
    p_val: tuple[float, ...]
    p_test: tuple[float, ...]
    tasks_start: tuple[Task, ...]
    tasks_val: tuple[Task, ...]
    tasks_test: tuple[Task, ...]
    metadata: dict


def load_config(path: str = DEFAULT_CONFIG_PATH) -> GeneratorConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load generator config files.") from exc

    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    conditions = tuple(
        Condition(
            name=str(item["name"]),
            alt_kind=str(item["alt_kind"]),
            alpha_val=float(item["alpha_val"]),
            alpha_test=float(item["alpha_test"]),
        )
        for item in raw["conditions"]
    )

    config = GeneratorConfig(
        output_dir=str(raw["output_dir"]),
        world_seeds=tuple(int(seed) for seed in raw["world_seeds"]),
        n_start=int(raw["sizes"]["n_start"]),
        n_val=int(raw["sizes"]["n_val"]),
        n_test=int(raw["sizes"]["n_test"]),
        motif_count=int(raw["motifs"]["count"]),
        motif_min_ops=int(raw["motifs"]["min_ops"]),
        motif_max_ops=int(raw["motifs"]["max_ops"]),
        rarity_floor=float(raw["motifs"]["rarity_floor"]),
        motifs_per_task_pattern=tuple(
            int(value) for value in raw["tasks"]["motifs_per_task_pattern"]
        ),
        glue_ops_per_task_pattern=tuple(
            int(value) for value in raw["tasks"]["glue_ops_per_task_pattern"]
        ),
        sample_motifs_with_replacement=bool(
            raw["tasks"]["sample_motifs_with_replacement"]
        ),
        min_filled_cells=int(raw["tasks"]["min_filled_cells"]),
        max_filled_cells=int(raw["tasks"]["max_filled_cells"]),
        max_rejection_attempts=int(raw["tasks"]["max_rejection_attempts"]),
        conditions=conditions,
    )
    _validate_config(config)
    return config


def make_sweep(config: GeneratorConfig) -> tuple[GeneratedWorld, ...]:
    return tuple(
        make_world(config, world_seed, condition)
        for world_seed in config.world_seeds
        for condition in config.conditions
    )


def make_world(
    config: GeneratorConfig, world_seed: int, condition: Condition
) -> GeneratedWorld:
    if type(world_seed) is not int:
        raise TypeError("world_seed must be an exact integer")
    motifs = _make_motifs(config, world_seed)
    p_start = _make_p_start(config, world_seed)
    p_alt = _make_p_alt(p_start, world_seed, condition.alt_kind)
    p_val = _mix(p_start, p_alt, condition.alpha_val)
    p_test = _mix(p_start, p_alt, condition.alpha_test)

    seen_targets: set[Grid] = {motif.target for motif in motifs}
    rejections: dict[str, Counter[str]] = {}

    tasks_start = _make_tasks(
        config,
        condition,
        world_seed,
        "start",
        config.n_start,
        motifs,
        p_start,
        seen_targets,
        rejections,
    )
    tasks_test = _make_tasks(
        config,
        condition,
        world_seed,
        "test",
        config.n_test,
        motifs,
        p_test,
        seen_targets,
        rejections,
    )
    tasks_val = _make_tasks(
        config,
        condition,
        world_seed,
        "val",
        config.n_val,
        motifs,
        p_val,
        seen_targets,
        rejections,
    )

    metadata = _make_metadata(
        config,
        motifs,
        p_start,
        p_val,
        p_test,
        tasks_start,
        tasks_val,
        tasks_test,
        rejections,
    )

    return GeneratedWorld(
        config=config,
        condition=condition,
        world_seed=world_seed,
        motifs=motifs,
        p_start=p_start,
        p_alt=p_alt,
        p_val=p_val,
        p_test=p_test,
        tasks_start=tasks_start,
        tasks_val=tasks_val,
        tasks_test=tasks_test,
        metadata=metadata,
    )


def world_to_dict(world: GeneratedWorld) -> dict:
    return {
        "condition": _condition_to_dict(world.condition),
        "world_seed": world.world_seed,
        "motifs": [_motif_to_dict(motif) for motif in world.motifs],
        "distributions": {
            "p_start": list(world.p_start),
            "p_alt": list(world.p_alt),
            "p_val": list(world.p_val),
            "p_test": list(world.p_test),
        },
        "tasks_start": [_task_to_dict(task) for task in world.tasks_start],
        "tasks_val": [_task_to_dict(task) for task in world.tasks_val],
        "tasks_test": [_task_to_dict(task) for task in world.tasks_test],
        "metadata": world.metadata,
    }


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("spearman_rho inputs must have the same length")
    if not xs:
        raise ValueError("spearman_rho inputs cannot be empty")

    x_ranks = _average_ranks(xs)
    y_ranks = _average_ranks(ys)
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_ranks, y_ranks))
    x_den = sum((x - x_mean) ** 2 for x in x_ranks)
    y_den = sum((y - y_mean) ** 2 for y in y_ranks)
    if x_den == 0 or y_den == 0:
        return 1.0 if x_ranks == y_ranks else 0.0
    return numerator / (x_den * y_den) ** 0.5


def program_op_count(program: Program) -> int:
    if not program.args:
        return 0
    return 1 + sum(program_op_count(arg) for arg in program.args)


def _validate_config(config: GeneratorConfig) -> None:
    if config.rarity_floor * config.motif_count >= 1:
        raise ValueError("rarity_floor * motif_count must be less than 1")
    if config.motif_min_ops < 1 or config.motif_min_ops > config.motif_max_ops:
        raise ValueError("invalid motif op range")
    if not config.conditions:
        raise ValueError("at least one condition is required")
    if len({condition.name for condition in config.conditions}) != len(config.conditions):
        raise ValueError("condition names must be unique")
    for condition in config.conditions:
        if condition.alt_kind not in {"reversed", "permuted"}:
            raise ValueError(f"unknown alt_kind: {condition.alt_kind}")
        if not (0 <= condition.alpha_val <= 1 and 0 <= condition.alpha_test <= 1):
            raise ValueError(f"condition {condition.name} has invalid alpha")
    if not config.sample_motifs_with_replacement:
        max_count = max(config.motifs_per_task_pattern)
        if max_count > config.motif_count:
            raise ValueError("cannot sample more motifs than exist without replacement")


def _make_motifs(config: GeneratorConfig, world_seed: int) -> tuple[Motif, ...]:
    rng = random.Random(f"{world_seed}:motifs")
    motifs: list[Motif] = []
    seen_targets = set()

    for index in range(config.motif_count):
        target_ops = _motif_op_count_for_rank(config, index)
        rejection_counts: Counter[str] = Counter()

        for _ in range(config.max_rejection_attempts):
            program = _random_program(rng, target_ops)
            target = execute(program)
            reason = _rejection_reason(config, target, seen_targets, PRIMITIVE_TARGETS)
            if reason is None:
                motif = Motif(f"M{index:02d}", program, target)
                motifs.append(motif)
                seen_targets.add(target)
                break
            rejection_counts[reason] += 1
        else:
            raise RuntimeError(
                f"could not generate motif {index} after "
                f"{config.max_rejection_attempts} attempts: {dict(rejection_counts)}"
            )

    return tuple(motifs)


def _motif_op_count_for_rank(config: GeneratorConfig, index: int) -> int:
    width = config.motif_max_ops - config.motif_min_ops + 1
    mirrored_index = min(index, config.motif_count - 1 - index)
    return config.motif_min_ops + (mirrored_index % width)


def _random_program(rng: random.Random, op_count: int) -> Program:
    if op_count == 0:
        return primitive(rng.choice(NON_BLANK_PRIMITIVES))

    use_unary = op_count == 1 or rng.random() < 0.45
    if use_unary:
        return call(rng.choice(GENERATOR_UNARY_OPS), _random_program(rng, op_count - 1))

    left_ops = rng.randint(0, op_count - 1)
    right_ops = op_count - 1 - left_ops
    return call(
        rng.choice(GENERATOR_BINARY_OPS),
        _random_program(rng, left_ops),
        _random_program(rng, right_ops),
    )


def _make_p_start(config: GeneratorConfig, world_seed: int) -> tuple[float, ...]:
    rng = random.Random(f"{world_seed}:p_start")
    raw = sorted((rng.random() + 0.2 for _ in range(config.motif_count)), reverse=True)
    raw_sum = sum(raw)
    remaining = 1 - config.rarity_floor * config.motif_count
    return tuple(config.rarity_floor + remaining * value / raw_sum for value in raw)


def _make_p_alt(
    p_start: tuple[float, ...], world_seed: int, alt_kind: str
) -> tuple[float, ...]:
    if alt_kind == "reversed":
        return tuple(reversed(p_start))

    rng = random.Random(f"{world_seed}:p_alt:{alt_kind}")
    shuffled = list(p_start)
    while True:
        rng.shuffle(shuffled)
        if tuple(shuffled) != p_start:
            return tuple(shuffled)


def _mix(left: tuple[float, ...], right: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    mixed = tuple((1 - alpha) * a + alpha * b for a, b in zip(left, right))
    total = sum(mixed)
    return tuple(value / total for value in mixed)


def _make_tasks(
    config: GeneratorConfig,
    condition: Condition,
    world_seed: int,
    split: str,
    count: int,
    motifs: tuple[Motif, ...],
    probabilities: tuple[float, ...],
    seen_targets: set[Grid],
    rejections: dict[str, Counter[str]],
) -> tuple[Task, ...]:
    rng = random.Random(_task_seed(world_seed, condition, split))
    tasks: list[Task] = []
    split_rejections: Counter[str] = Counter()

    while len(tasks) < count:
        task_index = len(tasks)
        motif_count = config.motifs_per_task_pattern[
            task_index % len(config.motifs_per_task_pattern)
        ]
        glue_count = config.glue_ops_per_task_pattern[
            task_index % len(config.glue_ops_per_task_pattern)
        ]
        for attempt in range(config.max_rejection_attempts):
            motif_indices = _sample_indices(
                rng, probabilities, motif_count, config.sample_motifs_with_replacement
            )
            glue_ops = tuple(rng.choice(UNARY_GLUE_OPS) for _ in range(glue_count))
            selected_motifs = tuple(motifs[index] for index in motif_indices)
            combine_ops = tuple(
                rng.choice(GENERATOR_BINARY_OPS) for _ in range(max(0, motif_count - 1))
            )
            program = _task_program(selected_motifs, combine_ops, glue_ops)
            target = execute(program)
            base_target = execute(_combine_motifs(selected_motifs, combine_ops))
            reason = _rejection_reason(config, target, seen_targets, PRIMITIVE_TARGETS)
            if reason is None and glue_ops and target == base_target:
                reason = "no_op_glue"

            if reason is not None:
                split_rejections[reason] += 1
                continue

            task = Task(
                id=f"{split}_{len(tasks):03d}",
                split=split,
                target=target,
                hidden_program=program,
                motif_ids=tuple(motifs[index].id for index in motif_indices),
                combine_ops=combine_ops,
                glue_ops=glue_ops,
            )
            tasks.append(task)
            seen_targets.add(target)
            break
        else:
            raise RuntimeError(
                f"could not generate task {len(tasks)} of {count} for {split} after "
                f"{config.max_rejection_attempts} attempts: {dict(split_rejections)}"
            )

    rejections[split] = split_rejections
    return tuple(tasks)


def _task_seed(world_seed: int, condition: Condition, split: str) -> str:
    if split == "start":
        return f"{world_seed}:tasks:start"
    return f"{world_seed}:tasks:{condition.name}:{split}:{_split_alpha(condition, split)}"


def _split_alpha(condition: Condition, split: str) -> float:
    if split == "val":
        return condition.alpha_val
    if split == "test":
        return condition.alpha_test
    return 0.0


def _sample_indices(
    rng: random.Random,
    probabilities: tuple[float, ...],
    count: int,
    with_replacement: bool,
) -> tuple[int, ...]:
    if with_replacement:
        return tuple(rng.choices(range(len(probabilities)), weights=probabilities, k=count))

    available = list(range(len(probabilities)))
    weights = list(probabilities)
    chosen = []
    for _ in range(count):
        pick_position = _weighted_position(rng, weights)
        chosen.append(available.pop(pick_position))
        weights.pop(pick_position)
    return tuple(chosen)


def _weighted_position(rng: random.Random, weights: Sequence[float]) -> int:
    total = sum(weights)
    threshold = rng.random() * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += weight
        if running >= threshold:
            return index
    return len(weights) - 1


def _task_program(
    motifs: tuple[Motif, ...], combine_ops: tuple[str, ...], glue_ops: tuple[str, ...]
) -> Program:
    program = _combine_motifs(motifs, combine_ops)
    for op in glue_ops:
        program = call(op, program)
    return program


def _combine_motifs(motifs: tuple[Motif, ...], combine_ops: tuple[str, ...]) -> Program:
    program = motifs[0].program
    for op, motif in zip(combine_ops, motifs[1:]):
        program = call(op, program, motif.program)
    return program


def _rejection_reason(
    config: GeneratorConfig,
    target: Grid,
    seen_targets: set[Grid],
    primitive_targets: set[Grid],
) -> str | None:
    filled_count = len(target)
    if filled_count < config.min_filled_cells:
        return "too_sparse"
    if filled_count > config.max_filled_cells:
        return "too_dense"
    if target in primitive_targets:
        return "primitive_equal"
    if target in seen_targets:
        return "duplicate"
    return None


def _make_metadata(
    config: GeneratorConfig,
    motifs: tuple[Motif, ...],
    p_start: tuple[float, ...],
    p_val: tuple[float, ...],
    p_test: tuple[float, ...],
    tasks_start: tuple[Task, ...],
    tasks_val: tuple[Task, ...],
    tasks_test: tuple[Task, ...],
    rejections: dict[str, Counter[str]],
) -> dict:
    start_counts = _motif_counts(motifs, tasks_start)
    val_counts = _motif_counts(motifs, tasks_val)
    test_counts = _motif_counts(motifs, tasks_test)

    return {
        "realized_rho": {
            "realized_start_val": spearman_rho(start_counts, val_counts),
            "realized_start_test": spearman_rho(start_counts, test_counts),
            "realized_val_test": spearman_rho(val_counts, test_counts),
        },
        "rejection_counts": {
            split: dict(counts) for split, counts in sorted(rejections.items())
        },
        "glue_usage": {
            "start": _glue_usage(tasks_start),
            "val": _glue_usage(tasks_val),
            "test": _glue_usage(tasks_test),
        },
        "density_summary": {
            "start": _density_summary(tasks_start),
            "val": _density_summary(tasks_val),
            "test": _density_summary(tasks_test),
        },
        "expected_motif_length": {
            "p_start": _expected_motif_length(motifs, p_start),
            "p_val": _expected_motif_length(motifs, p_val),
            "p_test": _expected_motif_length(motifs, p_test),
        },
        "config_summary": {
            "motif_count": config.motif_count,
            "n_start": config.n_start,
            "n_val": config.n_val,
            "n_test": config.n_test,
        },
    }


def _motif_counts(motifs: tuple[Motif, ...], tasks: tuple[Task, ...]) -> tuple[int, ...]:
    counts = Counter(motif_id for task in tasks for motif_id in task.motif_ids)
    return tuple(counts[motif.id] for motif in motifs)


def _glue_usage(tasks: tuple[Task, ...]) -> dict[str, int]:
    counts = Counter(op for task in tasks for op in task.glue_ops)
    return {op: counts[op] for op in UNARY_GLUE_OPS}


def _density_summary(tasks: tuple[Task, ...]) -> dict[str, float]:
    densities = [len(task.target) for task in tasks]
    return {
        "min": min(densities),
        "max": max(densities),
        "mean": sum(densities) / len(densities),
    }


def _expected_motif_length(motifs: tuple[Motif, ...], probabilities: tuple[float, ...]) -> float:
    return sum(program_op_count(motif.program) * p for motif, p in zip(motifs, probabilities))


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    sorted_pairs = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][1] == sorted_pairs[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for pair_index in range(index, end):
            ranks[sorted_pairs[pair_index][0]] = average_rank
        index = end
    return tuple(ranks)


def _condition_to_dict(condition: Condition) -> dict:
    return {
        "name": condition.name,
        "alt_kind": condition.alt_kind,
        "alpha_val": condition.alpha_val,
        "alpha_test": condition.alpha_test,
    }


def _motif_to_dict(motif: Motif) -> dict:
    return {
        "id": motif.id,
        "program": program_to_string(motif.program),
        "target": render_grid(motif.target),
    }


def _task_to_dict(task: Task) -> dict:
    return {
        "id": task.id,
        "split": task.split,
        "target": render_grid(task.target),
        "hidden_program": program_to_string(task.hidden_program),
        "motif_ids": list(task.motif_ids),
        "combine_ops": list(task.combine_ops),
        "glue_ops": list(task.glue_ops),
    }
