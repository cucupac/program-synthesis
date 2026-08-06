import dataclasses
import unittest
from unittest.mock import patch

from experiment.dsl import GRID_SIZE, Program
from experiment.generator import (
    Condition,
    GeneratorConfig,
    make_sweep,
    make_world,
    program_op_count,
    spearman_rho,
    world_to_dict,
)


def tiny_config(**overrides):
    config = GeneratorConfig(
        output_dir="experiment/data/problem_sets/test_generator",
        world_seeds=(6460,),
        n_start=12,
        n_val=6,
        n_test=8,
        motif_count=12,
        motif_min_ops=2,
        motif_max_ops=4,
        rarity_floor=0.03,
        motifs_per_task_pattern=(2, 3),
        glue_ops_per_task_pattern=(0, 1, 2),
        sample_motifs_with_replacement=False,
        min_filled_cells=3,
        max_filled_cells=85,
        max_rejection_attempts=1000,
        conditions=(
            Condition("reversed_a0", "reversed", 0.0, 0.0),
            Condition("reversed_a1", "reversed", 1.0, 1.0),
            Condition("permuted_a05", "permuted", 0.5, 0.5),
            Condition("stale_reversed", "reversed", 0.5, 1.0),
        ),
    )
    return dataclasses.replace(config, **overrides)


def contains_subtree(program, subtree):
    if program == subtree:
        return True
    return any(contains_subtree(arg, subtree) for arg in program.args)


class GeneratorTests(unittest.TestCase):
    def test_make_world_rejects_numeric_string_seed_before_generation(self):
        config = tiny_config()
        with patch("experiment.generator._make_motifs", return_value=()) as make_motifs, patch(
            "experiment.generator._make_p_start", return_value=()
        ), patch("experiment.generator._make_p_alt", return_value=()), patch(
            "experiment.generator._make_tasks", return_value=()
        ), patch("experiment.generator._make_metadata", return_value={}):
            with self.assertRaisesRegex(TypeError, "world_seed must be an exact integer"):
                make_world(config, "6481", config.conditions[0])

        make_motifs.assert_not_called()

    def test_same_config_produces_identical_output(self):
        config = tiny_config()

        first = tuple(world_to_dict(world) for world in make_sweep(config))
        second = tuple(world_to_dict(world) for world in make_sweep(config))

        self.assertEqual(first, second)

    def test_same_seed_shares_motifs_and_p_start_across_conditions(self):
        config = tiny_config()
        first = make_world(config, 6460, config.conditions[0])
        second = make_world(config, 6460, config.conditions[1])

        self.assertEqual([motif.program for motif in first.motifs], [motif.program for motif in second.motifs])
        self.assertEqual(first.p_start, second.p_start)

    def test_same_seed_shares_starter_tasks_across_conditions(self):
        config = tiny_config()
        first = make_world(config, 6460, config.conditions[0])
        second = make_world(config, 6460, config.conditions[1])

        self.assertEqual([task.target for task in first.tasks_start], [task.target for task in second.tasks_start])

    def test_changing_n_val_does_not_change_test_tasks(self):
        config = tiny_config()
        bigger_val_config = tiny_config(n_val=10)
        condition = config.conditions[1]

        first = make_world(config, 6460, condition)
        second = make_world(bigger_val_config, 6460, condition)

        self.assertEqual([task.target for task in first.tasks_test], [task.target for task in second.tasks_test])

    def test_same_seed_shares_test_task_specs_across_conditions(self):
        config = tiny_config()
        first = make_world(config, 6460, config.conditions[0])
        second = make_world(config, 6460, config.conditions[1])

        first_specs = [(len(task.motif_ids), len(task.glue_ops)) for task in first.tasks_test]
        second_specs = [(len(task.motif_ids), len(task.glue_ops)) for task in second.tasks_test]
        self.assertEqual(first_specs, second_specs)
        self.assertTrue(all(len(task.combine_ops) == len(task.motif_ids) - 1 for task in first.tasks_test))

    def test_stale_foresight_has_distinct_val_and_test_distributions(self):
        config = tiny_config()
        stale = config.conditions[-1]
        world = make_world(config, 6460, stale)

        self.assertNotEqual(world.p_val, world.p_test)

    def test_distributions_sum_to_one_and_respect_floor(self):
        config = tiny_config()
        world = make_world(config, 6460, config.conditions[0])

        for distribution in [world.p_start, world.p_alt, world.p_val, world.p_test]:
            self.assertAlmostEqual(sum(distribution), 1.0)
        self.assertTrue(all(p >= config.rarity_floor for p in world.p_start))

    def test_reversed_and_permuted_alt_differ(self):
        config = tiny_config()
        reversed_world = make_world(config, 6460, config.conditions[1])
        permuted_world = make_world(config, 6460, config.conditions[2])

        self.assertNotEqual(reversed_world.p_alt, permuted_world.p_alt)

    def test_alpha_zero_matches_start_distribution(self):
        config = tiny_config()
        world = make_world(config, 6460, config.conditions[0])

        self.assertEqual(world.p_start, world.p_val)
        self.assertEqual(world.p_start, world.p_test)

    def test_validation_and_test_targets_do_not_overlap(self):
        config = tiny_config()
        world = make_world(config, 6460, config.conditions[1])

        val_targets = {task.target for task in world.tasks_val}
        test_targets = {task.target for task in world.tasks_test}
        self.assertFalse(val_targets & test_targets)

    def test_task_targets_do_not_equal_motif_targets(self):
        config = tiny_config()
        world = make_world(config, 6460, config.conditions[1])

        motif_targets = {motif.target for motif in world.motifs}
        for task in world.tasks_start + world.tasks_val + world.tasks_test:
            self.assertNotIn(task.target, motif_targets)

    def test_generated_grids_are_valid(self):
        config = tiny_config()
        world = make_world(config, 6460, config.conditions[1])

        for task in world.tasks_start + world.tasks_val + world.tasks_test:
            self.assertTrue(all(0 <= row < GRID_SIZE for row, _ in task.target))
            self.assertTrue(all(0 <= col < GRID_SIZE for _, col in task.target))

    def test_hidden_motifs_survive_as_subtrees(self):
        config = tiny_config()
        world = make_world(config, 6460, config.conditions[1])
        motifs_by_id = {motif.id: motif.program for motif in world.motifs}

        for task in world.tasks_start[:5]:
            for motif_id in task.motif_ids:
                self.assertTrue(contains_subtree(task.hidden_program, motifs_by_id[motif_id]))

    def test_spearman_rho_handles_ties(self):
        self.assertEqual(spearman_rho([1, 1, 2], [1, 1, 2]), 1.0)
        self.assertAlmostEqual(spearman_rho([1, 1, 2], [1, 2, 2]), 0.5)
        self.assertAlmostEqual(spearman_rho([1, 2, 3], [3, 2, 1]), -1.0)

    def test_metadata_records_generation_diagnostics(self):
        config = tiny_config()
        world = make_world(config, 6460, config.conditions[1])

        self.assertIn("realized_rho", world.metadata)
        self.assertIn("rejection_counts", world.metadata)
        self.assertIn("glue_usage", world.metadata)
        self.assertIn("density_summary", world.metadata)
        self.assertIn("realized_start_test", world.metadata["realized_rho"])

    def test_rejection_exhaustion_raises_with_counts(self):
        config = tiny_config(max_rejection_attempts=1)

        with self.assertRaisesRegex(
            RuntimeError, "too_dense|too_sparse|duplicate|primitive_equal"
        ):
            make_world(config, 6460, config.conditions[1])

    def test_program_op_count_is_derived(self):
        program = Program("add", (Program("line_horizontal"), Program("invert", (Program("square"),))))

        self.assertEqual(program_op_count(program), 2)


if __name__ == "__main__":
    unittest.main()
