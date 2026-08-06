import json
import tempfile
import unittest
from pathlib import Path

import experiment.plotting as plotting
from experiment import capacity_curve as capacity
from experiment.commands.make_figures import main as make_figures_main
from experiment.plotting import (
    capacity_diagnostic_summaries,
    capacity_curve_summaries,
    compression_tradeoff_summaries,
    figure_summaries,
    k_sweep_summaries,
    load_capacity_curve_results,
    load_formal_results,
    load_k_sweep_results,
    make_figures,
    plot_capacity_curve,
    plot_capacity_diagnostics,
    plot_compression_tradeoff,
    plot_k_sensitivity,
    plot_similarity,
)
from experiment.tests.test_capacity_curve import synthetic_cell
from matplotlib import pyplot as plt


FORMAL_RESULTS = Path("experiment/data/selection/full_selection_experiment.json")
K_SWEEP_RESULTS = Path(
    "experiment/data/selection/k_sweep/full_selection_experiment_k_sweep.json"
)
CAPACITY_RESULTS = Path(
    "experiment/data/selection/capacity_curve/"
    "full_selection_experiment_capacity_curve.json"
)
BUDGET_INTERVENTION_RESULTS = Path(
    "experiment/data/selection/budget_intervention/"
    "full_selection_experiment_budget_intervention.json"
)


class PlottingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = load_formal_results(FORMAL_RESULTS)
        cls.k_sweep_results = load_k_sweep_results(K_SWEEP_RESULTS)
        cls.summaries = figure_summaries(cls.results)
        cls.compression_tradeoff = compression_tradeoff_summaries(cls.results)
        cls.k_sweep_summaries = k_sweep_summaries(cls.k_sweep_results)

    def test_loads_complete_registered_k_sweep(self):
        self.assertEqual(len(self.k_sweep_results["cells"]), 540)

    def test_rejects_malformed_k_sweep_results(self):
        source = json.loads(K_SWEEP_RESULTS.read_text(encoding="utf-8"))
        cases = {
            "wrong experiment": lambda data: data.update(experiment_name="wrong"),
            "smoke result": lambda data: data.update(smoke=True),
            "missing cell": lambda data: data["cells"].pop(),
            "wrong K values": lambda data: data["registration"].update(k_values=[2, 10]),
            "wrong random metadata": lambda data: data["cells"][0].update(
                random_draws=1
            ),
            "hidden capacity metadata": lambda data: data["cells"][0]["arms"][
                "hidden_motif_oracle"
            ].update(diagnostic_candidate_count=12),
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.json"
            for label, corrupt in cases.items():
                with self.subTest(label):
                    data = json.loads(json.dumps(source))
                    corrupt(data)
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_k_sweep_results(path)

    def test_k_sweep_summaries_match_the_registered_results(self):
        actual = []
        for row in self.k_sweep_summaries:
            actual.append(
                (
                    row["k"],
                    *(round(row[name], 2) for name in (
                        "compression_gain",
                        "utility_gain",
                        "scoring_effect",
                        "data_effect",
                    )),
                    *(tuple(round(value, 2) for value in row[name]) for name in (
                        "compression_gain_ci",
                        "utility_gain_ci",
                        "scoring_effect_ci",
                        "data_effect_ci",
                    )),
                )
            )
        self.assertEqual(
            actual,
            [
                (2, -6.61, -4.14, 2.08, 1.74, (-7.94, -5.33), (-5.14, -3.03), (0.97, 3.27), (0.03, 3.34)),
                (5, 0.41, 2.01, 3.48, 1.46, (-1.08, 1.82), (0.33, 3.83), (2.03, 4.99), (-0.28, 3.11)),
                (10, 7.00, 8.57, 4.54, 0.98, (5.04, 8.84), (6.59, 10.70), (3.03, 6.34), (-0.64, 2.64)),
            ],
        )

    def test_summaries_match_the_approved_claims(self):
        self.assertEqual(
            [round(row["advantage"], 1) for row in self.summaries["similarity"]],
            [2.7, 1.5, 0.5],
        )
        self.assertEqual(
            [round(row["primitive"], 1) for row in self.summaries["similarity"]],
            [59.5, 57.1, 59.6],
        )
        self.assertEqual(
            [round(row["compression_gain"], 1) for row in self.summaries["similarity"]],
            [3.6, 7.1, 6.5],
        )
        self.assertEqual(
            [round(row["utility_gain"], 1) for row in self.summaries["similarity"]],
            [6.2, 8.6, 7.0],
        )
        self.assertAlmostEqual(self.summaries["mechanism"]["data_effect"], -0.2556, places=3)
        self.assertAlmostEqual(self.summaries["mechanism"]["scoring_effect"], 4.0833, places=3)
        self.assertEqual(
            [round(row["advantage"], 1) for row in self.summaries["staleness"]],
            [2.6, 0.2],
        )
        self.assertEqual(
            self.summaries["cost"]["practical"],
            {"finite": 116, "never": 64, "finite_median": 9818.5},
        )
        self.assertEqual(
            self.summaries["cost"]["registered"],
            {"finite": 143, "never": 37, "finite_median": 8398},
        )

    def test_compression_tradeoff_matches_the_post_hoc_analysis(self):
        starter, validation = self.compression_tradeoff["rows"]
        self.assertEqual(
            (
                round(starter["compression"]["mean"], 2),
                round(starter["utility"]["mean"], 2),
                round(starter["difference"]["mean"], 2),
                starter["compression_wins"],
            ),
            (25.78, 12.32, -13.46, 30),
        )
        self.assertEqual(
            (
                round(validation["compression"]["mean"], 2),
                round(validation["utility"]["mean"], 2),
                round(validation["difference"]["mean"], 2),
                validation["compression_wins"],
            ),
            (21.71, 10.57, -11.15, 30),
        )
        self.assertEqual(
            [round(value, 2) for value in starter["difference"]["ci"]],
            [-16.25, -11.03],
        )
        self.assertEqual(
            [round(value, 2) for value in validation["difference"]["ci"]],
            [-12.39, -9.82],
        )
        self.assertEqual(
            round(
                self.compression_tradeoff["validation_utility_on_starter"][
                    "mean"
                ],
                2,
            ),
            9.79,
        )

    def test_compression_tradeoff_figure_is_directly_labeled(self):
        figure = plot_compression_tradeoff(self.compression_tradeoff)
        self.addCleanup(plt.close, figure)

        self.assertEqual(len(figure.axes), 1)
        axis = figure.axes[0]
        self.assertEqual(
            [text.get_text() for text in axis.get_legend().get_texts()],
            [
                "Greedy compression selection",
                "Search-utility selection",
            ],
        )
        self.assertEqual(
            figure._suptitle.get_text(),
            "Utility-selected abstractions also shortened starter solutions,\n"
            "but less than greedy compression.",
        )
        self.assertEqual(
            [label.get_text() for label in axis.get_yticklabels()],
            [
                "Greedy compression\nusing solved starter programs",
                "Search utility\nusing the same 25 starter problems",
                "Search utility\nusing 25 validation problems",
            ],
        )
        labels = {text.get_text() for text in axis.texts}
        self.assertTrue(
            {
                "25.8%",
                "12.3%",
                "9.8%",
            }
            <= labels
        )
        self.assertEqual(
            axis.get_xlabel(),
            "Starter-solution operations removed (%)",
        )

    def test_similarity_figure_has_approved_two_panel_layout(self):
        figure = plot_similarity(self.summaries)
        self.addCleanup(plt.close, figure)

        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(
            figure._suptitle.get_text(),
            "At K=10, both abstraction methods beat primitives; utility had the highest mean solve rate.",
        )
        self.assertIn(
            "registered analysis uses continuous realized ρ",
            " ".join(text.get_text() for text in figure.texts),
        )
        self.assertTrue(
            {"Primitives\nonly", "Past\ncompression", "Future\nutility"}
            <= {text.get_text() for text in figure.axes[0].texts}
        )
        self.assertEqual(
            [
                label.get_text()
                for label in figure.axes[1].get_yticklabels()
                if label.get_visible()
            ],
            [row["label"] for row in self.summaries["similarity"]],
        )

    def test_k_sensitivity_figure_has_approved_two_panel_layout(self):
        figure = plot_k_sensitivity(self.k_sweep_summaries)
        self.addCleanup(plt.close, figure)

        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(
            figure._suptitle.get_text(),
            "Library size determined whether abstractions helped, while utility scoring remained advantageous.",
        )
        for axis in figure.axes:
            self.assertIsNone(axis.get_legend())
            self.assertEqual(
                [label.get_text() for label in axis.get_xticklabels()],
                ["2", "5", "10"],
            )
            self.assertTrue(
                any(set(line.get_ydata()) == {0} for line in axis.get_lines())
            )
        self.assertEqual(
            {"Past compression", "Future utility"},
            {text.get_text() for text in figure.axes[0].texts if text.get_text() in {"Past compression", "Future utility"}},
        )
        self.assertEqual(
            {
                "Utility - matched compression",
                "Compression: validation - starter",
            },
            {
                text.get_text()
                for text in figure.axes[1].texts
                if text.get_text()
                in {
                    "Utility - matched compression",
                    "Compression: validation - starter",
                }
            },
        )

    def test_writes_six_pdf_png_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = make_figures(self.results, self.k_sweep_results, Path(temp))
            self.assertEqual(len(paths), 12)
            self.assertEqual({path.suffix for path in paths}, {".pdf", ".png"})
            self.assertTrue(all(path.stat().st_size > 1_000 for path in paths))
            self.assertTrue(
                (Path(temp) / "figure_10_compression_tradeoff.pdf").is_file()
            )

    def test_figure_command_accepts_registered_k_sweep_input(self):
        with tempfile.TemporaryDirectory() as temp:
            make_figures_main(
                [
                    "--input",
                    str(FORMAL_RESULTS),
                    "--k-sweep-input",
                    str(K_SWEEP_RESULTS),
                    "--output-dir",
                    temp,
                ]
            )
            self.assertTrue((Path(temp) / "figure_5_k_sensitivity.pdf").is_file())

    def test_loads_and_plots_complete_capacity_curve_without_legends(self):
        payload = _synthetic_capacity_results()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "capacity.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_capacity_curve_results(path)
            summary = capacity_curve_summaries(loaded)
            figure = plot_capacity_curve(summary)
            self.addCleanup(plt.close, figure)

        self.assertEqual(len(figure.axes), 2)
        self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))
        labels = {text.get_text() for axis in figure.axes for text in axis.texts}
        self.assertTrue(
            {
                "Primitives  0.00",
                "Past compression",
                "Future utility",
                "Same validation problems:\nutility - compression",
                "Practical comparison:\nutility - past compression",
            }
            <= labels
        )
        self.assertEqual(
            figure._suptitle.get_text(),
            "Test performance rose through about 11 abstractions, then fell sharply.",
        )
        self.assertTrue(
            any(
                len(set(line.get_ydata())) == 1
                and next(iter(set(line.get_ydata()))) == summary["primitive_mean"]
                for line in figure.axes[0].get_lines()
            )
        )
        self.assertTrue(
            any(set(line.get_ydata()) == {0} for line in figure.axes[1].get_lines())
        )

    def test_capacity_diagnostics_have_expected_values_and_labels(self):
        summary = capacity_diagnostic_summaries(_synthetic_capacity_results())
        self.assertEqual(len(summary), 21)
        self.assertEqual(summary[0]["past_unique_outputs"], 100)
        self.assertEqual(summary[2]["utility_unique_outputs"], 100)
        self.assertEqual(summary[0]["compression_removed_pct"], 0)
        self.assertEqual(summary[2]["compression_removed_pct"], 10)
        self.assertEqual(summary[11]["compression_removed_pct"], 55)
        self.assertEqual(summary[20]["compression_removed_pct"], 100)

        figure = plot_capacity_diagnostics(summary)
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(figure.axes), 2)
        self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))
        labels = {text.get_text() for axis in figure.axes for text in axis.texts}
        self.assertTrue(
            {
                "Past compression",
                "Future utility",
                "K=2 drop",
                "Later decline",
                "K=0",
                "K=2",
                "K=11",
                "K=20",
            }
            <= labels
        )

    def test_capacity_input_adds_only_figures_six_and_seven(self):
        payload = _synthetic_capacity_results()
        with tempfile.TemporaryDirectory() as temp:
            capacity_path = Path(temp) / "capacity.json"
            capacity_path.write_text(json.dumps(payload), encoding="utf-8")
            output = Path(temp) / "figures"

            make_figures_main(
                [
                    "--input",
                    str(FORMAL_RESULTS),
                    "--k-sweep-input",
                    str(K_SWEEP_RESULTS),
                    "--capacity-curve-input",
                    str(capacity_path),
                    "--output-dir",
                    str(output),
                ]
            )

            self.assertEqual(len(list(output.glob("*.pdf"))), 8)
            self.assertEqual(len(list(output.glob("*.png"))), 8)
            self.assertTrue((output / "figure_6_capacity_curve.pdf").is_file())
            self.assertTrue((output / "figure_7_capacity_diagnostics.pdf").is_file())


class BudgetInterventionPlottingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = (
            "load_budget_intervention_results",
            "budget_intervention_summaries",
            "plot_budget_intervention",
        )
        cls.available = all(hasattr(plotting, name) for name in required)
        if cls.available:
            cls.results = plotting.load_budget_intervention_results(
                BUDGET_INTERVENTION_RESULTS
            )
            cls.summary = plotting.budget_intervention_summaries(cls.results)

    def test_loads_complete_registered_budget_intervention(self):
        self.assertTrue(self.available, "budget-intervention plotting API is missing")
        self.assertEqual(len(self.results["cells"]), 180)

        with tempfile.TemporaryDirectory() as temp:
            malformed = Path(temp) / "malformed.json"
            malformed.write_text('{"experiment_name": "wrong"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                plotting.load_budget_intervention_results(malformed)

            changed_registration = json.loads(json.dumps(self.results))
            changed_registration["registration"]["budgets"] = [30_000]
            registration_path = Path(temp) / "changed-registration.json"
            registration_path.write_text(
                json.dumps(changed_registration), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                plotting.load_budget_intervention_results(registration_path)

            changed_source = json.loads(json.dumps(self.results))
            changed_source["cells"][0]["source_cell_hash"] = "0" * 64
            source_path = Path(temp) / "changed-source.json"
            source_path.write_text(json.dumps(changed_source), encoding="utf-8")
            with self.assertRaises(ValueError):
                plotting.load_budget_intervention_results(source_path)

    def test_budget_intervention_summaries_match_registered_results(self):
        self.assertTrue(self.available, "budget-intervention plotting API is missing")
        self.assertEqual(self.summary["budgets"], [30_000, 45_000, 60_000, 90_000])

        expected = {
            "past_compression": {
                "difference": [-8.08, 0.22, 3.43, 3.33],
                "interaction": 11.42,
                "interval": [10.17, 12.66],
                "lost_count": 1_904,
                "recovered_pct": [0.0, 90.1, 100.0, 100.0],
            },
            "future_utility": {
                "difference": [-6.14, -0.54, 3.23, 3.47],
                "interaction": 9.61,
                "interval": [8.52, 10.69],
                "lost_count": 1_681,
                "recovered_pct": [0.0, 85.1, 99.9, 100.0],
            },
        }
        for name, values in expected.items():
            actual = self.summary["methods"][name]
            self.assertEqual(
                [round(value, 2) for value in actual["difference"]],
                values["difference"],
            )
            self.assertEqual(round(actual["interaction"]["estimate"], 2), values["interaction"])
            self.assertEqual(
                [round(value, 2) for value in actual["interaction"]["interval"]],
                values["interval"],
            )
            self.assertEqual(actual["lost_count"], values["lost_count"])
            self.assertEqual(
                [round(value, 1) for value in actual["recovered_pct"]],
                values["recovered_pct"],
            )
            self.assertEqual(actual["positive_seed_interactions"], 30)

        self.assertEqual(
            [round(value, 1) for value in self.summary["size4_access_pct"]],
            [8.1, 99.2, 100.0, 100.0],
        )

    def test_budget_intervention_figure_has_clear_two_panel_layout(self):
        self.assertTrue(self.available, "budget-intervention plotting API is missing")
        figure = plotting.plot_budget_intervention(self.summary)
        self.addCleanup(plt.close, figure)

        self.assertEqual(len(figure.axes), 2)
        self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))
        self.assertEqual(
            figure._suptitle.get_text(),
            "More search budget reversed the K=2 loss and restored the missing solutions.",
        )
        self.assertTrue(
            any(set(line.get_ydata()) == {0} for line in figure.axes[0].get_lines())
        )
        labels = {text.get_text() for axis in figure.axes for text in axis.texts}
        self.assertTrue(
            {
                "Past compression",
                "Future utility",
                "K=2 searches reaching size four",
            }
            <= labels
        )

    def test_sustained_catch_up_handles_exact_event_boundaries(self):
        self.assertTrue(hasattr(plotting, "_observed_sustained_catch_up"))
        catch_up = plotting._observed_sustained_catch_up

        self.assertEqual(catch_up([20_000], [20_000]), (30_000, False))
        self.assertEqual(
            catch_up([20_000, 35_000], [31_000, 35_000]),
            (31_000, False),
        )
        self.assertEqual(
            catch_up([20_000, 40_000], [35_000, 50_000]),
            (50_000, True),
        )
        self.assertEqual(
            catch_up([20_000, 80_000], [40_000]),
            (None, True),
        )

    def test_budget_followup_summaries_use_all_cells_and_one_max_t_family(self):
        catch_up = self.summary["catch_up"]
        expected_catch_up = {
            "past_compression": {
                "median": 47_732,
                "by_45k": 71,
                "by_60k": 142,
                "censored": 21,
                "reversals": 61,
            },
            "future_utility": {
                "median": 47_707,
                "by_45k": 72,
                "by_60k": 134,
                "censored": 19,
                "reversals": 81,
            },
        }
        for name, expected in expected_catch_up.items():
            actual = catch_up[name]
            self.assertEqual(actual["median_budget"], expected["median"])
            self.assertEqual(actual["caught_by"][45_000], expected["by_45k"])
            self.assertEqual(actual["caught_by"][60_000], expected["by_60k"])
            self.assertEqual(actual["right_censored"], expected["censored"])
            self.assertEqual(actual["reversals"], expected["reversals"])
            self.assertEqual(len(actual["thresholds"]), 180)

        expected_effects = {
            "matched_compression": {
                1: [
                    (0.47, -0.45, 1.38),
                    (1.11, 0.14, 2.09),
                    (0.86, -0.03, 1.75),
                    (0.95, -0.01, 1.91),
                ],
                2: [
                    (2.63, 1.11, 4.15),
                    (1.18, -0.07, 2.44),
                    (1.71, 0.70, 2.71),
                    (1.91, 1.01, 2.82),
                ],
            },
            "past_compression": {
                1: [
                    (0.07, -1.06, 1.20),
                    (0.96, -0.32, 2.24),
                    (0.66, -0.50, 1.82),
                    (0.86, -0.07, 1.78),
                ],
                2: [
                    (2.02, 0.38, 3.65),
                    (0.21, -1.18, 1.59),
                    (0.46, -1.04, 1.95),
                    (0.99, -0.26, 2.24),
                ],
            },
        }
        for comparison, by_k in expected_effects.items():
            for k, expected_rows in by_k.items():
                actual_rows = self.summary["selector_effects"][comparison][k]
                actual = [
                    (
                        round(row["estimate"], 2),
                        round(row["interval"][0], 2),
                        round(row["interval"][1], 2),
                    )
                    for row in actual_rows
                ]
                self.assertEqual(actual, expected_rows)

    def test_budget_followup_figure_has_three_directly_labeled_panels(self):
        self.assertTrue(hasattr(plotting, "plot_budget_followup"))
        figure = plotting.plot_budget_followup(self.summary)
        self.addCleanup(plt.close, figure)

        self.assertEqual(len(figure.axes), 3)
        self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))
        self.assertEqual(
            figure._suptitle.get_text(),
            "Most K=2 libraries caught up by 60,000 candidates; utility's advantage depended on K and budget.",
        )
        for axis in figure.axes[1:]:
            self.assertTrue(
                any(set(line.get_ydata()) == {0} for line in axis.get_lines())
            )
        labels = {text.get_text() for axis in figure.axes for text in axis.texts}
        self.assertTrue({"Past compression", "Future utility", "K=1", "K=2"} <= labels)

    def test_budget_input_adds_figures_eight_and_nine(self):
        self.assertTrue(self.available, "budget-intervention plotting API is missing")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "figures"
            make_figures_main(
                [
                    "--input",
                    str(FORMAL_RESULTS),
                    "--k-sweep-input",
                    str(K_SWEEP_RESULTS),
                    "--capacity-curve-input",
                    str(CAPACITY_RESULTS),
                    "--budget-intervention-input",
                    str(BUDGET_INTERVENTION_RESULTS),
                    "--output-dir",
                    str(output),
                ]
            )

            self.assertEqual(len(list(output.glob("*.pdf"))), 10)
            self.assertEqual(len(list(output.glob("*.png"))), 10)
            self.assertTrue((output / "figure_8_budget_intervention.pdf").is_file())
            self.assertTrue((output / "figure_9_budget_followup.pdf").is_file())


def _synthetic_capacity_results():
    cells = [
        synthetic_cell(seed, condition)
        for seed in capacity.FORMAL_SEEDS
        for condition in capacity.CONDITIONS
    ]
    return {
        "experiment_name": capacity.EXPERIMENT_NAME,
        "smoke": False,
        "registration": capacity.registration(smoke=False),
        "cells": cells,
    }


if __name__ == "__main__":
    unittest.main()
