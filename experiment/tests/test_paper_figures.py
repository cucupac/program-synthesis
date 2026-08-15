import tempfile
import unittest
from pathlib import Path

from matplotlib import pyplot as plt

from experiment.commands.make_paper_figures import main as make_paper_figures_main
from experiment.paper_figures import (
    build_study1_cost_summary,
    build_study1_secondary_summary,
    build_study1_summary,
    make_paper_figures,
    plot_study1_cost,
    plot_study1_secondary,
    plot_study1_selection,
    plot_study2_capacity,
    plot_study3_budget,
    plot_studies2_and3,
)
from experiment.plotting import (
    budget_intervention_summaries,
    capacity_curve_summaries,
    capacity_diagnostic_summaries,
    load_budget_intervention_results,
    load_capacity_curve_results,
    load_formal_results,
)


FORMAL_RESULTS = Path("experiment/data/selection/full_selection_experiment.json")
CAPACITY_RESULTS = Path(
    "experiment/data/selection/capacity_curve/"
    "full_selection_experiment_capacity_curve.json"
)
BUDGET_RESULTS = Path(
    "experiment/data/selection/budget_intervention/"
    "full_selection_experiment_budget_intervention.json"
)
EXPECTED_FILES = {
    "study_1_selection.pdf",
    "study_1_selection.png",
    "study_1_cost.pdf",
    "study_1_cost.png",
    "study_1_secondary.pdf",
    "study_1_secondary.png",
    "study_2_3_capacity_budget.pdf",
    "study_2_3_capacity_budget.png",
}
FORBIDDEN_LABELS = (
    "past compression",
    "future utility",
    "subchains",
    "validation utility",
    "compression on validation solutions",
    "compression on starter solutions",
)


def figure_text(figure) -> str:
    parts = []
    for axis in figure.axes:
        legend = axis.get_legend()
        parts.extend(
            [
                axis.get_title(loc="left"),
                axis.get_title(),
                axis.get_title(loc="right"),
                axis.get_xlabel(),
                axis.get_ylabel(),
                *(text.get_text() for text in axis.texts),
                *((text.get_text() for text in legend.get_texts()) if legend else ()),
                *(label.get_text() for label in axis.get_xticklabels()),
                *(label.get_text() for label in axis.get_yticklabels()),
            ]
        )
    parts.extend(text.get_text() for text in figure.texts)
    return " ".join(" ".join(part.split()) for part in parts).lower()


class PaperFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formal = load_formal_results(FORMAL_RESULTS)
        cls.capacity = load_capacity_curve_results(CAPACITY_RESULTS)
        cls.budget = load_budget_intervention_results(BUDGET_RESULTS)
        cls.study1 = build_study1_summary(cls.formal)
        cls.study1_cost = build_study1_cost_summary(cls.formal)
        cls.study1_secondary = build_study1_secondary_summary(cls.formal)
        cls.study2 = capacity_curve_summaries(cls.capacity)
        cls.study2_diagnostics = capacity_diagnostic_summaries(cls.capacity)
        cls.study3 = budget_intervention_summaries(cls.budget)

    def assert_paper_figure_contract(self, figure, *, axes=2, legend_axes=()):
        self.assertEqual(len(figure.axes), axes)
        for index, axis in enumerate(figure.axes):
            if index in legend_axes:
                self.assertIsNotNone(axis.get_legend())
            else:
                self.assertIsNone(axis.get_legend())
            self.assertNotRegex(axis.get_title(loc="left"), r"^[A-Z]\s{2}")
        text = figure_text(figure)
        for forbidden in FORBIDDEN_LABELS:
            self.assertNotIn(forbidden, text)

    def test_study1_summary_matches_paper_values(self):
        absolute = {
            row["key"]: round(row["mean"], 2)
            for row in self.study1["absolute"]
        }
        self.assertEqual(
            absolute,
            {
                "primitives": 58.91,
                "matched_validation_compression": 62.00,
                "standard_compression": 64.53,
                "validation_utility": 66.08,
            },
        )

        contrasts = {
            row["key"]: (
                round(row["estimate"], 2),
                tuple(round(value, 2) for value in row["interval"]),
            )
            for row in self.study1["contrasts"]
        }
        self.assertEqual(
            contrasts,
            {
                "utility_minus_matched": (4.08, (2.46, 5.71)),
                "validation_minus_starter_compression": (-0.26, (-1.46, 0.95)),
                "utility_minus_standard": (1.55, (-0.27, 3.37)),
            },
        )

    def test_study1_figure_has_two_directly_labeled_panels(self):
        figure = plot_study1_selection(self.study1)
        self.addCleanup(plt.close, figure)
        self.assert_paper_figure_contract(figure)
        figure.canvas.draw()
        performance, contrasts = figure.axes
        text = figure_text(figure)
        self.assertIn("registered primary", text)
        self.assertIn("secondary", text)
        self.assertIn("validation search utility", text)
        self.assertIn("validation-solution compression", text)
        self.assertEqual(
            [label.get_text() for label in performance.get_xticklabels()],
            [
                "Primitives\nonly",
                "Validation-\nsolution\ncompression",
                "Standard\ncompression",
                "Validation\nSearch Utility",
            ],
        )
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            ["Held-Out Performance", "Method Contrasts"],
        )
        self.assertFalse(
            any(
                "compression" in label.get_text().lower()
                for label in performance.get_yticklabels()
            )
        )
        self.assertFalse(any(label.get_text() for label in contrasts.get_yticklabels()))
        self.assertGreater(contrasts.get_position().width, 0.35)

    def test_study1_secondary_figure_matches_existing_analyses(self):
        self.assertEqual(
            [
                (row["n"], round(row["estimate"], 2))
                for row in self.study1_secondary["similarity"]
            ],
            [(63, 2.67), (48, 1.54), (69, 0.54)],
        )
        self.assertEqual(
            [round(row["estimate"], 2) for row in self.study1_secondary["compression"]],
            [25.78, 12.32, 9.79],
        )

        figure = plot_study1_secondary(self.study1_secondary)
        self.addCleanup(plt.close, figure)
        self.assert_paper_figure_contract(figure)
        text = figure_text(figure)
        self.assertIn("advantage by problem-set similarity", text)
        self.assertIn("starter-solution compression", text)
        self.assertIn("different", text)
        self.assertIn("validation search utility", text)
        self.assertIn("operations removed", text)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            [
                "Validation Search Utility Advantage\nby Problem-Set Similarity",
                "Starter-Solution Compression",
            ],
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        self.assertTrue(
            all(
                axis._left_title.get_window_extent(renderer).x1 <= figure.bbox.x1
                for axis in figure.axes
            )
        )
        self.assertEqual(
            figure.axes[0].get_xlabel(),
            "Difference in test problems solved",
        )

    def test_study1_cost_figure_shows_selection_cost_and_search_savings(self):
        self.assertEqual(
            [round(row["candidate_programs"]) for row in self.study1_cost["upfront"]],
            [13_650_000, 995_770],
        )
        self.assertEqual(
            [
                (
                    row["median_future_problems"],
                    row["finite"],
                    row["no_payback"],
                )
                for row in self.study1_cost["payback"]
            ],
            [(8_398, 143, 37), (9_818.5, 116, 64)],
        )

        figure = plot_study1_cost(self.study1_cost)
        self.addCleanup(plt.close, figure)
        self.assert_paper_figure_contract(figure)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            ["Selection Cost", "Problems Until Savings Equal Cost"],
        )
        text = figure_text(figure)
        for label in (
            "13.65m",
            "1.00m",
            "8,398",
            "9,819",
            "37/180 savings stayed below cost",
            "64/180 savings stayed below cost",
        ):
            self.assertIn(label, text)
        self.assertNotIn("share of", text)
        self.assertNotIn("break even", text)
        self.assertNotIn("payback", text)

    def test_study2_data_and_figure_cover_the_capacity_curve(self):
        self.assertEqual(len(self.study2["curves"]["past_compression_gain"]), 21)
        self.assertEqual(len(self.study2["curves"]["utility_gain"]), 21)
        self.assertEqual(len(self.study2_diagnostics), 21)
        self.assertGreater(self.study2["primitive_mean"], 0)
        self.assertEqual(self.study2_diagnostics[0]["compression_removed_pct"], 0)

        figure = plot_study2_capacity(self.study2, self.study2_diagnostics)
        self.addCleanup(plt.close, figure)
        self.assert_paper_figure_contract(figure, legend_axes={0})
        text = figure_text(figure)
        self.assertNotIn("observed peak", text)
        self.assertIn("below primitives", text)
        self.assertNotIn("one-to-two drop", text)
        self.assertIn("operations removed", text)
        self.assertEqual(
            [axis.get_xlabel() for axis in figure.axes],
            ["Number of added abstractions", "Number of added abstractions"],
        )
        self.assertGreaterEqual(len(figure.axes[0].collections), 2)
        performance, compression = figure.axes
        primitive_label = next(
            text for text in performance.texts if text.get_text() == "Primitives only"
        )
        capacity_note = next(
            text for text in performance.texts if text.get_text().startswith("At 20")
        )
        self.assertGreater(primitive_label.get_position()[0], 20)
        self.assertGreater(capacity_note.get_position()[0], 20)
        self.assertEqual(capacity_note.get_horizontalalignment(), "right")
        self.assertEqual(performance.get_legend()._loc, 3)
        self.assertEqual(
            [text.get_text() for text in performance.get_legend().get_texts()],
            ["Standard compression", "Validation Search Utility"],
        )
        self.assertFalse(
            any(
                text.get_text()
                in {"Standard compression", "Validation Search Utility"}
                for text in performance.texts
            )
        )
        self.assertFalse(compression.texts)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            ["Held-Out Performance", "Starter-Solution Compression"],
        )

    def test_study3_data_and_figure_cover_the_budget_intervention(self):
        self.assertEqual(self.study3["budgets"], [30_000, 45_000, 60_000, 90_000])
        self.assertEqual(len(self.study3["size4_access_pct"]), 4)
        for key in ("past_compression", "future_utility"):
            row = self.study3["methods"][key]
            self.assertEqual(len(row["difference"]), 4)
            self.assertEqual(len(row["recovered_pct"]), 4)
            self.assertIn("estimate", row["interaction"])
            self.assertEqual(len(row["interaction"]["interval"]), 2)

        figure = plot_study3_budget(self.study3)
        self.addCleanup(plt.close, figure)
        self.assert_paper_figure_contract(figure, legend_axes={0, 1})
        text = figure_text(figure)
        self.assertIn("size-four access", text)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            [
                "Effect of the Second Abstraction",
                "Search Access and Recovery",
            ],
        )
        effects, mechanism = figure.axes
        self.assertEqual(
            [text.get_text() for text in effects.get_legend().get_texts()],
            ["Standard compression", "Validation Search Utility"],
        )
        self.assertEqual(
            [text.get_text() for text in mechanism.get_legend().get_texts()],
            [
                "Size-four access",
                "Recovered losses: Standard compression",
                "Recovered losses: Validation Search Utility",
            ],
        )
        self.assertFalse(effects.texts)
        self.assertFalse(mechanism.texts)

    def test_studies2_and3_share_one_four_graph_figure(self):
        figure = plot_studies2_and3(
            self.study2,
            self.study2_diagnostics,
            self.study3,
        )
        self.addCleanup(plt.close, figure)
        self.assert_paper_figure_contract(
            figure,
            axes=4,
            legend_axes={0, 2, 3},
        )
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            [
                "Held-Out Performance",
                "Starter-Solution Compression",
                "Effect of the Second Abstraction",
                "Search Access and Recovery",
            ],
        )

    def test_make_paper_figures_writes_exactly_eight_nonempty_files(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = make_paper_figures(
                self.formal,
                self.capacity,
                self.budget,
                Path(temp),
            )
            self.assertEqual({path.name for path in paths}, EXPECTED_FILES)
            self.assertEqual({path.name for path in Path(temp).iterdir()}, EXPECTED_FILES)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_command_writes_exactly_eight_nonempty_files(self):
        with tempfile.TemporaryDirectory() as temp:
            make_paper_figures_main(
                [
                    "--study1-input",
                    str(FORMAL_RESULTS),
                    "--study2-input",
                    str(CAPACITY_RESULTS),
                    "--study3-input",
                    str(BUDGET_RESULTS),
                    "--output-dir",
                    temp,
                ]
            )
            files = list(Path(temp).iterdir())
            self.assertEqual({path.name for path in files}, EXPECTED_FILES)
            self.assertTrue(all(path.stat().st_size > 0 for path in files))


if __name__ == "__main__":
    unittest.main()
