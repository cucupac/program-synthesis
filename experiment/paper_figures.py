"""Compact, publication-sized figures for the paper's three studies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from math import ceil, sqrt
from pathlib import Path
from statistics import mean, median, stdev

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from experiment.plotting import (
    budget_intervention_summaries,
    capacity_curve_summaries,
    capacity_diagnostic_summaries,
    compression_tradeoff_summaries,
    figure_summaries,
)


BLUE = "#4477AA"
CHARCOAL = "#333333"
LIGHT_GRAY = "#C9C9C9"
MID_GRAY = "#777777"
GRID_GRAY = "#E8E8E8"
FIGURE_SIZE = (6.5, 2.8)
SECONDARY_FIGURE_SIZE = (6.5, 2.4)
COST_FIGURE_SIZE = (6.5, 2.0)
CAPACITY_BUDGET_FIGURE_SIZE = (6.5, 4.8)
T_CRITICAL_975_DF29 = 2.045229642132703


def _solved(cell: dict, arm: str) -> float:
    return float(cell["arms"][arm]["summary"]["solved_count"])


def _seed_t_interval(
    cells: Sequence[dict], contrast: Callable[[dict], float]
) -> tuple[float, tuple[float, float]]:
    """Return a mean and Student-t interval over 30 seed-level means."""
    by_seed: dict[int, list[float]] = defaultdict(list)
    for cell in cells:
        by_seed[cell["seed"]].append(float(contrast(cell)))
    seed_means = [mean(by_seed[seed]) for seed in sorted(by_seed)]
    if len(seed_means) != 30:
        raise ValueError("Study I paper intervals require 30 independent seeds")
    estimate = mean(seed_means)
    half_width = T_CRITICAL_975_DF29 * stdev(seed_means) / sqrt(len(seed_means))
    return estimate, (estimate - half_width, estimate + half_width)


def build_study1_summary(results: dict) -> dict:
    """Build the Study I paper summary with seed-level Student-t intervals."""
    cells = [
        cell for cell in results["cells"] if cell["condition"] != "stale_reversed"
    ]
    if len(cells) != 180:
        raise ValueError("expected the 180 primary Study I condition cells")

    absolute_specs = (
        ("primitives", "Primitives only", "primitives_only", "primitive"),
        (
            "matched_validation_compression",
            "Compression on $P_{\\mathrm{SCM}}$",
            "compression_on_validation_assisted",
            "matched",
        ),
        (
            "standard_compression",
            "Compression on $P_{\\mathrm{prim}}$",
            "compression_on_all_100_starter",
            "standard",
        ),
        (
            "validation_utility",
            "Search cost minimization",
            "utility_on_validation",
            "utility",
        ),
    )
    absolute = [
        {
            "key": key,
            "label": label,
            "mean": mean(_solved(cell, arm) for cell in cells),
            "style": style,
        }
        for key, label, arm, style in absolute_specs
    ]

    contrast_specs = (
        (
            "utility_minus_matched",
            "Search cost minimization -\ncompression on $P_{\\mathrm{SCM}}$",
            "Registered primary",
            lambda cell: _solved(cell, "utility_on_validation")
            - _solved(cell, "compression_on_validation_assisted"),
            "utility",
        ),
        (
            "validation_minus_starter_compression",
            "Compression on $P_{\\mathrm{SCM}}$ -\ncompression on $P_{\\mathrm{prim,25}}$",
            "Registered primary",
            lambda cell: _solved(cell, "compression_on_validation_assisted")
            - _solved(cell, "compression_on_matched_25_starter"),
            "matched",
        ),
        (
            "utility_minus_standard",
            "Search cost minimization -\ncompression on $P_{\\mathrm{prim}}$",
            "Secondary",
            lambda cell: _solved(cell, "utility_on_validation")
            - _solved(cell, "compression_on_all_100_starter"),
            "utility",
        ),
    )
    contrasts = []
    for key, label, evidence, contrast, style in contrast_specs:
        estimate, interval = _seed_t_interval(cells, contrast)
        contrasts.append(
            {
                "key": key,
                "label": label,
                "evidence": evidence,
                "estimate": estimate,
                "interval": interval,
                "style": style,
            }
        )
    return {"absolute": absolute, "contrasts": contrasts}


def build_study1_secondary_summary(results: dict) -> dict:
    """Build the similarity and compression-transfer summaries for Study I."""
    similarity = [
        {
            "label": row["label"],
            "n": row["n"],
            "estimate": row["advantage"],
            "interval": row["advantage_ci"],
        }
        for row in figure_summaries(results)["similarity"]
    ]
    tradeoff = compression_tradeoff_summaries(results)
    starter = tradeoff["rows"][0]
    compression = [
        {
            "label": "Compression on $P_{\\mathrm{prim,25}}$\non 25 primitive-only problems",
            "estimate": starter["compression"]["mean"],
            "interval": starter["compression"]["ci"],
            "style": "standard",
        },
        {
            "label": "Search cost minimization\non same 25\nprimitive-only problems",
            "estimate": starter["utility"]["mean"],
            "interval": starter["utility"]["ci"],
            "style": "utility",
        },
        {
            "label": "Search cost minimization\non 25 $T_{\\mathrm{SCM}}$ problems",
            "estimate": tradeoff["validation_utility_on_starter"]["mean"],
            "interval": tradeoff["validation_utility_on_starter"]["ci"],
            "style": "utility",
        },
    ]
    return {"similarity": similarity, "compression": compression}


def build_study1_cost_summary(results: dict) -> dict:
    """Build the Study I selection-work and payback summary."""
    cells = [
        cell for cell in results["cells"] if cell["condition"] != "stale_reversed"
    ]
    if len(cells) != 180:
        raise ValueError("expected the 180 primary Study I condition cells")

    utility_costs = [
        cell["arms"]["utility_on_validation"]["selection_cost"][
            "selection_cost_candidate_programs_tried"
        ]
        for cell in cells
    ]
    compression_costs = [
        cell["arms"]["compression_on_validation_assisted"]["selection_cost"][
            "selection_cost_candidate_programs_tried"
        ]
        for cell in cells
    ]
    cost = figure_summaries(results)["cost"]
    return {
        "upfront": [
            {
                "label": "Search cost\nminimization",
                "candidate_programs": median(utility_costs),
                "style": "utility",
            },
            {
                "label": "Compression\non $P_{\\mathrm{SCM}}$",
                "candidate_programs": median(compression_costs),
                "style": "standard",
            },
        ],
        "payback": [
            {
                "label": "vs. compression\non $P_{\\mathrm{SCM}}$",
                "median_future_problems": cost["registered"]["finite_median"],
                "finite": cost["registered"]["finite"],
                "no_payback": cost["registered"]["never"],
            },
            {
                "label": "vs. compression\non $P_{\\mathrm{prim}}$",
                "median_future_problems": cost["practical"]["finite_median"],
                "finite": cost["practical"]["finite"],
                "no_payback": cost["practical"]["never"],
            },
        ],
        "cells": len(cells),
    }


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.edgecolor": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "figure.constrained_layout.w_pad": 0.12,
            "figure.constrained_layout.h_pad": 0.08,
            "figure.constrained_layout.wspace": 0.04,
            "figure.constrained_layout.hspace": 0.04,
        }
    )


def _clean_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID_GRAY, linewidth=0.65)
    axis.set_axisbelow(True)
    axis.tick_params(length=2.5, width=0.6)


def _graph_title(axis, title: str, *, fontsize: float = 9) -> None:
    axis.set_title(title, loc="left", fontweight="bold", fontsize=fontsize, pad=6)


def _style_for(name: str) -> dict:
    if name == "utility":
        return {"color": BLUE, "marker": "o", "facecolor": BLUE}
    if name == "matched":
        return {"color": CHARCOAL, "marker": "s", "facecolor": "white"}
    if name == "standard":
        return {"color": CHARCOAL, "marker": "s", "facecolor": CHARCOAL}
    return {"color": LIGHT_GRAY, "marker": "D", "facecolor": LIGHT_GRAY}


def plot_study1_selection(summary: dict) -> Figure:
    """Plot absolute Study I outcomes and the three paper contrasts."""
    _paper_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.9, 1.1]},
    )

    performance = axes[0]
    rows = summary["absolute"]
    positions = list(range(len(rows)))
    category_labels = (
        "Primitives\nonly",
        "Compression\non $P_{\\mathrm{SCM}}$",
        "Compression\non $P_{\\mathrm{prim}}$",
        "Search cost\nminimization",
    )
    for x, row in zip(positions, rows):
        style = _style_for(row["style"])
        performance.scatter(
            x,
            row["mean"],
            s=42,
            marker=style["marker"],
            edgecolor=style["color"],
            facecolor=style["facecolor"],
            linewidth=1.1,
            zorder=3,
        )
        performance.text(
            x,
            row["mean"] + 0.28,
            f"{row['mean']:.2f}",
            ha="center",
            va="bottom",
            color=style["color"] if row["style"] != "primitive" else MID_GRAY,
            fontsize=7,
        )
    performance.set_xticks(positions, category_labels)
    performance.set_xlim(-0.3, 3.3)
    performance.set_ylim(57.5, 67.3)
    performance.set_ylabel("Test problems solved (of 100)")
    _graph_title(performance, "Held-Out Performance")
    _clean_axis(performance)
    performance.tick_params(axis="x", length=0, pad=5, labelsize=6.0)

    contrasts = axes[1]
    contrasts.axvline(0, color=LIGHT_GRAY, linewidth=1.0, zorder=1)
    contrast_rows = summary["contrasts"]
    contrast_positions = [2.35, 1.35, 0.35]
    contrast_labels = {
        "utility_minus_matched": (
            "Search cost minimization - compression on $P_{\\mathrm{SCM}}$"
        ),
        "validation_minus_starter_compression": (
            "Compression on $P_{\\mathrm{SCM}}$"
            " - compression on $P_{\\mathrm{prim,25}}$"
        ),
        "utility_minus_standard": (
            "Search cost minimization - compression on $P_{\\mathrm{prim}}$"
        ),
    }
    label_transform = contrasts.get_yaxis_transform()
    for y, row in zip(contrast_positions, contrast_rows):
        style = _style_for(row["style"])
        low, high = row["interval"]
        contrasts.plot([low, high], [y, y], color=style["color"], linewidth=1.5)
        contrasts.plot(
            [low, low, high, high],
            [y - 0.08, y + 0.08, y + 0.08, y - 0.08],
            color=style["color"],
            linewidth=0.8,
        )
        contrasts.scatter(
            row["estimate"],
            y,
            s=30,
            marker=style["marker"],
            edgecolor=style["color"],
            facecolor=style["facecolor"],
            linewidth=1.0,
            zorder=3,
        )
        contrasts.text(
            high + 0.25,
            y,
            f"{row['estimate']:+.2f}  [{low:.2f}, {high:.2f}]",
            va="center",
            fontsize=6.2,
            color=style["color"],
        )
        contrasts.text(
            0.01,
            y + 0.3,
            contrast_labels[row["key"]],
            transform=label_transform,
            va="bottom",
            fontsize=6.5,
            color=CHARCOAL,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )
        contrasts.text(
            0.01,
            y + 0.14,
            row["evidence"],
            transform=label_transform,
            va="bottom",
            fontsize=5.6,
            color=MID_GRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
        )
    contrasts.axhline(0.85, color=GRID_GRAY, linewidth=0.65)
    contrasts.axhline(1.85, color=GRID_GRAY, linewidth=0.65)
    contrasts.set_yticks([])
    contrasts.set_ylim(0.0, 3.0)
    contrasts.set_xlim(-3.0, 8.4)
    contrasts.set_xticks([-2, 0, 2, 4, 6])
    contrasts.set_xlabel("Difference in problems solved")
    _graph_title(contrasts, "Method Contrasts")
    _clean_axis(contrasts)
    contrasts.grid(False)
    contrasts.spines["left"].set_visible(False)
    return figure


def plot_study1_secondary(summary: dict) -> Figure:
    """Plot Study I's similarity and compression-transfer analyses."""
    _paper_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=SECONDARY_FIGURE_SIZE,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.85, 1.15]},
    )

    similarity = axes[0]
    similarity.axvline(0, color=LIGHT_GRAY, linewidth=1.0)
    similarity_rows = summary["similarity"]
    positions = [2, 1, 0]
    for y, row in zip(positions, similarity_rows):
        low, high = row["interval"]
        similarity.errorbar(
            row["estimate"],
            y,
            xerr=[[row["estimate"] - low], [high - row["estimate"]]],
            color=BLUE,
            marker="o",
            markersize=4,
            linewidth=1.4,
            capsize=3,
        )
        similarity.text(
            high + 0.15,
            y,
            f"{row['estimate']:+.1f}",
            ha="left",
            va="center",
            color=BLUE,
            fontsize=6.5,
        )
    similarity.set_yticks(
        positions,
        [
            f"Different\n$\\rho < 0$  ($n={similarity_rows[0]['n']}$)",
            f"Moderate\n$0 \\leq \\rho < 0.5$  ($n={similarity_rows[1]['n']}$)",
            f"Similar\n$\\rho \\geq 0.5$  ($n={similarity_rows[2]['n']}$)",
        ],
    )
    similarity.set_xlim(-2.5, 5.2)
    similarity.set_xticks([-2, 0, 2, 4])
    similarity.set_xlabel("Difference in test problems solved")
    _graph_title(
        similarity,
        "Search Cost Minimization Advantage\nby Problem-Set Similarity",
        fontsize=7.0,
    )
    _clean_axis(similarity)
    similarity.grid(False)
    similarity.grid(axis="x", color=GRID_GRAY, linewidth=0.65)

    compression = axes[1]
    compression_rows = summary["compression"]
    for y, row in zip(positions, compression_rows):
        style = _style_for(row["style"])
        low, high = row["interval"]
        compression.errorbar(
            row["estimate"],
            y,
            xerr=[[row["estimate"] - low], [high - row["estimate"]]],
            color=style["color"],
            marker=style["marker"],
            markerfacecolor=style["facecolor"],
            markersize=4,
            linewidth=1.4,
            capsize=3,
        )
        compression.text(
            high + 0.4,
            y,
            f"{row['estimate']:.1f}%",
            ha="left",
            va="center",
            color=style["color"],
            fontsize=6.5,
        )
    compression.set_yticks(
        positions,
        [row["label"] for row in compression_rows],
    )
    compression.set_xlim(0, 33)
    compression.set_xticks([0, 10, 20, 30])
    compression.set_xlabel("Operations removed (%)")
    _graph_title(compression, "Primitive-Only Solution Compression", fontsize=7.0)
    _clean_axis(compression)
    compression.grid(False)
    compression.grid(axis="x", color=GRID_GRAY, linewidth=0.65)
    return figure


def plot_study1_cost(summary: dict) -> Figure:
    """Plot selection cost and the future search savings needed to recover it."""
    _paper_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=COST_FIGURE_SIZE,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.9, 1.1]},
    )
    positions = [1, 0]

    upfront = axes[0]
    upfront_rows = summary["upfront"]
    upfront_values = [row["candidate_programs"] / 1_000_000 for row in upfront_rows]
    upfront.barh(
        positions,
        upfront_values,
        color=[_style_for(row["style"])["color"] for row in upfront_rows],
        height=0.44,
    )
    for y, value in zip(positions, upfront_values):
        if value >= 3:
            x = value - 0.3
            horizontal_alignment = "right"
            color = "white"
        else:
            x = value + 0.25
            horizontal_alignment = "left"
            color = CHARCOAL
        upfront.text(
            x,
            y,
            f"{value:.2f}M",
            ha=horizontal_alignment,
            va="center",
            color=color,
            fontsize=7,
            fontweight="bold",
        )
    upfront.set_yticks(positions, [row["label"] for row in upfront_rows])
    upfront.set_xlim(0, 15)
    upfront.set_xticks([0, 5, 10, 15])
    upfront.set_xlabel("Program attempts used for selection (millions)")
    upfront.set_title(
        "Selection Cost", loc="left", fontweight="bold", fontsize=8.5, pad=6
    )
    _clean_axis(upfront)
    upfront.grid(False)
    upfront.grid(axis="x", color=GRID_GRAY, linewidth=0.65)

    payback = axes[1]
    payback_rows = summary["payback"]
    payback_values = [row["median_future_problems"] for row in payback_rows]
    payback.barh(positions, payback_values, color=BLUE, height=0.44)
    no_payback_x = 10_600
    for y, row, value in zip(positions, payback_rows, payback_values):
        payback.text(
            value - 250,
            y,
            f"{ceil(value):,}",
            ha="right",
            va="center",
            color="white",
            fontsize=7,
            fontweight="bold",
        )
        payback.text(
            no_payback_x,
            y,
            f"{row['no_payback']}/{summary['cells']} savings stayed\nbelow cost",
            ha="left",
            va="center",
            color=MID_GRAY,
            fontsize=6.5,
        )
    payback.set_yticks(positions, [row["label"] for row in payback_rows])
    payback.set_xlim(0, 15_000)
    payback.set_xticks([0, 5_000, 10_000, 15_000], ["0", "5k", "10k", "15k"])
    payback.set_xlabel("Median future problems")
    payback.set_title(
        "Problems Until Savings Equal Cost",
        loc="left",
        fontweight="bold",
        fontsize=8.5,
        pad=6,
    )
    _clean_axis(payback)
    payback.grid(False)
    payback.grid(axis="x", color=GRID_GRAY, linewidth=0.65)
    return figure


def plot_study2_capacity(
    capacity: dict, diagnostics: list[dict], axes=None
) -> Figure:
    """Plot the held-out capacity curve and compression trajectory."""
    _paper_style()
    if axes is None:
        figure, axes = plt.subplots(
            1, 2, figsize=FIGURE_SIZE, constrained_layout=True
        )
    else:
        figure = axes[0].figure
    ks = list(range(21))
    primitive = capacity["primitive_mean"]
    label_box = {
        "boxstyle": "round,pad=0.15",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.92,
    }

    performance = axes[0]
    performance.axhline(primitive, color=LIGHT_GRAY, linewidth=1.2)
    performance.text(
        23.8,
        primitive + 0.35,
        "Primitives only",
        ha="right",
        va="bottom",
        color=MID_GRAY,
        fontsize=6.5,
    )
    for key, label, color, marker in (
        (
            "past_compression_gain",
            "Compression on $P_{\\mathrm{prim}}$",
            CHARCOAL,
            "s",
        ),
        ("utility_gain", "Search cost minimization", BLUE, "o"),
    ):
        rows = capacity["curves"][key]
        estimates = [primitive + rows[k]["estimate"] for k in ks]
        lows = [primitive + rows[k]["interval"][0] for k in ks]
        highs = [primitive + rows[k]["interval"][1] for k in ks]
        performance.fill_between(ks, lows, highs, color=color, alpha=0.12, linewidth=0)
        performance.plot(
            ks,
            estimates,
            label=label,
            color=color,
            marker=marker,
            markevery=[0, 2, 5, 8, 11, 14, 17, 20],
            markersize=3.5,
            linewidth=1.6,
        )
    performance.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        fontsize=6.5,
        handlelength=2.2,
        borderpad=0.25,
        labelspacing=0.3,
    )
    performance.text(
        23.8,
        69.2,
        "At 20, both are\nbelow primitives",
        ha="right",
        va="center",
        fontsize=6.5,
        color=MID_GRAY,
        bbox=label_box,
    )
    performance.set_xlim(0, 24.5)
    performance.set_xticks([0, 5, 10, 15, 20])
    performance.set_xlabel("Number of added abstractions")
    performance.set_ylabel("Test problems solved (of 100)")
    _graph_title(performance, "Held-Out Performance")
    _clean_axis(performance)

    compression = axes[1]
    diagnostic_ks = [row["k"] for row in diagnostics]
    removed = [row["compression_removed_pct"] for row in diagnostics]
    compression.plot(
        diagnostic_ks,
        removed,
        color=CHARCOAL,
        marker="s",
        markevery=[0, 2, 5, 8, 11, 14, 17, 20],
        markersize=3.5,
        linewidth=1.6,
    )
    compression.set_xlim(0, 20.5)
    compression.set_xticks([0, 5, 10, 15, 20])
    compression.set_ylim(bottom=0)
    compression.set_xlabel("Number of added abstractions")
    compression.set_ylabel("Primitive-only solution\noperations removed (%)")
    _graph_title(compression, "Primitive-Only Solution Compression")
    _clean_axis(compression)
    return figure


def plot_study3_budget(summary: dict, axes=None) -> Figure:
    """Plot the fixed-library budget intervention and mechanism checks."""
    _paper_style()
    if axes is None:
        figure, axes = plt.subplots(
            1, 2, figsize=FIGURE_SIZE, constrained_layout=True
        )
    else:
        figure = axes[0].figure
    budgets = summary["budgets"]
    legend_style = {
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "none",
        "framealpha": 0.92,
        "fontsize": 6.5,
        "handlelength": 2.2,
        "borderpad": 0.25,
        "labelspacing": 0.3,
    }

    effects = axes[0]
    effects.axhline(0, color=LIGHT_GRAY, linewidth=1.0)
    for key, label, color, marker in (
        (
            "past_compression",
            "Compression on $P_{\\mathrm{prim}}$",
            CHARCOAL,
            "s",
        ),
        ("future_utility", "Search cost minimization", BLUE, "o"),
    ):
        row = summary["methods"][key]
        effects.plot(
            budgets,
            row["difference"],
            label=label,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.6,
        )
    effects.legend(loc="lower right", **legend_style)
    effects.set_xlim(28_000, 95_000)
    effects.set_ylim(-10, 5)
    effects.set_xticks(budgets, ["30k", "45k", "60k", "90k"])
    effects.set_xlabel("Candidate-program budget")
    effects.set_ylabel("K=2 minus K=1 test problems solved")
    _graph_title(effects, "Effect of the Second Abstraction")
    _clean_axis(effects)

    mechanism = axes[1]
    mechanism.plot(
        budgets,
        summary["size4_access_pct"],
        color=MID_GRAY,
        linestyle=":",
        marker="D",
        markersize=3.5,
        linewidth=1.4,
        label="Size-four access",
    )
    for key, label, color, marker in (
        (
            "past_compression",
            "compression on $P_{\\mathrm{prim}}$",
            CHARCOAL,
            "s",
        ),
        ("future_utility", "search cost minimization", BLUE, "o"),
    ):
        recovered = summary["methods"][key]["recovered_pct"]
        mechanism.plot(
            budgets,
            recovered,
            label=f"Recovered losses: {label}",
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.6,
        )
    mechanism.legend(loc="lower right", **legend_style)
    mechanism.set_xlim(28_000, 95_000)
    mechanism.set_ylim(0, 105)
    mechanism.set_xticks(budgets, ["30k", "45k", "60k", "90k"])
    mechanism.set_xlabel("Candidate-program budget")
    mechanism.set_ylabel("Searches or lost problems (%)")
    _graph_title(mechanism, "Search Access and Recovery")
    _clean_axis(mechanism)
    return figure


def plot_studies2_and3(
    capacity: dict,
    diagnostics: list[dict],
    budget: dict,
) -> Figure:
    """Plot the Study II capacity results and the Study III budget test."""
    _paper_style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=CAPACITY_BUDGET_FIGURE_SIZE,
        constrained_layout=True,
    )
    plot_study2_capacity(capacity, diagnostics, axes=axes[0])
    plot_study3_budget(budget, axes=axes[1])
    return figure


def _save_figure(figure: Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    figure.savefig(paths[0])
    figure.savefig(paths[1], dpi=300)
    plt.close(figure)
    return paths


def make_paper_figures(
    formal: dict,
    capacity: dict,
    budget: dict,
    output_dir: Path,
) -> list[Path]:
    """Generate the four paper figures as vector PDFs and PNG previews."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = (
        (
            "study_1_selection",
            plot_study1_selection(build_study1_summary(formal)),
        ),
        (
            "study_1_cost",
            plot_study1_cost(build_study1_cost_summary(formal)),
        ),
        (
            "study_1_secondary",
            plot_study1_secondary(build_study1_secondary_summary(formal)),
        ),
        (
            "study_2_3_capacity_budget",
            plot_studies2_and3(
                capacity_curve_summaries(capacity),
                capacity_diagnostic_summaries(capacity),
                budget_intervention_summaries(budget),
            ),
        ),
    )
    paths = []
    for stem, figure in figures:
        paths.extend(_save_figure(figure, output_dir, stem))
    return paths
