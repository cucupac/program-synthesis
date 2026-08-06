"""Build the paper figures from the formal selection experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from experiment.plotting import (
    load_budget_intervention_results,
    load_capacity_curve_results,
    load_formal_results,
    load_k_sweep_results,
    make_figures,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "experiment/data/selection/full_selection_experiment.json"
DEFAULT_K_SWEEP_INPUT = (
    ROOT
    / "experiment/data/selection/k_sweep/full_selection_experiment_k_sweep.json"
)
DEFAULT_OUTPUT = ROOT / "experiment/results/figures/full_selection_experiment"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--k-sweep-input", type=Path, default=DEFAULT_K_SWEEP_INPUT)
    parser.add_argument("--capacity-curve-input", type=Path)
    parser.add_argument("--budget-intervention-input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    paths = make_figures(
        load_formal_results(args.input),
        load_k_sweep_results(args.k_sweep_input),
        args.output_dir,
        (
            load_capacity_curve_results(args.capacity_curve_input)
            if args.capacity_curve_input
            else None
        ),
        (
            load_budget_intervention_results(args.budget_intervention_input)
            if args.budget_intervention_input
            else None
        ),
    )
    print(f"wrote {len(paths)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
