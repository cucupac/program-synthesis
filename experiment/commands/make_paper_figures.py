"""Generate the four compact empirical figures used in the paper."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from experiment.paper_figures import make_paper_figures
from experiment.plotting import (
    load_budget_intervention_results,
    load_capacity_curve_results,
    load_formal_results,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY1_INPUT = (
    ROOT / "experiment/data/selection/full_selection_experiment.json"
)
DEFAULT_STUDY2_INPUT = (
    ROOT
    / "experiment/data/selection/capacity_curve/"
    "full_selection_experiment_capacity_curve.json"
)
DEFAULT_STUDY3_INPUT = (
    ROOT
    / "experiment/data/selection/budget_intervention/"
    "full_selection_experiment_budget_intervention.json"
)
DEFAULT_OUTPUT = ROOT / "experiment/results/figures/paper"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study1-input", type=Path, default=DEFAULT_STUDY1_INPUT)
    parser.add_argument("--study2-input", type=Path, default=DEFAULT_STUDY2_INPUT)
    parser.add_argument("--study3-input", type=Path, default=DEFAULT_STUDY3_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    paths = make_paper_figures(
        load_formal_results(args.study1_input),
        load_capacity_curve_results(args.study2_input),
        load_budget_intervention_results(args.study3_input),
        args.output_dir,
    )
    print(f"wrote {len(paths)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
