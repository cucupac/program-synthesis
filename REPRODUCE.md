# Reproduce the Results

Run all commands from the repository root.

## Environment

The recorded environment used Python 3.14.0. Create an isolated environment
and install the two external packages.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r experiment/requirements.txt
```

## Rebuild the Combined Data

Experiments II and III store one JSON file for each seed and condition. This
command checks those files and rebuilds their combined analysis files.

```sh
python3 -m experiment.commands.rebuild_selection_aggregates
```

The command stops if a rebuilt file does not have its recorded SHA-256 hash.

## Test the Implementation

```sh
python3 -m unittest discover -s experiment/tests
```

## Rebuild the Figures

```sh
python3 -m experiment.commands.make_paper_figures
```

This command writes PDF and PNG files to
`experiment/results/figures/paper/`.

## Formal Experiment Commands

The formal data are included because a complete run needs substantial search
work. These commands show the available options.

```sh
python3 -m experiment.commands.run_full_selection_experiment --help
python3 -m experiment.commands.run_k_sweep_experiment --help
python3 -m experiment.commands.run_capacity_curve_experiment --help
python3 -m experiment.commands.run_budget_intervention --help
```

The registrations and cell-level results are in
`experiment/data/selection/`. The design record in
`experiment/docs/methodology.md` gives the seeds, conditions, budgets, and
changes made before each formal run.
