# Abstraction Selection in Bounded Program Synthesis

This repository contains the code and generated data for the experiments in
*On the Value of Abstractions: Abstraction Selection in Bounded Program
Synthesis*.

The experiments compare two methods for selecting reusable program fragments.
Compression selects fragments that shorten known solutions. Validation search
utility selects fragments that reduce bounded-search cost on validation
problems.

## Contents

- `experiment/`: implementation and command-line entry points
- `experiment/tests/`: automated tests
- `experiment/data/`: generated problems, registrations, and result data
- `experiment/docs/methodology.md`: experiment design and run record
- `REPRODUCE.md`: environment and reproduction steps

The repository includes cell-level results for all formal experiments. Two
large combined files are excluded because they duplicate those results. The
reproduction command rebuilds them and checks their SHA-256 hashes.

## Quick Start

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r experiment/requirements.txt
python3 -m experiment.commands.rebuild_selection_aggregates
python3 -m unittest discover -s experiment/tests
python3 -m experiment.commands.make_paper_figures
```

See [`REPRODUCE.md`](REPRODUCE.md) for details.
