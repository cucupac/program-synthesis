# Abstraction Selection in Bounded Program Synthesis

This repository contains the paper, code, and generated data for the experiments in
*On the Value of Abstractions: Abstraction Selection in Bounded Program
Synthesis*.

[Read the paper (PDF)](paper/abstraction_selection_in_bounded_program_synthesis.pdf).

## Abstract

Reusable abstractions can shorten programs. However, some do not improve
bounded search, so we compare two selection procedures on a synthetic grid
benchmark. Compression selects the abstractions that shorten known solutions
most. Validation Search Utility instead selects the abstractions that most
decrease capped search cost on separate validation targets. Both procedures use
the same starter-derived candidate set, and we evaluate their libraries on
separate test problems. Adding a library item can shorten a solution, but it
also creates more small programs. Because the solver examines programs by size,
these programs can consume the budget before the solver reaches a useful larger
program. In the validation comparison, both procedures used the same 25
problems. Compression acquired solutions for a mean of 16.59 problems, whereas
Utility scored all 25 target outcomes. With ten abstractions, both procedures
improved on primitive-only search. Utility solved 4.08 more test problems per
100 than validation-solution compression (95 percent CI 2.46 to 5.71). This
complete-procedure difference includes the scoring objective and solution
availability. Against standard compression, which used solutions acquired from
100 starter problems, Utility's smaller advantage remained uncertain. The
similarity means followed the hypothesis, although formal comparisons did not
establish the effect. Library size also mattered. At 30,000 attempts, observed
mean performance was highest at 11 abstractions and fell below primitives alone
at 20. Performance also decreased sharply from one abstraction to two. For
these fixed libraries, a larger budget reversed the decrease. In this benchmark,
abstraction value depended on selection method, library size, and search budget.

## Contents

- `paper/`: current manuscript, compiled PDF, and LaTeX dependencies
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
