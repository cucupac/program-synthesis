# Abstraction Selection in Bounded Program Synthesis

This repository contains the paper, code, and generated data for the experiments in
*On the Value of Abstractions: Abstraction Selection in Bounded Program
Synthesis*.

[Read the paper (PDF)](paper/on-the-value-of-abstractions.pdf).

## Abstract

Library learning allows a computational solver to use its experience solving problems to add abstractions to its library that aid in future problem solving. How it chooses those abstractions, however, affects its performance on unseen test problems. Additionally, the value of each abstraction a solver adds to its library is contextually dependent on factors like: its search process, search budgets, and library size. In this paper, we explore two different approaches for selecting abstractions and test the performance of the resulting libraries they construct. Namely, we compare **compression**–which chooses abstractions that compress the size of a set of previously-found past solution programs by minimizing the number of subprograms across the entire set–against **search cost minimization**–which chooses abstractions that minimize solution search cost on a new distinct problem set. Both selection procedures choose abstractions from the same set of subprogram candidates, created by initially finding solution programs using only primitive substates and operations that are permitted by a starter domain specific language (DSL). We used Pattern Builder for this study, which presents a solver with a pattern on a 10x10 pixel grid and tasks it to build a program using its library that constructs that pattern. Asking how a solver should optimally add subprograms to its library is worthwhile not only because it can improve held-out problem-solving performance but also because the procedures by which abstractions are chosen differ in computational cost: any increased performance should justify the computational cost required to achieve it. This paper also explores how library size affects solver performance under both abstraction selection procedures under a fixed search budget–and how a search budget interacts with library size and test performance. Each subprogram added to a library increases the number of possible shallow-depth programs a solver must search through, which can exhaust its search budget. While abstractions themselves reduce search by not requiring solvers to reconstruct their subprograms, it's possible that this search reduction is only used deeper into a program trajectory. Such cases may exhaust a search budget before a solution program is found, highlighting a tradeoff between library size and search cost.

In this study, we found that with 10 abstractions, both selection procedures outperformed the primitive-only DSL. When both abstraction selection methods used the same set of problems, search cost minimization solved 4.08 more test problems out of 100 than compression. At 30,000 program attempts, mean performance peaked at 11 abstractions but then fell sharply, ultimately underperforming the primitive-only DSL at 20 added abstractions. Interestingly, both selection procedures' library performance dropped sharply from library size 1 to 2 abstractions. We determined that this performance drop was caused by the 30,000 program search budget: after holding the libraries constant and increasing allowed search, all of the unsolved problems that constituted the performance reduction were recovered. These findings strongly support the view the value of a library abstraction depends on the context in which it's used and the constraints a solver is subject to.

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
