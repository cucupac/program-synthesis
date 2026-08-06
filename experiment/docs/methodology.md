# Experiment Design: Compression vs. Utility for Library Selection

> **How to read this document.** Part I is the current design: everything in it is
> live truth, and any agent or collaborator should implement from Part I alone
> without consulting history. Sections that changed since the original February
> 2026 design carry an *Amended* marker pointing to the entry in Part II.
> Part II is the experiment record: a chronological log of every instrument we
> ran, what it found, and what it changed. Nothing in Part II is ever edited
> after the fact; new entries are appended.
> **Rule: no experimental slice ships until this document reflects it.** This
> document is the project's memory. (Added after an agent regressed to the
> February design during a handoff — see Record entry R9.)

---

# PART I — CURRENT DESIGN

## The Question

We test two rules for choosing reusable subchains.

**Compression** picks subchains that shorten already-solved starter solutions.
It looks backward. It is cheap: no new search runs.

**Utility** picks subchains that reduce search cost on validation tasks. It
looks forward. It is expensive: it runs real searches for every candidate it
scores, and it needs future-like tasks to exist.

We measure when the cheap backward rule suffices, when the expensive forward
rule earns its cost, and — if utility wins — whether the credit goes to fresher
data or to measuring search directly.

## Vocabulary

- A **primitive** is one basic operation.
- A **chain** is a sequence of operations: `OP1 -> OP2 -> OP3`.
- A **solution** is a chain that solves a task.
- A **library** is the set of operations the solver can use.
- A **hidden motif** is a reusable chain the task generator builds tasks from.
  The solver and the selectors never see motifs. Motifs exist so tasks contain
  real structure worth finding.
- The **candidate menu C** is the pool of reusable candidates promoted from the
  primitive frontier using starter-only evidence (see Candidate Menu C). Every
  selection rule chooses from C.
- The **capture ratio** of a cell is (Best-K-from-C solved delta) divided by
  (hidden-motif-oracle solved delta): the fraction of achievable library value
  the menu contains. It is a reported covariate, not a gate (Record R7–R8).

## The Comparison

Every expanded library has the same shape:

```text
library = primitives + K subchains chosen from C     (K = 10)
```

*Amended: K was 5 in the original design; raised to 10 prospectively from
starter-only evidence — Record R4.*

Same menu, same count. Only the choosing rule differs. Every library attempts
the same unseen test tasks under the same frozen solver and budget. K is shared
by all selector arms, baselines, and oracles in any comparison — this is the
fairness invariant. The choosing rules that produce faster search are the
better rules. That is the entire experiment; everything below implements it
fairly.

## Distributions

```text
TASKS_start ~ D_start
D_fut(alpha) = (1 - alpha) * D_start + alpha * D_alt
TASKS_val  ~ D_fut(alpha)
TASKS_test ~ D_fut(alpha)
```

D_start assigns each motif a popularity: some motifs appear in many tasks,
others in few. D_alt uses the same motifs with a different popularity ordering.
Alpha mixes the two, so alpha controls how different the future is from the
past. Validation and test share a distribution but are disjoint task instances
— no task appears in both.

## Two Versions of D_alt

**Reversed.** The most common motif in D_start becomes the least common in
D_alt, and so on down the line. Same ingredients, popularity exactly flipped.
This is worst-case for compression: everything it learned about what's common
is now precisely backwards.

**Permuted.** Popularity is shuffled. The future still uses the same motifs,
but past popularity is less predictive.

We run both. This shows the result holds when the future is hostile *and* when
it's just unrelated — not only in the case rigged against compression.

## Alpha vs. Rho

**Alpha** is the knob we set. **Rho (drift)** is the rank correlation of motif
popularity between D_start and D_fut — the shift that actually happened. Rho is
a function of alpha only on average: two worlds with the same alpha can realize
different rho, because sampling a finite task set adds noise.

The methods experience rho, not alpha. So all results are plotted against
realized rho. This also makes the conclusion portable: "measure drift, pick a
method" works anywhere, while "set alpha = 0.5" only means something inside
this generator.

Compute rho as Spearman rank correlation over motif frequencies. If motifs tie,
assign each tied motif its average rank.

## Setup

1. Freeze the solver, search budget, cost model, subchain cost, timeout policy,
   tie-breaking rule, and primitive set before any results exist. Anything
   tuned afterward could quietly pick the winner. Search is deterministic. One
   selected subchain costs one operation, in both compression scoring and
   search. Candidate programs tried is the main cost metric; wall-clock is
   secondary.
2. Sample 12 hidden motifs — chains of 2 to 4 operations. Tasks use 2 to 3
   motifs plus 0 to 2 glue operations. K is chosen from starter-only marginal
   curves and kept below the motif count to preserve selection pressure.
   *Amended from fixed K=5 — Record R4.*
3. Build D_start with a floor on how rare any motif can be. If a motif is too
   rare to appear in starter tasks, no method could ever discover it, and
   high-drift results would measure extraction failure rather than selection
   quality.
4. Build D_alt twice: reversed and permuted.
5. Hold task length constant across alpha. Otherwise task difficulty drifts
   along with motif popularity, and every result confounds "the future is
   different" with "the future is harder."
6. Generate 100 starter, 25 validation, and 100 test tasks per world.
7. Deduplicate validation against test by behavior, not by generator text.
   Near-duplicates would let the forward-looking method win by memorizing.
8. Record the realized rho for each world.

## Solver Cost Conventions

These conventions are frozen with the solver.

- A primitive leaf has size 0.
- A helper/subchain leaf also has size 0.
- Each operation node has size 1.
- So a selected helper costs the same as a primitive when the solver uses it.
- Compression scoring must use the same convention: helper occurrence =
  primitive occurrence, operation node = 1.
- Mirrored `add` and `overlap` candidates are skipped before execution, so
  they are not included in "candidate programs tried."
- **Evaluation engine.** All library scoring uses the target-independent
  `FrontierIndex`: enumerate once per fixed library under the frozen budget,
  score any target by lookup. This is exact with respect to
  `solve_task(max_solutions=1)` and roughly 50x cheaper than per-target
  solving. Never loop the raw solver per candidate per round. *Added — Record R3.*

## Candidate Menu C

*Amended: this section replaces the original solution-subtree extraction rule
in full — Record R2, R3. It was then characterized and frozen — Record R7, R8.*

1. Build the primitive frontier once with the frozen solver budget
   (`node_budget=30_000`, `max_program_size=7`). The frontier is truncated at
   this budget (`hit_budget=true`); this is known and accepted — larger budgets
   were tested and measurably hurt (Record R8, Experiment 2).
2. Promote frontier grids into C using starter-only evidence. A grid is
   eligible if its first frontier program has 1 to 4 operations, its output is
   not blank or primitive-equal, and it is one cheap step (one binary-op
   completion whose complement is in the frontier, or one unary op) from at
   least two distinct starter task targets.
3. Cap C at 50 candidates, ranked by distinct starter-task support, then
   first-hit frontier cost, then program string. Fairness requires that all
   methods shop from this same frozen menu, built per world from that world's
   own 100 starter tasks. Extraction code never reads validation or test tasks.
4. Report the menu's capture ratio per cell as a covariate. This recipe
   captures roughly 0.5–0.6 of hidden-motif-oracle headroom; that ceiling is
   structural, not an artifact of data size, frontier budget, or evidence
   depth (Record R8). It is one of the project's findings, and the selector
   results are read in its light.

**Why this recipe.** The original rule — extract subchains from starter
solutions — leaked useful structure at two stages: solved starter solutions
route around motif grids (shortest-path bias), and the support filter kills
large fragments (recurrence probability falls exponentially with size).
Output-equivalence pooling was tested first and was a no-op. Frontier promotion
fixed both leaks while preserving starter-only provenance and shared-menu
fairness. Full history: Record R2–R3.

## Menu Quality: Measured, Not Gated

*Added — Record R5–R8.*

Menu quality is a property of the (recipe, world) pair, not of the recipe
alone: primitive baselines swing 54–67 solved across worlds, and menu headroom
varies with them. Two consequences, both learned at cost:

1. **No minimum-over-cells pass rules, ever.** A conjunctive all-cells gate
   over high-variance worlds has a false-failure rate that grows with every
   cell added; for this recipe it was estimated near 80% (Record R6). All
   pass/fail rules are distribution-level: medians across worlds plus a tail
   bound, with pre-frozen thresholds.
2. **Capture ratio is a covariate, not a criterion.** The gate that protects
   the selector comparison must test what the comparison needs (see next
   section) — not recovery of the generator's answer key. Motif-capture
   thresholds were retired after the headroom survey and three rescue nulls
   (Record R7–R8).

## The Selector-Relevance Gate

*Added — Record R9. This gate is registered and terminal in both directions.*

Before the full selector experiment, one registered run establishes that the
comparison is interpretable. It tests exactly the properties the comparison
requires — kill-switches 2, 3, and 4 — and nothing else.

- Seeds: 6477–6480 (fresh). Conditions: `reversed_a0`, `reversed_a1`. Recipe,
  K=10, cap 50, budgets: frozen as above. One look.
- Arms per cell, all K=10 from the same per-world menu: primitives only;
  random-K (20 deterministic draws, full distribution reported); most-frequent-K;
  compression-on-starter; utility-on-validation; Best-K-from-C oracle
  (test-peeking, diagnostic); hidden-motif oracle (diagnostic).
- **Pass, per endpoint, median across the four worlds:**
  (a) oracle solved minus median random-K solved ≥ the frozen noise threshold
  `max(10, 3n)` for that endpoint — selection does real work;
  (b) at `reversed_a0` only: Spearman rho ≥ 0.5 between validation and test
  solved counts across the library spread (20 random + real arms per world) —
  validation can estimate test, utility's premise.
- **Terminal outcomes.** Pass → full experiment on seeds 6481+. Fail (a) → the
  finding is that menu content is interchangeable at K=10 and the selection
  question dissolves in this world class. Fail (b) → the finding is that 25
  validation tasks cannot estimate test gains even at zero drift. No further
  gate revisions in any case; this commitment is part of the registration.
- The compression-vs-utility contrast in this run is reported but not gated;
  it is the first data point of the real experiment.

## Selection Procedure

Every real selector uses the same greedy loop: score each remaining candidate
in the context of what's already selected, keep the best, repeat K times.

```python
selected = []
for round in range(K):
    best = argmax(score(primitives + selected + [c]) for c in remaining)
    selected.append(best)
    remaining.remove(best)
```

**Compression score:** how much the trial library shortens the scoring
solution set, measured by shortest segmentation under that library — not naive
occurrence counting, because subchains overlap. Segmentation matches candidates
by **output grid**, not program string: menu candidates are frontier grids and
need not appear verbatim inside any solution. A matched subtree costs 0 (helper
leaf); every remaining operation node costs 1, per the frozen conventions.
*Matching rule added — Record R9.*

**Utility score:** how much the trial library reduces search cost on the
scoring task set, measured with the frozen solver via `FrontierIndex`.

If utility is too expensive, use two stages: score all candidates cheaply on a
small validation subset, keep the top 30–40, run greedy selection on that
shortlist. Never prefilter for utility alone.

## Library Arms

All non-oracle arms choose only from C, even when scored on validation tasks.

**The two contenders.**

*Compression on starter.* Pick K subchains that most shorten the starter
solutions. The standard method in the literature — the thing we're
questioning. It runs twice: on all 100 starter solutions (the practice
standard) and on 25 (the matched cell for the 2x2 below).

*Utility on validation.* Pick K subchains that most speed up search on the 25
validation tasks. This is the challenger.

**The problem with comparing just those two, and the fix.**

The challenger differs from the incumbent in two ways at once: scoring rule
and data. So we add the two missing combinations:

| | Compression score | Utility score |
| --- | --- | --- |
| 25 starter tasks | compression on starter (25) | utility on starter (25) |
| 25 validation tasks | compression on validation | utility on validation |

Any victory splits into its two ingredients. If compression on validation does
as well as utility on validation, the win was really data freshness, and the
cheapest practical advice follows: don't change your method, refresh your
tasks.

**Fairness details.**

The validation arms see 25 tasks, so the starter cells of the 2x2 are also
restricted to 25 starter tasks.

Compression can only learn from tasks it has solutions for. Compression on
validation runs in two variants: **A. Skip** (only validation tasks primitives
can solve — honest) and **B. Assisted** (extra solve budget spent beforehand
to hand compression solver-produced solutions — fair). Hidden generator
programs are never selector inputs. The A–B gap separates "can't see hard
tasks" from "scores badly even when it sees more of them."

**Three cheap baselines.** *Primitives only.* *Random K from C* (multiple
deterministic draws; report the distribution). *Most frequent K from C.*

**Two oracles, to diagnose failure.** *Best K from C by test peeking* — the
menu's ceiling. *Primitives plus the true hidden motifs* — the world's
ceiling. Hidden motifs are imported only in diagnostic reporting code, never
in extraction or selection code.

## Metrics

- Solve rate on test tasks under the fixed budget.
- Candidate programs tried, including on failures.
- Selection cost: everything a method spent choosing its K subchains. Utility
  selection cost includes the `FrontierIndex` search work spent while scoring
  trial libraries, not just final test-time search. Utility must win net of
  this bill or the win is fake. Starter-task solution programs are treated as
  pre-existing inputs. Compression on validation charges the candidate-program
  search used to produce its own solutions: the skip arm charges its default
  primitive search, and the assisted arm charges its 90,000-node-budget search
  (not the skip search plus the assisted search). Compression trial-library and
  segmentation counts are reported separately because they are not measured in
  candidate programs tried.
- Break-even point: future tasks needed before utility's incremental upfront
  cost pays back. For the registered comparison this is utility-on-validation's
  candidate-program selection cost minus assisted-compression's candidate-program
  selection cost, floored at zero, divided by utility's per-task search-cost
  savings.
- Selected-subchain overlap between methods.
- Motif recovery: precision/recall of selected subchains against hidden motifs
  (diagnostic).
- Capture ratio per cell (covariate).
- Validation/test prediction: whether a method's validation gains track its
  test gains.

## Runs

- Primary K=10 formal run first: 30 worlds, seeds 6481–6510, crossed with
  reversed and permuted D_alt at alpha in {0, 0.5, 1}: 180 main cells.
- One stale-foresight run: validation at alpha = 0.5, test at alpha = 1:
  30 additional cells, reported separately from the main rho curves.
- **Required** sensitivity sweep: K in {2, 5, 10}, paired within each of 30
  fresh seeds (6511–6540) and the six primary conditions. This produces 540
  cells. The stale-validation condition is excluded. The candidate menu, arms,
  search budgets, 25-task validation set, and 20 random draws remain fixed.
  There is no early stopping and no change to the run after any sweep result is
  visible. The sweep records raw cells only; paired analysis and plotting occur
  later. Validation size {10, 25, 50} remains optional.

## Reading the Results

For each level of drift rho, compute two numbers. **Data effect:** improvement
from swapping old tasks for future-like tasks, scoring rule fixed. **Scoring
effect:** improvement from utility scoring over compression scoring, data
fixed. Plot both against rho; those two curves are the finding. Selector
effects are within-world contrasts; per-world capture ratio contextualizes
them.

## Kill Switches

*Amended: switches 2 and 5 restated after the gate-criterion correction —
Record R7–R8.*

1. Hidden-motif oracle fails (distribution-level, median across worlds below
   the frozen floor). The testbed is broken; fix before interpreting anything.
2. Best-K-from-C oracle fails to beat primitives at the distribution level.
   C contains no useful subchains; extraction failed.
3. Random K matches the real selectors (and the oracle). Search rewards
   library size, not content; this world class can't discriminate choosing
   rules. Tested by the selector-relevance gate.
4. Validation gains don't predict test gains at zero drift. The challenger's
   premise dies regardless of drift. Tested by the selector-relevance gate.
5. Menu capture ratio is reported per world as a covariate. Persistently
   near-zero capture across worlds indicates extraction is the bottleneck;
   moderate capture (the measured ~0.5–0.6) does not block the comparison —
   it frames it.
6. No expanded library beats primitives. The tasks, subchain cost, or budget
   is wrong.

## Seed Ledger

Every diagnostic look spends seeds. Contaminated seeds are never reused for
results.

| Seeds | Spent on | Status |
| --- | --- | --- |
| 6460 | Gate 2 test look | contaminated |
| 6461–6462 | primitive noise estimation | contaminated |
| 6463–6464 | Gate 3 test look | contaminated |
| 6465–6472 | headroom survey | contaminated |
| 6473–6476 | rescue-factor diagnostics | contaminated |
| 6477–6480 | selector-relevance gate | contaminated — gate passed |
| 6481–6510 | primary K=10 selector experiment | spent — formal run complete |
| 6511–6540 | registered K-sensitivity sweep | spent — formal run complete and inspected |
| 6541–6570 | registered R13 capacity curve; reused by R14 mechanism follow-up | spent — capacity curve complete and inspected |
| 6571+ | unused | virgin |

Every gate/survey command carries a fresh-look guard that scans existing
artifacts and refuses to rerun on spent seeds.

The primary formal selector runner is intentionally one-shot. It atomically
claims the formal artifact directory before generating any formal world. Any
formal aggregate, formal cell artifact, or prior claim blocks a fresh run;
interrupted formal runs require an explicit repair decision rather than silent
cache reuse. A cell whose candidate menu contains fewer than K candidates
aborts instead of emitting undersized libraries. Every seed-bearing runner and
world-generation boundary requires an exact integer seed. Numeric strings are
rejected before generation, and formal-seed strings found in artifact metadata
block the fresh-look guard.

---

# PART II — EXPERIMENT RECORD

Append-only. Each entry: what was run, what it showed, what changed.

### R0 — Original design (February 2026)
K=5. C = all subchains of length ≥ 2 from primitive-only starter solutions
(up to 3 per task), deduped by program string, support ≥ 2 distinct solutions,
gated on future-motif-mass coverage. Binary "C good enough" pre-gate implied.
Preserved here as the baseline all amendments are measured against.

### R1 — Gate 1: original C fails (seed 6460, K=5)
Best-5-from-C: +6 / +8 solved vs thresholds +10 / 25% cost. Hidden-motif
oracle passed (+15 / +25), so the world rewards libraries and extraction was
the bottleneck. Leak attribution on all 12 motifs: 0 missing from the frontier,
7 routed-around in solved solutions, 5 killed by the support filter.
**Change:** reserved failure branch fired; extraction redesign authorized.

### R2 — Output pooling: exact no-op
Starter-split harness (extract on 50, score on held-out 50, both directions).
Pooling by output grid merged spellings but changed nothing: 0 output-duplicate
pairs existed. Fragmentation was falsified as the leak.
**Change:** none to the recipe; hypothesis eliminated.

### R3 — Frontier promotion adopted; FrontierIndex built
FrontierIndex: target-independent enumeration, exact vs
`solve_task(max_solutions=1)`, ~50x cheaper; became the evaluation engine.
Frontier-promotion menu beat the subtree menu on the starter harness
(+4.5 solved mean, +32.8pp cost). All 12 motif grids present in the primitive
frontier — extraction was a promotion problem, not a reachability problem.
**Change:** C recipe replaced (Part I, Candidate Menu C).

### R4 — Gate 2 fails at K=5; K amended to 10 prospectively (seed 6460)
V2 menu: +9 (a0, threshold 10.82) / +16 (a1, threshold 13.53). a1 passed; a0
missed by ~2. Best-10 diagnostics (+16/+23) and the motif ceiling (+15 at a0
with 5 slots) showed the container, not the contents, was binding at a0.
K amendment justified from starter-only evidence only: 12 motifs x 2–3 per
task; harness marginal curves. Registered K choice: 8 if it captures ≥90% of
K=10's starter-heldout gain, else 10 → **10**. Noise term formalized:
threshold = max(10, 3n), n = between-seed primitive stddev.
**Change:** K=10; K-sensitivity sweep promoted to required; seeds 6460–6462
retired.

### R5 — Gate 3 fails 1 of 4 cells (seeds 6463–6464, K=10)
Cells: 6463/a0 +3 (fail), 6463/a1 +12 (pass), 6464/a0 +17 (pass),
6464/a1 +21 (pass). Registered rule was all-cells; verdict `not_ok` honored.
**Change:** none immediately; attribution ordered.

### R6 — Instrument characterization: the conjunctive gate was broken
From spent data only: per-cell headroom at K≈10 has mean ≈12 (a0) / ≈19 (a1)
with stddevs ≈8 / ≈6 against thresholds ≈11 / ≈13.5. Estimated probability
that a menu exactly as good as observed passes all four cells: **≈0.21** —
an ~80% false-failure rate. Root cause: minimum-over-cells statistics under
world variance (primitive baselines 54–67).
**Change:** min-over-cells rules banned; distribution-level survey registered
(median capture ≥ 0.5 per endpoint, ≤25% of cells below 0.25, low-headroom
cells excused); Part I "Menu Quality" section added.

### R7 — Headroom survey fails just under the bar (seeds 6465–6472)
16 cells, 0 low-headroom exclusions. Median capture: 0.479 (a0), 0.468 (a1)
vs registered 0.5. Verdict honored: menu insufficiency earned *under that
criterion*. Noted: 0.5 was pre-registered but never derived; capture is
nearly identical at both endpoints, so drift is not the cause.
**Change:** triggered rescue-factor experiments before accepting the finding.

### R8 — Three rescue nulls; ceiling declared structural; gate criterion corrected (seeds 6473–6476)
Experiment 1 (n_start 100/300/1000): capture ~0.55→0.59 mean, plateau by 300.
Experiment 2 (frontier budget 30k/100k/3M): 3M drives capture **negative** —
support-then-cost ranking drowns in generic grids under the cap-50; 30k frozen.
Experiment 3 (evidence depth 1 vs 2): pool triples, capture flat.
Conclusion: ~0.5–0.6 capture is the structural ceiling of starter-only
one-step evidence. This is a project finding.
Criterion correction: the gate must protect the selector comparison
(kill-switches 2/3/4), not motif recovery. Capture demoted to covariate.
**Change:** Part I "Menu Quality" finalized; selector-relevance gate designed.

### R9 — Selector-relevance gate passed (seeds 6477–6480); doc discipline added
A draft selector slice regressed to the February design (subtree C, K=5,
spent seed 6460) because this document lagged the decisions. Registration
frozen as in Part I: oracle-vs-random gap ≥ max(10, 3n) per endpoint (median
across worlds) and val→test Spearman ≥ 0.5 at a0; terminal both directions;
compression-vs-utility reported but not gated. Compression segmentation
matches by output grid (consequence of R3). Doc rule added to the header.
The gate passed: reversed_a0 oracle-minus-random median 17.0 vs threshold
10.95; reversed_a1 median 22.5 vs threshold 10.78; validation/test Spearman
rho 0.696 vs threshold 0.5.
**Change:** selector comparison is cleared; next registered step is the
primary K=10 full selector experiment on seeds 6481+.

### R10 — Primary run complete; K-sensitivity run frozen
The primary K=10 experiment spent seeds 6481–6510. Before running the required
sensitivity experiment, its shape was frozen: K in {2, 5, 10}; seeds
6511–6540; the six primary conditions; and 20 random draws. K is paired within
every seed and condition, for 540 cells total. All other menus, arms, budgets,
and task-set sizes remain unchanged. The stale-validation condition is not
repeated. The runner writes raw cells only and cannot silently resume a claimed
formal directory.
**Change:** seeds 6511–6540 are reserved for the registered K sweep; analysis
and plotting are deferred until data generation is complete.

### R11 — K=20 follow-up frozen after inspecting the registered sweep
The registered K={2, 5, 10} sweep is complete and has been inspected. A
prospective follow-up adds K=20 only, using the same seeds 6511–6540 and six
primary conditions. It produces 180 new cells. The candidate menu, arms,
budgets, 25-task validation set, and 20 random draws remain fixed. Each new
cell is paired with its stored K=10 cell by seed and condition. The follow-up
reports gain over primitives, utility versus standard past compression, the
matched scoring and data effects, and paired K=20-minus-K=10 changes for past
compression and future utility. There is no early stopping and no additional
K after these results become visible. This is a registered follow-up, not part
of the original confirmatory sweep. Primary selectors choose exactly 20 menu
candidates. The diagnostic hidden-motif oracle instead saturates at all 12
true motifs; it is not padded with non-motifs. Its data remain separate and
additive.
**Change:** K=20 is frozen as the only follow-up library size; existing sweep
artifacts remain authoritative and read-only.

### R12 — R11 withdrawn before formal execution
R11 was withdrawn before any formal K=20 cells were generated. One smoke cell
used spent seed 6460 under `reversed_a0`. Its artifacts remain preserved at
`experiment/data/selection/k20_extension_smoke/`, but they are not scientific
evidence. The formal K=20 output directory and aggregate were never created.
The completed K={2, 5, 10} sweep remains authoritative and unchanged. R11 must
not be executed. Any replacement capacity experiment requires a separate
prospective registration before it consumes fresh seeds.
**Change:** the same-seed, endpoint-only K=20 follow-up is abandoned; this
record changes no completed result or data artifact.

### R13 — Fresh K=0--20 capacity curve registered before formal execution
The replacement follow-up uses 30 unseen seeds, 6541--6570, crossed with the
six primary reversed and permuted conditions. Each of the resulting 180 cells
selects one ordered 20-program path for compression on 100 starter solutions,
assisted compression on 25 validation problems, utility scoring on those same
validation problems, and a greedy test-peeking diagnostic. Twenty random menu
permutations are derived from `(seed, draw)`. Every path is evaluated at each
nested prefix K=0--20. K=0 is the shared primitive baseline.

The candidate menu remains capped at 50. A formal world with fewer than 20
candidates aborts the run without seed replacement. The generator, tasks,
budgets, solver, tie-breaking, and assisted-validation solve configuration
remain fixed. Assisted compression records validation-solution acquisition
cost separately and charges it once to every nonzero prefix. The test-peeking
path is diagnostic because it uses the test problems for selection.

The five primary test-set contrasts are U(20)-Cv(20), the change in that
scoring effect from K=10 to K=20, U(20)-C100(20), U(20)-U(10), and
C100(20)-C100(10). They form one single-step max-t family. Inference averages
the six conditions within each seed and uses 10,000 seed-cluster resamples from
`random.Random(20260713)`. The 95% critical value is sorted bootstrap maximum
element 9499. Descriptive trajectories use separate simultaneous bands across
K=1--20. The inspected K={2,5,10} cohort is never pooled with this fresh cohort
for primary inference.

No early stopping, post-result K additions, test-based K choice, or silent
artifact reuse is allowed. The formal directory is claimed once. This record
was written before any seed from 6541--6570 was generated or inspected.
**Change:** R13 prospectively replaces R11. R11 and R12 remain as the history
of an abandoned design; the completed K={2,5,10} data remain unchanged.

### R14 — Fixed-library budget intervention registered after inspecting R13
The R13 capacity curve showed a sharp test-performance drop from K=1 to K=2.
This follow-up tests whether the fixed 30,000-candidate evaluation budget caused
that drop by restricting access to larger abstract search programs. It reuses
the completed R13 worlds, tasks, and selected library paths. It does not rerun
selection. It is a registered mechanism follow-up on inspected worlds, not an
independent confirmation.

The authoritative source is
`experiment/data/selection/capacity_curve/full_selection_experiment_capacity_curve.json`,
SHA-256
`a70e2156897da8473e2891d2f2e9daaf78142400fbda696f479bae985fa62f3d`.
The run uses seeds 6541--6570, the same six conditions, K in {0,1,2}, and
evaluation budgets in {30,000,45,000,60,000,90,000}. The solver, enumeration
order, tasks, maximum abstract search size seven, helper order, and atomic-helper
cost remain fixed. K=1 and K=2 use the exact stored library prefixes.

The 90,000-candidate ceiling was chosen before inspecting any above-30,000
outcome for seeds 6541--6570. Calibration used spent K-sweep seeds 6511--6520
under `reversed_a0`, from
`experiment/data/selection/k_sweep/full_selection_experiment_k_sweep.json`,
SHA-256
`1454be9e733ff655bb9761465bab5d4166a2cfe1a9bb7b233442bd16fa7b787e`.
At budgets 30k/45k/60k/90k, K=2 reached size-four search in 0/10, 10/10,
10/10, 10/10 worlds for past compression; 1/10, 10/10, 10/10, 10/10 for
assisted compression; 1/10, 10/10, 10/10, 10/10 for future utility; and 9/10,
10/10, 10/10, 10/10 for the test-peeking diagnostic. The corresponding mean
K=2-minus-K=1 test differences were (-9.2, +1.1, +3.2, +3.4), (-8.5, +0.2,
+1.0, +1.8), (-5.8, -1.8, +1.3, +1.5), and (-0.6, -0.3, +0.8, +0.9).

For primary method m in {past compression, future utility}, let J_m(k,B) be
the number of test problems solved out of 100. Define

```
D_m(B) = J_m(2,B) - J_m(1,B)
I_m    = D_m(90000) - D_m(30000).
```

The six condition-level contrasts are averaged within each seed. The estimate
is the mean across 30 seeds. I_past-compression and I_future-utility form one
single-step max-t family. The procedure uses 10,000 seed-cluster resamples from
`random.Random(20260714)`. Each resample draws 30 seeds with replacement. A
standard error is the sample standard deviation of the 30 seed values divided
by sqrt(30). Each bootstrap statistic is the maximum absolute centered
studentized deviation across the two effects. The 95% critical value is
zero-based sorted bootstrap maximum element 9499. Each interval is estimate
plus or minus the critical value times its standard error. A
zero-standard-error endpoint receives a point interval and is omitted from the
maximum. No p-values are introduced.

A familywise lower bound above zero supports the claim that additional budget
attenuates the selected K=2 deficit for these fixed libraries in this inspected
cohort. D_m(90000) is also reported to distinguish attenuation from elimination;
I_m alone cannot establish that the deficit fully closes. A positive I_m whose
interval crosses zero is suggestive only. Neither outcome is called independent
confirmation or population-general.

For each primary method and cell, the lost set contains test problems solved at
K=1 and 30,000 candidates but unsolved at K=2 and 30,000 candidates. At every
registered budget, the run records how many lost problems K=2 recovers, their
first-hit abstract search sizes, all attempted candidates by search size, the
first size-four candidate rank, and whether the budget exposes that rank.
Failures are charged the effective candidate total. These checks are
descriptive. They do not establish formal mediation. Assisted compression,
primitive-only, random draw 0, validation outcomes, and the fixed test-peeking
path remain descriptive. The test-peeking path was selected using the stored
30,000-candidate test set and is not reoptimized at larger budgets.

Every library is enumerated once to at most 90,000 candidates. Results for
30,000, 45,000, and 60,000 are exact deterministic prefixes of that search.
Before any intervention cell is written, its 30,000-candidate validation and
test summaries must exactly reproduce the authoritative R13 summaries for
primitives and every included K=1 and K=2 path. The R13 and K-sweep artifacts
remain byte-for-byte unchanged. The new formal output directory is claimed
once, and interrupted runs cannot silently resume.

**Change:** R14 adds a fixed-library evaluation-budget intervention. It changes
no completed selection, capacity, K-sweep, plotting, or report artifact.
