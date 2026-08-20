# Architecture Exploration — Panel Results & Decision

> *Historical note:* the incumbent design carried the working name **ChronoBoost** while
> this panel ran; it ships as **Heartwood**. The name has been updated throughout for
> consistency — the design it refers to is unchanged. The five challenger names (GRAFT,
> RocketFuse, RegimeBoost, ChenBoost, AttuneBoost) are as the designers proposed them.

**Question:** what is the best algorithm architecture for learning from mixed
**static + raw time-series** data in the small-data regime (n ≈ 100–50k, laptop CPU,
pure NumPy, interpretable, classification + regression, NaN/variable-length native)?

**Method:** five designer agents each proposed one architecture from a distinct school
of thought; three independent judges (pragmatic industry engineer, academic reviewer,
skeptical benchmarker) scored every proposal *and* the incumbent Heartwood design
(PLAN.md) on weighted criteria: small-data accuracy (0.25), temporal fidelity (0.20),
static×temporal interaction modeling (0.15), implementability by a mid-level
engineer in pure NumPy (0.15), interpretability (0.10), training speed (0.05),
novelty (0.10).

---

## 1. The six candidates

### Incumbent — Heartwood (PLAN.md)
XGBoost-style second-order booster; each tree node picks the max-gain split among
static thresholds, random-window interval statistics, and shapelets sampled from
in-node rows (min z-norm distance + match position).
**Strength:** leanest, fully specced, lowest implementation risk; *natively conditional*
(per-node search handles XOR-with-static signals structurally).
**Weakness (named independently by every serious proposal):** blind per-node candidate
sampling pays a "luck tax" — at n=100 with 20 candidates/node it must get lucky to find
narrow noisy temporal signal, and its shapelets are noisy single-row snippets.

### 1. GRAFT — *boosting-theory lens*
Keeps the boosting spine; re-architects the temporal search: a per-round gradient
**saliency map** over timesteps aims candidate windows/shapelets; admitted features go
into a **persistent feature bank** (materialized once, reused by all later trees);
top features get **mutations** including a one-step shapelet *centroid refinement*
(denoise by averaging best-matching windows); per-node "wildcard" candidates that win
get **promoted** into the bank.
**Strength:** feature reuse (compute once, split forever; cheap inference), aimed search.
**Weakness:** saliency is a *marginal* (root-level) linear probe — blind on
interaction-gated signals, which are exactly 3 of the 5 planned benchmark scenarios;
heaviest bookkeeping surface (eviction, credit, promotion) = silent-bug farm.

### 2. RocketFuse — *time-series-classification SOTA lens*
Three stages: (i) a **deterministic label-free temporal bank** — MiniROCKET-style
84-kernel dilated convolutions with five poolings (incl. positional "when" poolings)
+ a dyadic interval-stat pyramid that *provably contains the aggregate baseline*;
(ii) **exact leave-one-out ridge** on the bank (closed-form LOO via SVD hat-diagonals)
whose *out-of-fold* margins become the boosting base score — leak-free stacking at
n=100; (iii) a second-order boosted tree stage over statics ∪ bank ∪ the margin itself
(static-gates-temporal) ∪ **rank-comparison splits** `rank(position feature) −
rank(static col)` — a one-split answer to "did the event happen before this row's
deadline".
**Strength:** the only proposal whose small-n advantage rests on published evidence
(ROCKET-family dominance on small UCR/UEA train sets) *and* airtight statistics
(exact-LOO cross-fitting); ridge aggregates thousands of weak features — the one
capability trees fundamentally lack.
**Weakness:** ridge base is flat on pure-interaction (XOR) tasks; 5,000-coefficient
linear layer weakens the every-split-readable story; stacking hygiene is a silent
failure mode if implemented sloppily.

### 3. RegimeBoost — *probabilistic / functional lens*
Pooled Kalman (local-linear-trend) smoother + pooled HMM fit across all rows once;
every series becomes denoised **posterior trajectories** (level, slope, innovations,
regime occupancies) that the booster's interval splits consume; plus per-round
**likelihood-ratio "probes"** — pairs of tiny HMMs fitted by weighted EM to the signed
boosting residuals, their per-row LLR offered as a candidate feature.
**Strength:** principled exact NaN marginalization; pooling = real small-n statistics;
regime/duration/ordering structure no shapelet can express.
**Weakness:** NaN-aware vectorized Kalman + scaled forward–backward in pure NumPy is
the worst numerics-per-benefit ratio of the batch; Stage-0 optimizes an unsupervised
objective (capacity spent on variance irrelevant to y).

### 4. ChenBoost — *kernel / path-signature lens*
Log-signature coordinates (Lyndon basis) of time-augmented observed-point paths, over
a dyadic window lattice computed by one Chen tree-reduction; per-node adaptive window
signatures via Chen quotients; boosted-tree head (monotone-invariance neutralizes the
notorious signature-scaling fragility).
**Strength:** the only design with native **cross-channel lead-lag** (Lévy areas) and
one-scalar event-timing coordinates; NaN handled as irregular sampling, eliminating
the z-norm masking pitfall class entirely.
**Weakness:** hand-rolled truncated tensor algebra with no reference library to test
against is a correctness trap; on the C=1 localized-template benchmarks that dominate
the target use case, it's the wrong inductive bias.

### 5. AttuneBoost — *wildcard lens*
Replaces blind shapelet sampling with **gradient-attuned matched filters**: per node,
9-tap filters at dyadic NaN-aware pooling scales, seeded from a residual-weighted donor
row, then **refit in closed form** (5-coefficient DC-free DCT basis, hessian-weighted
ridge on Newton residuals, one align↔refit alternation — no SGD). Emits signed NCC
response ("does the learned pattern occur, with what polarity") + normalized argmax
position ("when"). Iteration-0 (no refit) **exactly reproduces Heartwood's shapelet
features** (dist = 2(1−ρ)) — the family strictly contains the incumbent's.
**Strength:** solves for the feature instead of sampling and hoping; 5 degrees of
freedom per filter = a template that *cannot* memorize a noisy snippet; the deepest
static×temporal mechanism on the table (the filter is fitted to the residuals of the
static subpopulation carved out above the node); smallest code delta (~+400–500 LOC).
**Weakness:** fitted candidates can inflate in-sample gain at tiny noisy nodes (the
in-sample gain tournament does not arbitrate overfitting — the DCT capacity limit and
a mixed fitted/unfitted default are the real defenses).

---

## 2. Scores (total_weighted, per judge)

| Design | Engineer | Academic | Benchmarker | Mean |
|---|---|---|---|---|
| **AttuneBoost** | **8.05** | 7.73 | **7.70** | **7.83** |
| RocketFuse | 7.55 | **7.80** | 7.20 | 7.52 |
| GRAFT | 7.60 | 7.00 | 6.80 | 7.13 |
| Heartwood (incumbent) | 7.30 | 6.90 | 7.05 | 7.08 |
| RegimeBoost | 6.50 | 6.75 | 6.45 | 6.57 |
| ChenBoost | 6.35 | 6.65 | 5.65 | 6.22 |

Winners: AttuneBoost (engineer, benchmarker), RocketFuse (academic, "by a nose ...
within noise" over AttuneBoost). The incumbent beat RegimeBoost and ChenBoost outright
— lean-and-testable beats fancy-but-risky at this LOC budget — but three designs beat
the incumbent, so shipping PLAN.md v1 unchanged would leave accuracy on the table.

## 3. Where the judges converged

All three judges independently recommended **the same hybrid shape**: keep the
Heartwood spine (losses, gain math, scan_threshold, tree, API, NaN discipline, tests)
verbatim, then add, in order of value-per-risk:

1. **AttuneBoost's matched-filter family** as the shapelet upgrade — with a **mixed
   default of fitted + unfitted candidates** (unfitted ≡ incumbent shapelets), so the
   hypothesis space strictly contains v1 and risk is monotone. *(3/3 judges)*
2. **GRAFT's bank, stripped to "winners-only promote-on-win"** — any temporal candidate
   that wins a node gets its column cached and re-offered everywhere, capped; no
   saliency, no bandits, no mutations, no root-screening. Buys the reuse/inference/
   shared-dictionary wins with near-zero bookkeeping risk. *(2/3 judges; 3rd compatible)*
3. **RocketFuse's rank-comparison splits** `ECDF(position feature) − ECDF(static col)`
   — the one-split answer to event-time-vs-per-row-deadline tasks. *(2/3 judges)*
4. **RocketFuse's exact-LOO ridge base margin** over a deterministic interval pyramid —
   as an **opt-in flag**, because it is the only mechanism that aggregates dense/diffuse
   weak signal (trees can't), but it is dead weight on XOR tasks and adds a stacking-
   hygiene failure mode. *(judge 1: opt-in; judge 2: yes; judge 3: skip → opt-in wins)*
5. **ChenBoost's Lévy-area columns** for C≥2 only — computable with cumsums in ~50 LOC,
   no tensor-algebra engine; the single irreplaceable signature feature. *(judge 2)*

**Rejected for v1:** GRAFT's saliency/foundry/mutation machinery (blind on conditional
signals, biggest bookkeeping risk); RegimeBoost's Kalman/HMM stage (worst
LOC-per-benefit; revisit for heavy-missingness domains); ChenBoost's full signature
engine (correctness trap; revisit for multichannel lead-lag problems); MiniROCKET conv
bank (defer until benchmarks show dense-signal scenarios losing); TabPFN-style
prior-fitted approaches (violate CPU/no-pretraining constraints — considered and
disqualified honestly by two designers).

## 4. Decision

**Final architecture — "Heartwood" (package `heartwood`), built in phases:**

- **Phase A (v0.1):** the incumbent core exactly as specced in PLAN.md §1–§9.
  It is the spine everything else plugs into, it is the easiest to implement correctly,
  and it is already a publishable baseline.
- **Phase B (v0.2):** the three convergent additions — matched-filter family (mixed
  fitted/unfitted), winners-only bank, comparison splits — each additive, each behind a
  parameter, each with mandatory regression tests (specced in PLAN.md §10).
- **Phase C (opt-in extras):** `dense_base=True` LOO-ridge base margin; Lévy-area
  columns when C≥2 (PLAN.md §11).
- Every addition must prove itself on the PLAN.md §5.8 benchmark grid via ablations
  (core / +filters / +bank / +comparison / +dense_base); pure-static control must stay
  within ±2 points of XGBoost with all temporal machinery enabled.

**Revisit triggers:** heavy informative missingness → RegimeBoost derived channels;
multichannel lead-lag domains → ChenBoost views; dense diffuse signal losses on
benchmarks → MiniROCKET conv bank inside the dense base.

## 5. The three judges' implementation warnings (verbatim risks → now mandatory tests)

1. **NaN/z-norm silent corruption, now at every pyramid scale:** any sliding window
   touching NaN must be masked to −inf *before* the argmax over |NCC|; all-NaN pools
   must yield NaN, not 0. Plant-a-NaN regression test per scale.
2. **Iteration-0 equivalence:** with n_alt=0, dist = 2(1−ρ) must reproduce
   `shapelet_features` to allclose — otherwise the NCC itself is wrong.
3. **Gain-leak ban:** a fitted filter is scored ONLY by the exact `scan_threshold` on
   its emitted feature column, never by its ridge-refit fit; β stored via `.copy()`;
   DC component excluded from the DCT basis (else the solve degenerates silently).
4. **Fit/predict symmetry:** predict-time alignment must reproduce fit-time feature
   values bit-for-bit (same ε rules, signed-response convention, missing_left routing);
   test that predict-on-train reproduces fit-time leaf assignment exactly.
5. **If dense_base is implemented — stacking hygiene:** the train-time base margin must
   equal the closed-form LOO predictions `(ŷᵢ − hᵢᵢyᵢ)/(1 − hᵢᵢ)`; add a pure-noise-labels
   test where a leak-free implementation shows chance-level eval performance.
