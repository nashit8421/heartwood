# Validation Plan V6 — beat MiniROCKET

**Status: PRE-REGISTERED.** Written before any V6 code was run. Same rules as
`VALIDATION.md` §2 and `VALIDATION_V5.md` §6, for the same reason: this is now the third
attempt at a claim, and the previous two both produced results that looked good until
someone checked the baseline.

---

## 1. What the evidence says to build

V5 left a precise diagnosis (`validation/HEADROOM.md`). Heartwood beats a fair aggregate
baseline in the shape regime on 7 of 8 cells, by +3.8 to +14.5 points. It loses to
MiniROCKET nearly everywhere — mean rank 1.50 against 2.33. And the deficit is **not** a
search-budget problem: ×16 the candidate budget moves a ~20-point gap by +1.5 points, and
makes two of three datasets worse.

The reason is structural. Each node keeps the single highest-gain split from a pool of
random draws, so enlarging the pool raises the winner's expected gain whether or not
anything in it is informative. Heartwood cannot buy accuracy with candidates. MiniROCKET
has no such ceiling because it never selects: it computes ~10,000 fixed dilated-kernel
features and hands **all** of them to a ridge, which shrinks them jointly.

So the move is not to search better. It is to stop selecting for the part of the problem
where selection is the bottleneck, and keep selecting where it is not:

* a **ridge over a dilated-convolution bank** as the boosting base — the thing MiniROCKET
  is, used as a starting point rather than a competitor;
* **gradient-boosted trees on top**, which add what a ridge structurally cannot: static
  covariates, static × temporal interactions, and nonlinearity.

The floor of this design is MiniROCKET's own accuracy. The upside is the half MiniROCKET
has no answer for at all, which is the half this project was started for.

## 2. What is being built

`heartwood/rocket.py` — a MiniROCKET-style feature bank in numpy: 84 length-9 kernels with
weights in {−1, 2}, exponentially spaced dilations, per-kernel biases drawn from the
training data's own convolution quantiles, pooled by proportion-of-positive-values.

`dense_features` on the estimator: `"stats"` (today's window statistics), `"rocket"`, or
`"both"`. `DenseBase` already accepts an arbitrary bank and already returns honest
leave-one-out margins with a guard against the p ≫ n interpolation trap, so it is reused
unchanged.

Interpretability is a real cost and is stated up front: the base is a black box. Every
tree split on top stays readable, so `dump_splits()` reports what the model learned
*beyond* the rocket baseline rather than the whole model. That is a weaker claim than v0.3
made and it is the honest one.

## 3. Hypotheses — frozen, with numeric pass/fail

**H-V6.1 — the implementation is not broken (parity floor).** On the V5 Arm B datasets
(UEA ranks 9–16, no static block), Heartwood with `dense_features="rocket"` lands within
**2 points** of the better `aeon` MiniROCKET on **≥ 75%** of datasets.
*Fails if* it is more than 5 points behind on a majority. This is a bug check, not a
result: if a ridge over our own convolution bank cannot match a ridge over theirs, the bank
is wrong and nothing downstream means anything.

**H-V6.2 — the win.** On mixed datasets with a non-degenerate static block — PTB-XL
primarily, ICU and credit reported alongside — Heartwood beats the better MiniROCKET by
**≥ 2 points** at a **majority of training sizes**.
*Fails if* it beats it at no size. This is the hypothesis the project now lives or dies on.
PTB-XL is the fair test because it is the only dataset here with both a real static block
and shape-regime series.

**H-V6.3 — no regression.** On the six synthetic scenarios, the new configuration loses no
more than **2 points** to the current default on any scenario × size cell.
*Fails if* any cell drops by more than 5 points. The existing wins were earned and are not
being traded away silently.

**H-V6.4 — earns its default.** `dense_features="rocket"` becomes the default only if
H-V6.2 passes and H-V6.3 does not fail. Otherwise it ships opt-in with the numbers
recorded, exactly as matched filters and the ridge base did before it.

## 4. Rules

No tuning per dataset. Matched budgets. Every dataset and every baseline reported. Splits
and seeds fixed in advance and never redrawn. Five subsample seeds. MiniROCKET is credited
with the better of 2,000 and 10,000 kernels, because a baseline should not lose by being
under-run.

**Comparisons are against `aeon`'s MiniROCKET, not our own reimplementation of it.** Grading
ourselves against our own copy of the opponent is how a benchmark flatters itself, and this
repository has already produced one of those.

If H-V6.2 fails, that goes in the README as the headline and the conclusion is that trees
over searched temporal features do not beat convolution banks on this data — which, after
three studies, would be a genuine and publishable negative result about the approach rather
than about the implementation.
