# Validation Plan V8 — make the trees stop paying a tax

**Status: PRE-REGISTERED.** Written and committed before the change was implemented or run.
Fifth study here. The first two produced results that collapsed under audit; the discipline
is why the later three did not.

---

## 1. What V7 found, and the hole it leaves

The V7 decomposition at n=1000 is the same on a heart and on a brain:

| arm | CPSC | Sleep-EDF | |
|---|---|---|---|
| A — `aeon` MiniROCKET | 0.654 | 0.660 | the reference |
| B — ridge over our bank | 0.654 | 0.661 | bank quality **+0.0 / +0.1** |
| C — + trees, no statics | 0.609 | 0.604 | trees add **−4.6 / −5.7** |
| D — + statics | 0.691 | 0.673 | statics add **+8.3 / +6.9** |

Read C. **Boosting trees onto a good base, with no static block to work with, destroys
about five points.** The final model wins anyway because the statics are worth more than
the trees cost — but it is paying a tax on the way, and the tax is measurable and
consistent across two unrelated modalities.

Why it happens is already established, in `validation/HEADROOM.md`. A node keeps the single
highest-gain split from a pool of randomly drawn candidates, so the winner's expected gain
rises with pool size **whether or not anything in the pool is informative**. Measured there:
16× the candidate budget moved a 20-point deficit by 1.5 points and made two of three
datasets *worse*. Selection is the ceiling. With a strong base underneath, that same
selection bias now has something good to damage.

**V8 removes the tax rather than working around it.**

## 2. The change

One intervention, aimed squarely at the diagnosed cause: **a split must beat what chance
achieves on the same candidates.**

At each node, after scoring every candidate, permute the gradient/hessian pairs across rows
— destroying any relationship between features and the target while preserving both
marginal distributions — and rescore the *same* candidates. The best gain under permutation
is what this node's pool would produce from noise alone. The real best split is accepted
only if it exceeds that floor; otherwise the node becomes a leaf.

This is the same instrument that fixed the ridge base in V6, applied one level down. There
it stopped a null ridge from being boosted into confident nonsense; here it should stop a
noise split from being taken seriously.

Exposed as `selection_null: int = 0`, **off by default**, so nothing about the shipped model
changes unless V8 earns it. `selection_null=1` is one permutation per node, which roughly
doubles the split scan and therefore fit time. That cost is part of the trade and is
reported, not hidden.

## 3. Hypotheses — frozen, with numeric pass/fail

**H-V8.1 — the tax comes off.** On CPSC-2018 and Sleep-EDF at n=1000, arm C (trees, no
statics) is within **2 points** of arm B (base alone). It is currently −4.6 and −5.7.
*Fails if* it remains worse than −4 on both.

**H-V8.2 — the win gets bigger.** The margin over the better `aeon` MiniROCKET at n=1000
improves by **≥ 2 points on at least one dataset**, and falls by no more than 1 point on
either.
*Fails if* it falls by more than 1 point on both. This is the hypothesis that decides
whether removing the tax converts into accuracy, or whether the trees were only ever
useful *because* they were reckless.

**H-V8.3 — no regression.** On the six synthetic scenarios, no cell is worse than the
current default by more than 2 points.
*Fails if* any cell drops by more than 5. The synthetic wins were earned and are not being
traded away silently.

**H-V8.4 — earns its default.** `selection_null` ships on only if H-V8.1 and H-V8.2 pass
and H-V8.3 does not fail. Otherwise it stays opt-in with the numbers recorded, as matched
filters, the ridge base and the convolution base all did before it.

## 4. Rules and scope

`VALIDATION.md` §2 in force: no tuning, matched budgets, splits fixed by seed, every arm
reported, comparisons against `aeon` rather than our own copy of it.

`selection_null` is fixed at **1** a priori. It is not swept. A permutation count chosen by
trying several and keeping the best would be a threshold fitted to the answer, which is the
habit that produced the v0.3 headline.

Runs, scoped by cost — a single CPSC cell at n=1000 is ~3500 s and this change roughly
doubles it:

* CPSC-2018 and Sleep-EDF at **n=1000 only**, arms C and D, 3 seeds.
* The full synthetic grid for H-V8.3.
* Larger sizes are not run. n=1000 is where the decomposition that motivated V8 was
  measured, so it is where the prediction is tested. Anything not run is reported as not
  run, with this reason.

## 5. What counts as done

If H-V8.1 and H-V8.2 pass, the ceiling identified in `HEADROOM.md` is not merely worked
around by the convolution base but actually lowered, and the README says so. If H-V8.1
passes and H-V8.2 fails — the tax comes off but accuracy does not improve — that is the
more interesting result, because it would mean the trees' contribution to the win is
inseparable from their recklessness, and the honest conclusion is that this architecture's
value is the static block and nothing else.
