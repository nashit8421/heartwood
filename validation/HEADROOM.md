# Is the gap to MiniROCKET search or signal?

**EXPLORATORY, NOT PRE-REGISTERED.** This is tuning, deliberately, and it can never become
a headline number. `VALIDATION_V5.md` §6 puts algorithm changes out of scope for V5; the
only job of this document is to decide what a V6 would even be, and it answers that in the
negative.

## The question

V5 left one clean fact. Heartwood beats a fair aggregate baseline in the shape regime on
7 of 8 cells, by +3.8 to +14.5 points — and loses to MiniROCKET nearly everywhere, mean
rank 2.33 against 1.50. On Handwriting the deficit is ~20 points.

Two explanations, and they imply opposite next moves:

* **search** — the useful features are inside the hypothesis space, but ~4 random shapelet
  draws and 16 random interval draws per node almost never find them. Fix by targeting.
  This is what README's "known limitations" has assumed since v0.1: *"Better targeting
  rather than a bigger budget is the open problem."*
* **signal** — interval statistics and shapelet distances cannot express what a bank of
  dilated convolutions expresses on this data, and no amount of drawing will change that.

They are distinguishable without building anything: multiply the candidate budget and see
whether the deficit moves.

## Method

`n_interval_candidates` 16 → 64 → 256 and `n_shapelet_candidates` 4 → 16 → 64, i.e. ×1,
×4, ×16. Three seeds, library defaults otherwise, official UEA splits. Three shape-regime
datasets from V5 Arm B, chosen by how large the MiniROCKET deficit was, so any effect has
room to show.

## Result

| dataset | default | ×4 | ×16 | best MiniROCKET | ×16 fit cost |
|---|---|---|---|---|---|
| Handwriting (26 classes) | 0.321±0.011 | 0.308±0.013 | **0.336±0.019** | 0.514 | 9.1× |
| NATOPS | **0.891±0.012** | 0.865±0.020 | 0.872±0.011 | 0.944 | 7.0× |
| Libras | **0.898±0.003** | 0.869±0.009 | 0.894±0.010 | 0.917 | 9.5× |

**It is signal, not search.** Sixteen times the candidate budget buys +1.5 points on
Handwriting against a ~20-point deficit, at nine times the fit cost. On NATOPS and Libras
×16 lands *below* the default (−1.9 and −0.4), and ×4 is worse than default on all three.
One dataset of three improves slightly; two get worse. The deficit is not a sampling
problem, and the budget knob is not merely exhausted — past a point it is harmful.

## Why more candidates can hurt

This is not noise, and the project has already met the effect once. From the Phase B
write-up on the feature bank: *"Offering all of it at every node made the model markedly
worse, because every extra candidate is another chance for noise to win the best-gain
contest."*

That is the winner's curse, and it is structural. A node picks the single highest-gain
split from a pool of randomly drawn candidates. Enlarging the pool raises the expected gain
of the winner **whether or not any candidate is genuinely informative**, so the selected
split is increasingly over-fitted to the node's own rows. Heartwood cannot buy accuracy
with candidates, by construction.

MiniROCKET does not have this problem because it does not select. It computes ~10,000
fixed dilated-kernel features and hands *all of them* to a ridge regression, which shrinks
them jointly. Global L2 shrinkage over a large fixed bank behaves completely differently
from greedy max-gain selection over a random sample of one.

## What this rules out, and what it points at

Ruled out: bigger budgets, and — more importantly — the standing assumption that better
*targeting* of the same greedy draw is the open problem. Targeting improves the hit rate
of a sampler, but the ×16 arm already raises the hit rate substantially and still does not
move the deficit. The bottleneck is the selection rule, not the sampler feeding it.

What it points at is already half-built in this repository. `dense_base` boosts trees from
a ridge over ~490 window statistics, and Phase C measured it as a genuine trade: +5.2
points on `slope_window`, 10% worse error on `amp_regression`, because a ridge only helps
when the temporal signal has marginal structure. The natural V6 is to keep that
architecture and change what the ridge sees — a bank of dilated convolutions rather than
window statistics — so the shrinkage estimator handles the shape features and the trees
handle statics and interactions, which is the half MiniROCKET has no answer for at all.

That is a hypothesis, not a result. It would need pre-registering as V6, with the same
thresholds and the same rule that a change earns its default or does not ship. It is
recorded here because it is what the evidence points at, and because the direction the
README has assumed for three milestones is now measurably wrong.
