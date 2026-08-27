# V8 results — and a correction that invalidates V8's premise

## The correction first

V7 reported that trees boosted onto the convolution base, with no static block,
**cost 4.6 points on CPSC and 5.7 on Sleep-EDF**, and that the static block was
then worth +8.3 and +6.9. That was wrong, and it was a bug in
`validation/report_v7.py`, not in the experiment.

The no-static results file contains four models — `heartwood_rocket`, `agg`,
`minirocket`, `minirocket10k` — and the reporting code averaged **all of them**
into "arm C" instead of filtering to Heartwood. Averaging a model with three
weaker baselines drags it down by about five points, which is exactly the
"tax" that appeared. `--drop-static` itself worked correctly throughout.

Corrected, verified per-seed and paired (arms C and D share seeds and splits):

| arm | CPSC | Sleep-EDF | |
|---|---|---|---|
| A — `aeon` MiniROCKET | 0.654 | 0.660 | the reference |
| B — ridge over our bank | 0.654 | 0.661 | bank quality **+0.0 / +0.1** |
| C — + trees, no statics | 0.692 | 0.673 | **trees add +3.8 / +1.2** |
| D — + statics | 0.691 | 0.673 | **statics add −0.1 / +0.0** |

Paired per-seed D−C: CPSC `+0.0, −0.4, +0.2`; Sleep-EDF `−0.1, −0.2, +0.3`.

**The conclusion inverts.** The trees are not a liability — they are the entire
contribution, worth +3.8 and +1.2 over a base that already matches MiniROCKET.
And the static block, which is this library's founding premise, contributes
**nothing** on either dataset.

This also removes the puzzle V7 invented. Heartwood beats MiniROCKET on
Handwriting and HandMovementDirection, which have no static block at all; under
the corrected reading that is not anomalous, it is the same mechanism.

**H-V7.3 therefore FAILS**, where it was reported as a pass.

## What that does to V8

V8 was pre-registered to remove a five-point tax. **The tax did not exist.** The
work still ran and the results still count, but the hypothesis it was built on
was an artefact.

**H-V8.1 — vacuous.** Arm C is within 2 points of arm B under the chance floor
(CPSC +2.8, Sleep-EDF +1.7). It already was, by +3.8 and +1.2, before the change.
Nothing was recovered because nothing was lost.

**H-V8.2 — FAIL.** The margin over the better MiniROCKET at n=1000 moves by
**−0.2 (CPSC)** and **+0.3 (Sleep-EDF)**. Neither improves by the 2 points
required. The chance floor also made trees-without-statics slightly *worse* on
CPSC (0.692 → 0.682).

**H-V8.3 — FAIL.** Worst synthetic cell −11.2 (slope_window n=250), against a
−5 fail bar. The two large swings, +10.3 on bump_order n=100 and −11.3 on
slope_window n=250, land on the two highest-variance cells in the grid, so they
are probably three-seed noise in both directions — but the rule is the rule.

**H-V8.4 — does not ship.** `selection_null` stays off by default, opt-in, with
these numbers recorded.

## What is actually true after V8

* The convolution bank matches `aeon` MiniROCKET exactly (+0.0 / +0.1). The
  reimplementation is not where any edge lives.
* **Trees over that base are the edge**, worth +3.8 and +1.2 — nonlinearity the
  ridge cannot express.
* **The static block is worth nothing measurable** on the two datasets where it
  was isolated, despite being the reason this library exists.
* Pricing a node's selection bias does not convert into accuracy here.

The last two are the uncomfortable ones and they are the ones to act on.
