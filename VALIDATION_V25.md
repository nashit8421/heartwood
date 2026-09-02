# V25 — what is the best model to put on top of the bank?

Written and committed **before any V25 cell is run.**

## 1. Why

This library is, after V15 and V23 deleted everything else, *MiniROCKET's kernel bank with
gradient-boosted trees on top.* MiniROCKET's own answer to what goes on top is a
cross-validated ridge. `RESULTS_V24.md` has the ridge ahead of our trees on **11 of 16** UEA
datasets.

**Nobody has ever asked whether either is the right choice.** The trees were inherited from
the project's origin as a gradient booster; the ridge was inherited from MiniROCKET's paper.
Neither was selected by comparison against alternatives on this bank.

A scouting measurement in `validation/DATALETS.md` made the question concrete: a tournament of
stock scikit-learn models over the bank beat a plain histogram booster by 6 and 13 points.
That was pooled 3-fold CV accuracy and proves nothing on its own. This study asks the same
question at the protocol the rest of the suite uses.

## 2. Design

**One bank, many heads.** Every arm gets the identical MiniROCKET transform — aeon's
`MiniRocket(n_kernels=10000, random_state=0)` followed by `StandardScaler(with_mean=False)`,
which is exactly the pipeline behind the `minirocket10k` baseline. Only the final estimator
changes. That isolates the head, which is the whole question.

**Heads, fixed now and not added to afterwards:**

| arm | head |
|---|---|
| `ridge` | `RidgeClassifierCV` — MiniROCKET's own choice, and the control |
| `logreg` | multinomial logistic regression |
| `linsvc` | linear SVM |
| `hgb` | histogram gradient boosting |
| `rf` | random forest |
| `extratrees` | extremely randomised trees |
| `knn` | k-nearest neighbours |
| `mlp` | a small multilayer perceptron |

**Suite:** the same sixteen UEA datasets as V24, official splits, 5 seeds, balanced accuracy.
Identical to V24 by construction, so **`RESULTS_V24.md`'s Heartwood column is directly
comparable** and does not need re-running.

**The winner is judged on one head applied to every dataset**, never the per-dataset best.
V16 established why: a sweep reported at its per-dataset best measures its own search, and the
gap between the two is tuning optimism. The report prints both and the gap.

## 3. Hypotheses

* **H-V25.1 — the protocol is identical (control).** The `ridge` arm must reproduce V24's
  `minirocket10k` to within **0.5 points on every dataset**. It is the same pipeline on the
  same splits, so anything larger means the two studies are not comparable and **nothing else
  here is interpretable.** This is checked first.
* **H-V25.2 — is the ridge the right head?** PASS if some head beats `ridge` by **≥ +1.0
  points on ≥ 9 of 16** datasets, same head throughout.
* **H-V25.3 — are the trees the right head?** PASS if some head beats V24's Heartwood by the
  same bar. Heartwood is a *different* model, not a head on this bank — it also draws
  interval, shapelet and filter candidates and can split on statics — so this compares
  architectures, not heads, and is reported as such.

## 4. What each outcome means

* **A head beats the ridge.** MiniROCKET's standard pipeline is leaving points on the table,
  which is a result about MiniROCKET rather than about this library, and worth reporting as
  one.
* **Nothing beats the ridge.** The ridge is the right head, which is a real finding: it says
  the bank's value is in the features and a shrinkage estimator is all that is needed on top —
  and that the trees in this library are a worse use of it on general benchmarks, exactly as
  V24 measured.
* **A head beats Heartwood.** Then the honest recommendation for general time-series data is
  that head rather than this library, and the README says so. This project has withdrawn three
  claims already; a fourth is not a catastrophe.
* **Heartwood beats every head.** The trees earn their place on the bank after all, and V24's
  UEA deficit is about something other than the choice of head.

## 5. The outcome I would least like

`hgb` — a stock histogram booster — beating both the ridge *and* Heartwood.

It would mean that a plain gradient booster over the bank, which any user could write in four
lines, outperforms the library this project has spent two weeks validating, and that the
temporal split machinery is not merely unhelpful on these datasets but strictly worse than
doing nothing special at all. I would rather know. `RESULTS_V24.md` already says the trees
lose to a ridge on 11 of 16; this would say they lose to the *simplest possible* alternative
too, and the README would have to carry that.

## 6. Pre-committed analysis

Balanced accuracy, paired within seed, per-seed vectors always reported. No arm, seed or
dataset dropped; a failed cell is reported as failed. The control (H-V25.1) is evaluated and
printed **before** any comparison, and if it fails the rest is reported as uninterpretable.

**On the winner's curse at study level:** eight heads are compared, so the best of them is
optimistically biased even under a null. That is why H-V25.2 requires a **margin** (+1.0)
rather than merely being ahead, why the head must be the same on every dataset, and why mean
rank across datasets is reported alongside mean score — a head that wins on rank *and* on mean
is a different claim from one that wins on mean alone because of a single large margin.
