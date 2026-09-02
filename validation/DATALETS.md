# Would `datalets` improve Heartwood?

**EXPLORATORY, NOT PRE-REGISTERED.** Same status as `HEADROOM.md`: this is a scouting
measurement to decide whether a full study is worth pre-registering. Nothing here may become
a headline number, and its protocol deliberately differs from the validation suite's.

## What datalets is

[`datalets`](https://pypi.org/project/datalets/) implements *partition-then-fit*: cluster the
data into subpopulations unsupervised, run a model tournament inside each cluster, and route
each prediction to the specialist for its cluster. It is scikit-learn compatible, takes a 2-D
`X`, and accepts `groups` — which matters here, because this project has shipped the
row-wise/group-wise confusion three times.

It also ships `honest_cv`, which scores a configuration against baselines on identical folds,
including a **global tournament with no clustering**. That baseline is the right one: it
isolates the contribution of *partitioning* from the contribution of *model selection*.

## The measurement

Heartwood's own feature representation is the MiniROCKET kernel bank, so that is what was
handed to datalets. 3-fold CV, `candidates="fast"`, accuracy.

| dataset | n | partitioned | global tournament | plain HGB |
|---|---|---|---|---|
| NATOPS | 360 | 0.911 | **0.928** | 0.867 |
| Handwriting | 1000 | 0.686 | 0.686 | 0.560 |

**On NATOPS the partition costs 1.7 points** against the same tournament without clustering.
It is not a degenerate fit: four real clusters formed, of sizes 62, 180, 61 and 57.

**On Handwriting the two are identical to six decimal places**, which means the partition
collapsed to a single cluster — 26 classes against `min_class_count=8` leaves no valid split.

So on both, the partition contributes nothing or less than nothing. What *does* contribute is
the tournament: model selection beat a plain histogram booster by 6.1 and 12.6 points.

**These numbers are not comparable to `RESULTS_V24.md`.** That suite uses official train/test
splits and balanced accuracy; this uses pooled 3-fold CV and accuracy. They answer different
questions and must not be placed in the same table.

## Why this was the expected outcome

Three reasons, each already measured in this repository.

**1. Partitioning spends the resource this library is shortest of.** Everything here is
designed and measured between n=100 and n=2000, and the README's known limitations record
that n=100 is still the frontier. Splitting 1000 rows into 8 clusters gives 125 rows per
specialist — squarely inside the regime where this model is weakest. Partitioning trades
global sample size for local specialisation, and there is no global sample size to spare.

**2. Model selection at this scale is a coin flip, and V20 measured it.** V20 fit three
models, judged them on a held-out fold and kept the winner. Across eight datasets the
selection landed on the better model in **15 of 31 cells — 48%** — and cost 1.6 points
against not selecting at all. Datalets runs a tournament *per cluster*, on a fraction of the
rows V20's selection had. The direction of that evidence is not ambiguous.

**3. On this project's datasets the natural clusters are probably subjects.** Physiological
recordings cluster by patient long before they cluster by pathology, and a specialist trained
on one set of subjects has no claim on an unseen one. Every benchmark here that has groups is
split subject-disjoint, so any gain from subject-shaped clusters would evaporate exactly where
it is measured. `RESULTS_SCREEN.md` records what happened the last time this project forgot
that: a row-wise split read Apnea's exogeneity as 0.82 instead of 0.25.

## What is worth taking from it anyway

**`honest_cv` is this project's own method, packaged.** Identical folds, explicit baselines,
group support, and a comparison designed to reveal that the added machinery did not help. That
is a methodology fit even though it is not a model improvement.

**The uncomfortable observation.** A tournament of stock scikit-learn models over the
MiniROCKET bank beat a plain histogram booster by 6 and 13 points here. Heartwood is also a
model over that bank, and `RESULTS_V24.md` has it behind MiniROCKET on 11 of 16 UEA datasets.
"What is the best model to put on top of this bank" is a live question that this project has
never actually run a tournament on, and the answer might not be trees.

## Verdict

**Not worth pre-registering as an improvement to Heartwood.** The partition mechanism is
measured negative on the one feature representation it would have to work on, the sample sizes
are wrong for it by construction, and the closest analogue this project has already run (V20)
failed at 48%.

The one integration that would be architecturally novel — cluster on the *static* block, fit a
Heartwood per cluster, so the statics route rather than predict — is blocked by the same thing
everything else is blocked by: no dataset where the statics are strong, exogenous and the
regime is temporal. `RESULTS_SCREEN.md` rejects all eight tried so far. A new fusion mechanism
has nothing to prove itself on until that changes.
