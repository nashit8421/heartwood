# Validation Plan V11 — a base that can express an interaction

**Status: PRE-REGISTERED.** Written and committed before the change was implemented or run.
Eighth study.

---

## 1. The measured problem

V10 put the static block into the ridge base and won on Apnea-ECG, but cost 2–3 points on
the `static_control` synthetic scenario. That scenario is the one where the series is pure
noise and *all* the signal is static, so it is exactly where a static-aware base should
shine. It was recorded as MARGINAL and left as the first thing to look at.

Measured, at n=500 over 3 seeds:

| | accuracy |
|---|---|
| default (trees alone) | **0.936** |
| base without statics | 0.936 — the base declines, the series being noise |
| base + statics (V10) | 0.908 |
| the base by itself | 0.895 |

The base is not wrong, it is a **ceiling**. Its leave-one-out R² is 0.53, it reaches 0.895
alone, and the trees lift it only to 0.908 — below the 0.936 they reach from a standing
start.

The cause is in the label: `1.5·x0 − 1.2·x1 + 1.0·x0·x2`. Two linear terms and **a product**.
A linear base captures the first two confidently and cannot express the third at all, and
boosting from a confident partial fit leaves the trees less room than boosting from nothing.

## 2. The change

Give the unpenalised block a **non-linear basis**: the static columns *and their pairwise
products*. The interaction that defeats a linear base is then literally a column in it.

Two things keep this from being a licence to overfit:

* **A size guard.** Products are included only when `1 + p + p(p−1)/2 ≤ n/4`, so the
  unpenalised block never approaches the row count. Otherwise the block stays linear. This
  is a rule on shapes, fixed here, with no reference to any score.
* **The existing guards already do the rest.** If the expanded base overfits, its
  leave-one-out R² falls, it fails the permutation null added in V8, and the base is
  declined outright — returning the model to its default behaviour. The failure mode is
  *safe*: worst case we get the trees alone, which is exactly the 0.936 that V10 lost.

Exposed as `dense_static_interactions: bool`, defaulting on where the size guard permits,
since V10's static block already ships on inside the opt-in ridge base.

## 3. Hypotheses — frozen, with numeric pass/fail

**H-V11.1 — the regression is repaired.** On `static_control`, every cell is within **1
point** of the current default. It is currently −2.0 to −3.1.
*Fails if* any cell remains worse than −2.

**H-V11.2 — Apnea is not sacrificed.** On Apnea-ECG at n=1000, 5 seeds, AUC is within **1
point** of V10's 0.856, and still beats `static_only` (0.835) by ≥2.
*Fails if* it drops more than 2 points or stops beating `static_only`.

**H-V11.3 — nothing else breaks.** No synthetic cell worse than the current default by more
than 2 points; CPSC-2018 and Sleep-EDF within 1 point of V10.
*Fails if* any synthetic cell drops more than 5, or either real dataset by more than 2.

**H-V11.4 — earns its default.** Ships on only if H-V11.1 and H-V11.2 pass and H-V11.3 does
not fail.

## 4. Rules and scope

`VALIDATION.md` §2 in force. The size guard is a shape rule fixed in advance and is not
tuned; no threshold in this change is chosen by looking at a score.

Runs: the full synthetic grid (H-V11.1 and most of H-V11.3), Apnea-ECG at n=1000 with 5
seeds (H-V11.2), CPSC-2018 and Sleep-EDF at n=1000 with 3 seeds (H-V11.3).

## 5. What counts as done

A base that can express a product is a strictly larger hypothesis class reached without
adding any selection, which is the property that made the convolution base work in the
first place. If H-V11.1 passes, the last known regression from V10 is gone. If H-V11.2
fails, the interaction terms cost more on real data than they buy on synthetic, and the
change ships off by default with the numbers recorded.
