# Validation Plan V10 — let the linear layer see the static block

**Status: PRE-REGISTERED.** Written and committed before the change was implemented or run.
Seventh study.

---

## 1. The defect V9 found

Apnea-ECG is the first dataset in this project where the static block and the series are
both strong. On it, at n=1000:

```
    ECG alone (arm C)        0.807
    statics alone (arm S)    0.835
    both together (arm D)    0.827   <- worse than statics alone
```

Two informative, largely independent sources, and combining them yields **less than the
better one**. A competent combination should clear 0.87. That is a defect in this library.

The cause is structural and localised. `booster._augment` builds the ridge base from
`X_series` alone — the static block never reaches `_dense_bank`. So the strong linear
layer, the component worth +3.8 points over MiniROCKET in V7, **cannot see BMI at all**.
The statics enter only through greedy per-node tree splits, which `validation/HEADROOM.md`
measured as this architecture's ceiling. The model is made to learn its single best
predictor through the one mechanism it is worst at.

## 2. The change

Fit the ridge base on the static block **and** the convolution features together, with the
static block **unpenalised**.

Unpenalised matters. Simply concatenating five static columns onto ten thousand
convolution features leaves them sharing a single global penalty chosen to tame the ten
thousand; BMI would be shrunk as hard as an arbitrary kernel response and would drown. The
statics are few and known to carry signal, so they get ordinary least squares while the
convolution bank keeps its ridge — the standard partially-penalised formulation, via
Frisch–Waugh: residualise both target and bank on the statics, ridge the residuals, and add
the static fit back.

**The leave-one-out machinery must stay exact.** Its hat matrix becomes
`P_Z + M X (X'MX + λI)⁻¹ X' M`, so leverages pick up the static projection's diagonal. This
is where the ridge base nearly shipped broken once before (`README`, Phase C), so it is
enforced by a test that checks the closed form against literally refitting without each
row — not by inspection.

Exposed as `dense_include_static: bool = False`. Off until V10 earns it.

## 3. Hypotheses — frozen, with numeric pass/fail

**H-V10.1 — the combination beats its best half.** On Apnea-ECG at n=1000, the full model
beats `static_only` by **≥ 2 points** of AUC. `static_only` measured 0.835, so the target is
**≥ 0.855**.
*Fails if* it does not beat `static_only` at all. This is the hypothesis. Everything this
library claims rests on combining two sources better than either alone, and it has never
been demonstrated.

**H-V10.2 — the statics contribute.** Paired D−C at n=1000 is **≥ 4 points**, up from the
+2.0 V9 measured with the statics confined to tree splits.
*Fails if* it is below +2.0, i.e. no better than leaving them where they were.

**H-V10.3 — no regression.** On CPSC-2018 and Sleep-EDF at n=1000, within **1 point** of
the V7 configuration; on the six synthetic scenarios, no cell worse than the current
default by more than 2 points.
*Fails if* any real cell drops more than 2, or any synthetic cell drops more than 5.

**H-V10.4 — earns its default.** Ships on only if H-V10.1 passes and H-V10.3 does not fail.

## 4. Rules and scope

`VALIDATION.md` §2 in force throughout. No tuning; the unpenalised-static formulation has
no free parameter to tune, which is part of why it was chosen over rescaling the static
columns by some factor.

Runs:

* **Apnea-ECG, n=1000, arms C and D, 5 seeds.** V9 used 3 and saw a spread from 0.702 to
  0.899 across ~10 held-out subjects; three is too few and five is the most this dataset's
  35 subjects will honestly support.
* **CPSC-2018 and Sleep-EDF, n=1000, 3 seeds** — regression check only.
* **Full synthetic grid** — regression check only.

Anything not run is reported as not run.

## 5. What counts as done

If H-V10.1 passes, this library does the thing it was built to do — combine per-row static
facts with a raw series and beat either alone — demonstrated on a dataset chosen by a rule
written before it was seen. That would be the first time in seven studies.

If it fails, the conclusion is not that the idea is wrong but that this architecture cannot
express it, after two attempts at the mechanism (greedy splits in V9, a joint linear layer
here). At that point the README should stop describing it as a mixed static-and-series
method and describe it as what it has repeatedly measured as: a strong time-series
classifier.
