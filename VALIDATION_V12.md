# Validation Plan V12 — make the base's self-check match the real test

**Status: PRE-REGISTERED.** Written and committed before the change was implemented or run.
Ninth study.

---

## 1. What V11 actually exposed

V11 put products of the static columns into the ridge base. It half-repaired the
`static_control` regression (−3.1 → −1.5) and was ruinous elsewhere: `amp_regression` fell
13.4 points, and Apnea-ECG went from **0.856 AUC to 0.478 — below chance**.

The mechanism is extrapolation. Products grow quadratically, so a held-out subject whose
body weight sits outside the training range gets an exploding term.

**The important part is that the guard could not see it.** I argued the change was safe
because an overfitted base would show a poor leave-one-out R², fail V8's permutation null,
and be declined. That argument was wrong, and its wrongness is more useful than the feature
was:

> Leave-one-out hides **one row**. Apnea-ECG is split by **subject**. So the base validated
> itself on other minutes from patients it had already seen, and never once encountered a
> body weight it had not been trained on. It was measuring interpolation while the
> benchmark measured extrapolation.

That blind spot is not specific to interaction terms. It applies to **every** judgement the
base makes on grouped data — including the V10 result this project currently leads with,
which was validated the same blind way. Fixing the instrument comes before trusting it
again.

## 2. The changes

**A — the base leaves out groups, not rows.** When the caller supplies `groups`, the base's
honesty check hides whole groups. This stays closed-form: for a group `G`, the
leave-group-out prediction is `(I − H_GG)⁻¹ (fitted_G − H_GG y_G)`, where `H_GG` is that
group's block of the hat matrix. No refitting, one small solve per group. Leave-one-out is
the special case where every group has one row, so the existing path is unchanged when no
groups are given.

**B — interactions on a bounded scale.** With A in place, re-test the V11 idea using
products of **rank-transformed** statics rather than raw ones. `features.ecdf` already maps
a value into [0, 1] against a frozen training distribution — the same mechanism comparison
splits use — so a body weight beyond anything seen in training saturates at 1 instead of
exploding. Products of bounded terms are bounded, by construction rather than by hope.

## 3. Hypotheses — frozen, with numeric pass/fail

**H-V12.1 — the guard can now see the failure.** With interactions on *and* group-aware
validation, Apnea-ECG at n=1000 scores **≥ 0.84** AUC.
*Fails if* below 0.80. V11 scored 0.478 here. This tests whether the instrument now detects
what it previously missed — either by declining the base or by the bounded terms not
exploding.

**H-V12.2 — honesty costs nothing when nothing is wrong.** With interactions **off**,
group-aware validation leaves Apnea-ECG within **1 point** of V10's 0.856 and still beating
`static_only` (0.835) by ≥2.
*Fails if* it drops more than 2 points. If a more honest check destroys the V10 result, then
the V10 result was partly an artefact of the blind one, and that is the finding.

**H-V12.3 — the original target.** `static_control` is within **1 point** of the plain
default at every size. It is −2.0 to −3.1 under V10 and −0.7 to −1.5 under V11.
*Fails if* any cell is worse than −2.

**H-V12.4 — no regression.** No synthetic cell worse than the default by more than 2 points;
CPSC-2018 and Sleep-EDF within 1 point of V10.
*Fails if* any synthetic cell drops more than 5, or either real dataset by more than 2.

**H-V12.5 — earns its default.** Group-aware validation (A) ships on whenever groups are
supplied, unless H-V12.2 fails. Bounded interactions (B) ship on only if H-V12.1 and
H-V12.3 pass and H-V12.4 does not fail.

## 4. Rules and scope

`VALIDATION.md` §2 in force. No threshold here is chosen by looking at a score.

The block leave-group-out formula is checked against **literally refitting without each
group**, in a test, before any result is read. This is the third time this project has
touched the leave-one-out machinery and the second time a subtle error in it produced a
confident wrong answer, so it is checked rather than reasoned about.

Runs: the full synthetic grid; Apnea-ECG at n=1000 with 5 seeds in three configurations
(interactions off, interactions on, and V10 for reference); CPSC-2018 and Sleep-EDF at
n=1000 with 3 seeds.

## 5. What counts as done

If H-V12.2 fails, the honest conclusion is that V10's headline was measured with an
instrument that could not see the thing that matters, and the README says so. That is the
outcome I would least like and the one this plan exists to make findable.
