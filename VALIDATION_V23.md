# V23 — do comparison splits and Lévy areas work anywhere real?

Written and committed **before any V23 cell is run.**

## 1. Why this study exists

V15 measured four extras this library adds to MiniROCKET's bank and failed all four on eight
UEA datasets. `VALIDATION_V15.md` §4 pre-committed to deleting a failed extra.

Before deleting, the four were re-tested on the five synthetic generators — which V15 never ran
— and two of them turned out to be load-bearing:

| removing | effect |
|---|---|
| comparison splits | −9.9 `bump_order`, −8.7 `lead_lag`, −1.2 `timing` |
| Lévy areas | −10.6 `lead_lag` |
| virtual channels | nothing anywhere |
| window statistics | nothing anywhere (0.015% RMSE) |

The two inert ones were deleted. **These two were kept, against the pre-registration**, and
that decision is what this study exists to test rather than leave as a judgement call.

## 2. The problem with both existing pieces of evidence

**The synthetic evidence is close to tautological.** `bump_order` is an XOR of *which transient
came first*; a comparison split is literally the operator that question needs. `lead_lag` is
*which channel led*; a Lévy area is the signed quantity that answers it. Finding these features
useful on scenarios written to require them says they are implemented correctly. It does not
say they are useful.

**The UEA evidence is real but narrow.** Eight datasets, chosen for an earlier study, none
selected for having ordering or lead–lag structure.

So neither result settles it, and the honest question is the one neither asked: **is there a
real dataset — not one written to need them — where these features earn their place?**

## 3. Design

**Suite: eight UEA datasets that V15 did not use.** Selected by a rule fixed before any score:
between 3 and 64 channels, loads from the local cache, and not in V15's suite. That yields
ArticularyWordRecognition, BasicMotions, CharacterTrajectories, Cricket, ERing,
EthanolConcentration, StandWalkJump, UWaveGestureLibrary. Every one is multichannel, so both
extras are live on all eight.

The rule matters more than the list. Having just watched both extras fail one suite and pass
another, picking datasets by hand now would be suite-shopping, and the result would be worth
nothing whichever way it fell. AtrialFibrillation (2 channels) and DuckDuckGeese (1345) are
excluded by the rule, not by preference.

**Arms.** `v23_base` is the current library with both extras off; `v23_cmp` and `v23_levy` add
exactly one back; `v23_both` adds both. Official splits, 5 seeds.

**A positive control, run in the same study.** The same four arms run on the five synthetic
scenarios. This is not evidence for the extras — §2 explains why — it is evidence about the
*study*: if the arms cannot reproduce the −9.9 and −10.6 effects on scenarios built for them,
then the arms are broken or the design lacks power, and a null result on the real suite would
mean nothing.

## 4. Hypotheses

* **H-V23.1 — comparison splits earn their place.** `margin ≥ +0.5` over `v23_base` on **≥ 5 of
  8** datasets. Same bar as V15, deliberately: changing the bar between suites is how a
  failed result gets rescued.
* **H-V23.2 — Lévy areas earn their place.** Same bar.
* **H-V23.3 — the study can detect these effects.** On the synthetic scenarios, `v23_cmp` must
  beat `v23_base` by ≥ 5 points on `bump_order` and `v23_levy` by ≥ 5 points on `lead_lag`.
  **If this fails, H-V23.1 and H-V23.2 are uninterpretable and are reported as such** rather
  than as failures.

## 5. What each outcome means

* **Either extra passes.** It stays, with a proper justification for the first time: it now
  works on a real suite that was not chosen to flatter it, and the V15 verdict is understood as
  suite-specific rather than wrong.
* **Both fail, control passes.** This is the outcome I expect. Sixteen real datasets across two
  independent suites, no effect on either, against two scenarios written to require these
  operators. The honest reading is then that **both features work only on tasks constructed to
  need them**, and they should be deleted after all — completing what
  `VALIDATION_V15.md` §4 instructed and this project declined to do on one suite's evidence.
  The synthetic wins would stay in the record as what they are: proof the operators function,
  not proof they are worth carrying.
* **Both fail, control fails.** The study is void and the arms are debugged before anything is
  concluded.

## 6. The outcome I would least like

One extra clearing the bar on exactly five of eight with a mean near zero, after both were
already condemned once. That is a rescue, not a result, and the per-seed vectors are what would
expose it. If it happens the extra does not ship on this evidence; it goes to a third suite
named in advance — the physiological datasets at n=500 — before any claim is made.

## 7. Pre-committed analysis

Margins are paired within seed and reported with the full per-seed vector. No arm, seed or
dataset is dropped; a cell that fails to run is reported as failed. Bars are applied
mechanically by `validation/report_v23.py` from `V23_EXTRAS`. The synthetic control is reported
in the same document and is never counted toward H-V23.1 or H-V23.2.
