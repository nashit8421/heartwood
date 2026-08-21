# V6 results — PARTIAL, run stopped early

**Coverage, stated first.** The grid was cut short. Complete: 6 UEA datasets at 5 seeds.
Partial: SelfRegulationSCP2 (2 of 5 seeds), PTB-XL (seed 0 only, 3 sizes). Not started:
Heartbeat, and PTB-XL seeds 1–4. Nothing below should be read as a final verdict, and
H-V6.1/2/3 are **not** adjudicated — `report_v5.py`-style mechanical scoring waits for a
complete grid.

`heartwood_rocket` is `dense_base=True, dense_features="rocket"`. MiniROCKET is `aeon`,
credited with the better of 2,000 and 10,000 kernels.

| dataset | n | hw+rocket | best MiniROCKET | gap | seeds |
|---|---|---|---|---|---|
| ptbxl | 100 | 0.421 | 0.434 | −1.3 | 1 |
| ptbxl | 250 | 0.427 | 0.414 | +1.3 | 1 |
| ptbxl | 500 | 0.490 | 0.479 | +1.1 | 1 |
| Epilepsy | 137 | 0.995 | 1.000 | −0.5 | 5 |
| HandMovementDirection | 160 | **0.459** | 0.387 | **+7.2** | 5 |
| Handwriting | 150 | 0.520 | 0.514 | +0.7 | 5 |
| Libras | 180 | 0.916 | 0.917 | −0.1 | 5 |
| NATOPS | 180 | 0.924 | 0.944 | −2.0 | 5 |
| RacketSports | 151 | **0.892** | 0.866 | **+2.5** | 5 |
| SelfRegulationSCP2 | 200 | 0.422 | 0.556 | −13.3 | 2 |

Beat MiniROCKET by ≥2 points on 2 of 10 cells; lost by ≥2 on 1. **Median gap +0.3.**

## What this does and does not show

**The base works.** Against the shipped default, the rocket base is a large gain wherever
the shape regime bites: Handwriting 0.309 → 0.520 (+21), HandMovementDirection 0.398 →
0.459 (+6), PTB-XL at n=100 0.332 → 0.421 (+9). V5 measured Heartwood at mean rank 2.33
against MiniROCKET's 1.50; on this partial grid the two are at parity. That is the
diagnosis in `HEADROOM.md` being right about the mechanism — selection was the ceiling, and
not selecting removes it.

**Parity is not a win.** H-V6.2 asks for ≥2 points at a majority of sizes, and a median of
+0.3 does not get there. Seed 0 alone looked much better (Handwriting +3.1, RacketSports
+4.3) and five seeds pulled it back to +0.7 and +2.5 — the same lesson as the v0.3 seed
bug, arriving from the other direction. Single-seed reads on this project have been
misleading three times now.

**One result needs a fix, not a report.** SelfRegulationSCP2 puts `heartwood_rocket` at
0.422 on a *binary* task — below the 0.500 chance line, and below the plain default's
0.528. A base that lands below chance is a bug, most likely the leave-one-out ridge
overfitting at 10,000 features against 200 rows and the trees then boosting from a base
that is worse than useless. `DenseBase` already refuses penalties that interpolate; that
guard is evidently not sufficient in this regime. This is the first thing to chase.

## Next, in order

1. Fix the SelfRegulationSCP2 regression — a below-chance base is a correctness bug.
2. Finish the grid: Heartbeat, SelfRegulationSCP2 seeds 2–4, PTB-XL seeds 1–4.
3. Run H-V6.3, the synthetic no-regression check, before `rocket` is considered for default.
4. Only then adjudicate H-V6.1/2/3 mechanically.
