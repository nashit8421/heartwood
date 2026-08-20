"""Brute-force cross-checks of features.py and splits.py.

Runnable with plain python (no pytest required):

    python tests/smoke/smoke_core.py

Every check here compares the vectorised implementation against a slow, obviously
correct reference — including the NaN cases that are the library's main
silent-corruption risk.  Milestone M2 formalises these as a pytest suite.
"""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heartwood.features import STAT_NAMES, interval_stat, shapelet_features
from heartwood.splits import scan_threshold

rng = np.random.default_rng(0)
fail = 0


def check(name, ok, extra=""):
    global fail
    if not ok:
        fail += 1
        print(f"FAIL {name} {extra}")
    else:
        print(f"ok   {name}")


# ---------------------------------------------------------------- interval stats
def brute_stat(row, stat):
    v = row[np.isfinite(row)]
    if stat == "mean":
        return v.mean() if v.size else np.nan
    if stat == "std":
        return v.std() if v.size else np.nan
    if stat == "min":
        return v.min() if v.size else np.nan
    if stat == "max":
        return v.max() if v.size else np.nan
    if stat == "median":
        return np.median(v) if v.size else np.nan
    if stat == "slope":
        idx = np.nonzero(np.isfinite(row))[0].astype(float)
        if idx.size < 2:
            return np.nan
        den = idx.size * (idx**2).sum() - idx.sum() ** 2
        if abs(den) <= 1e-12:
            return np.nan
        return (idx.size * (idx * v).sum() - idx.sum() * v.sum()) / den
    if stat == "mean_abs_change":
        d = np.abs(np.diff(row))
        d = d[np.isfinite(d)]
        return d.mean() if d.size else np.nan
    if stat == "last":
        return v[-1] if v.size else np.nan
    if stat == "delta":
        return v[-1] - v[0] if v.size else np.nan
    raise AssertionError(stat)


X = rng.normal(size=(60, 25))
X[rng.random(X.shape) < 0.25] = np.nan
X[3, :] = np.nan          # all-NaN row
X[4, :] = 7.0             # constant row
X[5, :] = np.nan; X[5, 6] = 1.0   # single valid point
X[6, :] = np.arange(25) * 0.5 + 2.0  # exact line

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for stat in STAT_NAMES:
        got = interval_stat(X, stat)
        want = np.array([brute_stat(r, stat) for r in X])
        check(f"interval_stat[{stat}]",
              np.allclose(got, want, equal_nan=True, rtol=1e-9, atol=1e-12),
              f"\n got={got[:8]}\n want={want[:8]}")
check("no warnings from interval_stat", len(caught) == 0, [str(w.message) for w in caught])
check("slope of exact line", abs(interval_stat(X, "slope")[6] - 0.5) < 1e-9)
check("slope of constant is 0", abs(interval_stat(X, "slope")[4]) < 1e-12)
check("all-NaN row -> NaN everywhere",
      all(np.isnan(interval_stat(X, s)[3]) for s in STAT_NAMES))


# ------------------------------------------------------------------- shapelets
def brute_shapelet(X2d, shp, znorm=True):
    m, T = X2d.shape
    l = len(shp)
    P = T - l + 1
    if znorm:
        sd = shp.std()
        s = (shp - shp.mean()) / sd if sd > 1e-12 else np.zeros_like(shp)
    else:
        s = shp
    dist = np.full(m, np.nan)
    pos = np.full(m, np.nan)
    for i in range(m):
        best, bestj = np.inf, -1
        for p in range(P):
            w = X2d[i, p:p + l]
            if not np.isfinite(w).all():
                continue
            if znorm:
                sd = w.std()
                wn = (w - w.mean()) / sd if sd > 1e-12 else np.zeros_like(w)
            else:
                wn = w
            d = np.mean((wn - s) ** 2)
            if d < best:
                best, bestj = d, p
        if bestj >= 0:
            dist[i] = best
            pos[i] = bestj / (P - 1) if P > 1 else 0.0
    return dist, pos


Y = rng.normal(size=(30, 40))
Y[rng.random(Y.shape) < 0.10] = np.nan
Y[0, :] = np.nan
Y[1, 10:20] = 5.0  # constant stretch
shp = rng.normal(size=7)
for zn in (True, False):
    gd, gp = shapelet_features(Y, shp, znorm=zn)
    wd, wp = brute_shapelet(Y, shp, znorm=zn)
    check(f"shapelet dist (znorm={zn})", np.allclose(gd, wd, equal_nan=True, atol=1e-9),
          f"\n got={gd[:6]}\n want={wd[:6]}")
    check(f"shapelet pos  (znorm={zn})", np.allclose(gp, wp, equal_nan=True, atol=1e-9))

# planted shapelet: exact match -> dist ~ 0 at the plant position
Z = rng.normal(size=(5, 50))
tpl = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
Z[2, 12:19] = tpl * 3.0 + 4.0        # scaled/shifted copy: z-norm should still match
d, p = shapelet_features(Z, tpl)
check("planted shapelet dist ~ 0", d[2] < 1e-9, d[2])
check("planted shapelet pos correct", abs(p[2] - 12 / (50 - 7)) < 1e-9, p[2])

# NaN windows must never win
W = np.full((3, 30), np.nan)
W[0, 5:12] = tpl        # only one usable window
d, p = shapelet_features(W, tpl)
check("NaN-only row -> NaN", np.isnan(d[1]) and np.isnan(p[1]))
check("only-usable-window found", abs(d[0]) < 1e-9 and abs(p[0] - 5 / 23) < 1e-9, (d[0], p[0]))

# chunking equivalence
d1, p1 = shapelet_features(Y, shp, chunk_bytes=64 << 20)
d2, p2 = shapelet_features(Y, shp, chunk_bytes=1)
check("chunked == unchunked",
      np.allclose(d1, d2, equal_nan=True) and np.allclose(p1, p2, equal_nan=True))


# ---------------------------------------------------------------- scan_threshold
def brute_scan(f, g, h, lam, gamma, mcw, msl):
    n = f.size
    finite = np.isfinite(f)
    G, H = g.sum(), h.sum()
    parent = G * G / (H + lam)
    best = None
    cands = np.unique(f[finite])
    for ml in (True, False):
        for i in range(len(cands) - 1):
            thr = 0.5 * (cands[i] + cands[i + 1])
            if thr >= cands[i + 1]:
                thr = cands[i]
            left = np.where(finite, f <= thr, ml)
            nL, nR = left.sum(), n - left.sum()
            GL, HL = g[left].sum(), h[left].sum()
            GR, HR = G - GL, H - HL
            if HL < mcw or HR < mcw or nL < msl or nR < msl:
                continue
            gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - parent) - gamma
            if best is None or gain > best[0] + 1e-15:
                best = (gain, thr, ml)
    if best is None or best[0] <= 1e-12:
        return None
    return best


for trial in range(300):
    n = int(rng.integers(6, 40))
    f = rng.normal(size=n).round(1)          # forces duplicate values
    if rng.random() < 0.6:
        f[rng.random(n) < 0.25] = np.nan
    g = rng.normal(size=n)
    h = rng.uniform(0.05, 1.5, size=n)
    lam, gamma = float(rng.choice([0.0, 1.0, 5.0])), float(rng.choice([0.0, 0.05]))
    mcw, msl = float(rng.choice([0.0, 0.1])), int(rng.integers(1, 4))
    got = scan_threshold(f, g, h, lam, gamma, mcw, msl)
    want = brute_scan(f, g, h, lam, gamma, mcw, msl)
    if (got is None) != (want is None):
        check(f"scan trial {trial} None-agreement", False, f"got={got} want={want}")
        break
    if got is not None and not np.isclose(got[0], want[0], rtol=1e-9, atol=1e-12):
        check(f"scan trial {trial} gain", False, f"got={got} want={want}\nf={f}")
        break
else:
    check("scan_threshold vs brute force (300 random trials)", True)

const = np.ones(20)
check("constant feature -> None",
      scan_threshold(const, rng.normal(size=20), np.ones(20), 1.0, 0.0, 1e-3, 5) is None)

print("\nFAILURES:", fail)
sys.exit(1 if fail else 0)
