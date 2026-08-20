"""End-to-end smoke test of the estimators.

Runnable with plain python (no pytest required):

    python tests/smoke/smoke_model.py

Covers the mechanics (loss decreases, determinism, early stopping, all three
tasks, both None-modes, ragged input, fit/predict routing agreement, leaf
arithmetic) plus one signal-recovery check.  Milestone M2 formalises these as a
pytest suite.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heartwood import HeartwoodClassifier, HeartwoodRegressor
from heartwood.datasets import (
    make_bump_interaction, make_timing_task, make_slope_window,
    make_shape_amplitude_regression, make_pure_static, make_static_plus_noise_series,
)

fail = 0
def check(name, ok, extra=""):
    global fail
    if not ok:
        fail += 1
        print(f"FAIL {name} {extra}")
    else:
        print(f"ok   {name} {extra}")

# ------------------------------------------------------- binary + timing
t0 = time.time()
Xs, Xt, y = make_bump_interaction(n=300, seed=0)
Xs_te, Xt_te, y_te = make_bump_interaction(n=1000, seed=1)
m = HeartwoodClassifier(n_estimators=60, random_state=0)
m.fit(Xs, Xt, y)
fit_s = time.time() - t0
hist = m.train_history_
check("bump: train loss decreases", hist[-1] < hist[0] * 0.5, f"{hist[0]:.4f} -> {hist[-1]:.4f}")
check("bump: monotone-ish", sum(b > a for a, b in zip(hist, hist[1:])) <= 3)

# Signal recovery, at the operating point where this scenario is learnable.  The
# aggregate-and-concatenate baseline sits at chance here by construction, so
# anything well above 0.5 is signal the workaround cannot reach.  (At n=300 this
# task is a pure XOR with no marginal signal at the root and the model is near
# chance — a known v0.1 limitation, see README.)
Xs5, Xt5, y5 = make_bump_interaction(n=500, seed=0)
t0 = time.time()
m5 = HeartwoodClassifier(n_estimators=150, random_state=0).fit(Xs5, Xt5, y5)
acc = (m5.predict(Xs_te, Xt_te) == y_te).mean()
check("bump: recovers signal at n=500", acc > 0.65,
      f"acc={acc:.3f}  fit={time.time()-t0:.0f}s")

# ------------------------------------------------------------ determinism
p1 = HeartwoodClassifier(n_estimators=15, random_state=7).fit(Xs, Xt, y).predict_proba(Xs_te, Xt_te)
p2 = HeartwoodClassifier(n_estimators=15, random_state=7).fit(Xs, Xt, y).predict_proba(Xs_te, Xt_te)
p3 = HeartwoodClassifier(n_estimators=15, random_state=8).fit(Xs, Xt, y).predict_proba(Xs_te, Xt_te)
check("determinism: same seed identical", np.array_equal(p1, p2))
check("determinism: different seed differs", not np.array_equal(p1, p3))

# --------------------------------------------------------- early stopping
m_es = HeartwoodClassifier(n_estimators=200, early_stopping_rounds=5, random_state=0)
m_es.fit(Xs, Xt, y, eval_set=(Xs_te[:300], Xt_te[:300], y_te[:300]))
check("early stopping fires", m_es.best_iteration_ < 199, f"best_iter={m_es.best_iteration_}")

# ------------------------------------------------------------- regression
Xs_r, Xt_r, y_r = make_shape_amplitude_regression(n=300, seed=0)
Xs_r2, Xt_r2, y_r2 = make_shape_amplitude_regression(n=500, seed=1)
r = HeartwoodRegressor(n_estimators=60, random_state=0).fit(Xs_r, Xt_r, y_r)
pred = r.predict(Xs_r2, Xt_r2)
rmse = np.sqrt(np.mean((pred - y_r2) ** 2))
base = np.sqrt(np.mean((y_r2 - y_r.mean()) ** 2))
check("regression beats mean baseline", rmse < 0.6 * base, f"rmse={rmse:.3f} vs {base:.3f}")

# ------------------------------------------------------------- multiclass
rng = np.random.default_rng(0)
Xs_a, Xt_a, ya = make_bump_interaction(n=150, seed=2)
Xs_b, Xt_b, yb = make_timing_task(n=150, seed=3)
Xs_m = np.vstack([Xs_a, Xs_b, Xs_a])[:300]
Xt_m = np.vstack([Xt_a, Xt_b, Xt_a])[:300]
ym = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
mm = HeartwoodClassifier(n_estimators=20, random_state=0).fit(Xs_m, Xt_m, ym)
proba = mm.predict_proba(Xs_m, Xt_m)
check("multiclass proba shape", proba.shape == (300, 3), proba.shape)
check("multiclass rows sum to 1", np.allclose(proba.sum(1), 1.0))
check("multiclass classes_ round-trip", set(mm.predict(Xs_m, Xt_m)) <= {"a", "b", "c"})

# ------------------------------------------------- None modes + ragged input
Xs_p, _, y_p = make_pure_static(n=300, seed=0)
ms = HeartwoodClassifier(n_estimators=30, random_state=0).fit(Xs_p, None, y_p)
check("static-only fits", (ms.predict(Xs_p, None) == y_p).mean() > 0.85,
      f"train acc={(ms.predict(Xs_p, None) == y_p).mean():.3f}")

mt = HeartwoodClassifier(n_estimators=20, random_state=0).fit(None, Xt, y)
check("series-only fits", mt.predict(None, Xt).shape == y.shape)

ragged = [Xt[i, :, : 60 + (i % 40)] for i in range(len(Xt))]
maxT = max(r_.shape[1] for r_ in ragged)          # pad to the ragged max, not the original T
padded = np.full((len(ragged), Xt.shape[1], maxT), np.nan)
for i, r_ in enumerate(ragged):
    padded[i, :, : r_.shape[1]] = r_
a = HeartwoodClassifier(n_estimators=10, random_state=3).fit(Xs, ragged, y).predict_proba(Xs, ragged)
b = HeartwoodClassifier(n_estimators=10, random_state=3).fit(Xs, padded, y).predict_proba(Xs, padded)
check("ragged list == NaN-padded array", np.array_equal(a, b))

# --------------------------------------------------- routing fit == predict
from heartwood.tree import TemporalTree, TreeParams
from heartwood.losses import Logistic
loss = Logistic()
raw = np.tile(loss.init_score(y.astype(float)), (len(y), 1))
g, h = loss.grad_hess(y.astype(float), raw)
tree = TemporalTree(TreeParams(max_depth=3)).fit(
    Xs, Xt, g[:, 0], h[:, 0], np.arange(len(y)), np.random.default_rng(0))
leaves = tree.apply(Xs, Xt)
vals = tree.predict(Xs, Xt)
by_leaf = {int(li): vals[leaves == li][0] for li in np.unique(leaves)}
check("tree leaves consistent", all(np.allclose(vals[leaves == li], v) for li, v in by_leaf.items()))
check("max_depth honored", len(np.unique(leaves)) <= 2 ** 3)

# leaf value == -G/(H+lambda) by hand
lam = 1.0
ok = all(abs(v - (-g[leaves == li, 0].sum() / (h[leaves == li, 0].sum() + lam))) < 1e-9
         for li, v in by_leaf.items())
check("leaf value == -G/(H+lambda)", ok)

# ---------------------------------------------------------- interpretability
imp = m.feature_importances()
dump = m.dump_splits(top=5)
check("importances non-empty, positive", len(imp) > 0 and all(v > 0 for v in imp.values()))
check("dump_splits sorted desc", all(a[1] >= b[1] for a, b in zip(dump, dump[1:])))
print("\n  top families:", list(imp.items())[:4])
print("  top splits:")
for d, gn in dump:
    print(f"    {d}   gain={gn:.3f}")

# --------------------------------------------- slope scenario finds 'slope'
Xs_s, Xt_s, y_s = make_slope_window(n=300, seed=0)
msl = HeartwoodClassifier(n_estimators=40, random_state=0).fit(Xs_s, Xt_s, y_s)
fams = msl.feature_importances()
check("slope scenario surfaces slope splits",
      any("slope" in k for k in fams), list(fams)[:5])

# ------------------------------------------------- control: noise series ok
Xs_c, Xt_c, y_c = make_static_plus_noise_series(n=300, seed=0)
Xs_c2, Xt_c2, y_c2 = make_static_plus_noise_series(n=1000, seed=1)
mc = HeartwoodClassifier(n_estimators=60, random_state=0).fit(Xs_c, Xt_c, y_c)
acc_c = (mc.predict(Xs_c2, Xt_c2) == y_c2).mean()
check("control with noise series still learns", acc_c > 0.80, f"acc={acc_c:.3f}")

print("\nFAILURES:", fail)
sys.exit(1 if fail else 0)
