"""Heartwood quickstart: the aggregate-and-concatenate workaround vs. the real thing.

The dataset is built so that summarising the series destroys the signal: every
row contains one transient, either up-then-down or down-then-up, and the label is
that orientation XOR a static flag.  The two orientations are exact negations of
each other, so the two classes have identical global statistics — mean, standard
deviation, min, max, median and mean-absolute-change are all uninformative by
construction.  Only *when and how* the series moves separates them.

Both models below are the same booster with the same hyperparameters.  The only
difference is what a split is allowed to ask about.

Run:  python examples/quickstart.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heartwood import HeartwoodClassifier
from heartwood.datasets import make_bump_interaction

N_TRAIN, N_TEST, ROUNDS, SEED = 500, 2000, 150, 0


def aggregate(X_series):
    """The industry-standard workaround: collapse each series to global summaries."""
    X = X_series[:, 0, :]
    t = np.arange(X.shape[1], dtype=float)
    tc = t - t.mean()
    slope = (X * tc).sum(1) / (tc * tc).sum()
    return np.column_stack([
        X.mean(1), X.std(1), X.min(1), X.max(1), slope, np.median(X, axis=1),
        np.abs(np.diff(X, axis=1)).mean(1), X[:, 0], X[:, -1], X[:, -1] - X[:, 0],
    ])


def main() -> int:
    X_static, X_series, y = make_bump_interaction(n=N_TRAIN, seed=SEED)
    X_static_te, X_series_te, y_te = make_bump_interaction(n=N_TEST, seed=SEED + 99)
    print(f"train {X_static.shape[0]} rows | static {X_static.shape[1]} cols | "
          f"series {X_series.shape[1]}x{X_series.shape[2]} | test {N_TEST} rows\n")

    common = dict(n_estimators=ROUNDS, learning_rate=0.1, max_depth=4, random_state=SEED)

    # ---- the workaround: aggregate the series, concatenate, boost on the table
    t0 = time.time()
    baseline = HeartwoodClassifier(**common)
    baseline.fit(np.hstack([X_static, aggregate(X_series)]), None, y)
    base_acc = (baseline.predict(np.hstack([X_static_te, aggregate(X_series_te)])) == y_te).mean()
    base_s = time.time() - t0

    # ---- Heartwood: same booster, but splits may look inside the raw series
    t0 = time.time()
    model = HeartwoodClassifier(**common)
    model.fit(X_static, X_series, y)
    acc = (model.predict(X_static_te, X_series_te) == y_te).mean()
    fit_s = time.time() - t0

    hist = model.train_history_
    marks = [0, ROUNDS // 4, ROUNDS // 2, 3 * ROUNDS // 4, ROUNDS - 1]
    print("training loss (logloss), raw series:")
    print("  " + "  ".join(f"round {m:>3}: {hist[m]:.4f}" for m in marks) + "\n")

    print(f"{'aggregate + boost':<20s} test accuracy = {base_acc:.3f}   ({base_s:.0f}s)")
    print(f"{'Heartwood':<20s} test accuracy = {acc:.3f}   ({fit_s:.0f}s)")
    print(f"{'':<20s}          gain = {100 * (acc - base_acc):+.1f} points\n")

    print("what the model looked at (total gain by feature family):")
    for family, gain in list(model.feature_importances().items())[:6]:
        print(f"  {family:<34s} {gain:8.1f}")

    print("\nhighest-gain individual splits:")
    for description, gain in model.dump_splits(top=5):
        print(f"  {description:<62s} gain={gain:6.2f}")

    assert hist[-1] < hist[0], "training loss should decrease"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
