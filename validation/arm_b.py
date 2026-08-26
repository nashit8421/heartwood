"""Arm B of the H-V7.3 decomposition: a ridge over our bank, with no trees.

    python validation/arm_b.py sleepedf 1000

Isolates bank quality from everything the booster adds. Read against `aeon`
MiniROCKET (arm A, same shape but their bank) and against `heartwood_rocket`
(arms C and D, which add the trees and then the statics).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartwood.rocket import RocketBank
from validation.loaders import MIXED, load_uea
from validation.run_validation import balanced_accuracy, make_split, subsample


def main() -> int:
    key, sizes = sys.argv[1], [int(v) for v in sys.argv[2:]] or [1000]
    dataset = load_uea(key.split(":", 1)[1]) if key.startswith("uea:") else MIXED[key]()
    print(dataset.summary(), flush=True)

    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler

    rows = []
    for seed in range(3):
        train_idx, test_idx = make_split(dataset, seed)
        for size in sizes:
            if size and size > len(train_idx):
                continue
            tr = subsample(train_idx, dataset.y, size, seed)
            started = time.perf_counter()
            bank = RocketBank(n_features=10000, random_state=seed).fit(dataset.X_series[tr])
            features = bank.transform(dataset.X_series[tr])
            held = bank.transform(dataset.X_series[test_idx])
            scaler = StandardScaler().fit(features)
            model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(
                scaler.transform(features), dataset.y[tr]
            )
            score = balanced_accuracy(
                dataset.y[test_idx], model.predict(scaler.transform(held))
            )
            rows.append({"dataset": dataset.key, "model": "arm_b_ridge_our_bank",
                         "n_train": len(tr), "seed": seed, "fit_seconds":
                         time.perf_counter() - started,
                         "metrics": {"balanced_accuracy": score}})
            print(f"  seed {seed} n={len(tr)}: arm_b={score:.3f} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    out = Path(__file__).parent / "rerun" / f"v7_armb_{dataset.key.replace(':', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"config": {"arm": "B"}, "unavailable": {}, "results": rows}, indent=1))
    print(f"wrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
