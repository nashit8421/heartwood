"""V25: one bank, many heads — what belongs on top of MiniROCKET's features?

    python validation/run_v25.py --out validation/rerun/v25_uea

Every arm shares the identical transform (aeon MiniRocket at 10,000 kernels plus
a no-centering standard scaler, which is exactly the ``minirocket10k`` baseline's
pipeline) and differs only in the final estimator.  The bank is computed once per
cell and reused by every head, which is both faster and the thing that makes the
comparison about heads.

Splits come from ``run_validation.make_split``, so the cells line up with
``RESULTS_V24.md`` and its Heartwood column needs no re-running.  The ``ridge``
arm is the control: it is the same pipeline as ``minirocket10k`` and must
reproduce it (H-V25.1).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

_spec = importlib.util.spec_from_file_location("run_validation", HERE / "run_validation.py")
run_validation = importlib.util.module_from_spec(_spec)
sys.modules["run_validation"] = run_validation
_spec.loader.exec_module(run_validation)

make_split = run_validation.make_split
balanced_accuracy = run_validation.balanced_accuracy
load_uea = run_validation.load_uea

DATASETS = [
    "Epilepsy", "BasicMotions", "ERing", "RacketSports", "Libras", "NATOPS",
    "UWaveGestureLibrary", "ArticularyWordRecognition", "StandWalkJump",
    "Heartbeat", "CharacterTrajectories", "HandMovementDirection",
    "SelfRegulationSCP2", "Handwriting", "Cricket", "EthanolConcentration",
]


def heads(seed: int) -> dict:
    """The eight candidates, fixed in VALIDATION_V25.md before any cell ran."""
    from sklearn.ensemble import (ExtraTreesClassifier,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    return {
        "ridge": RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        "logreg": LogisticRegression(max_iter=2000, random_state=seed),
        "linsvc": LinearSVC(random_state=seed),
        "hgb": HistGradientBoostingClassifier(random_state=seed),
        "rf": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "extratrees": ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, random_state=seed),
    }


def build_bank(X_train, X_test, seed: int):
    """The identical transform every head sees."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.preprocessing import StandardScaler

    rocket = MiniRocket(n_kernels=10000, random_state=0)
    train = rocket.fit_transform(np.nan_to_num(X_train))
    test = rocket.transform(np.nan_to_num(X_test))
    scaler = StandardScaler(with_mean=False).fit(train)
    return scaler.transform(train), scaler.transform(test)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "rerun" / "v25_uea")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results, unavailable = [], {}

    def checkpoint():
        (args.out / "results.json").write_text(json.dumps(
            {"unavailable": unavailable, "results": results}, indent=1))

    for name in args.datasets:
        key = f"uea:{name}"
        try:
            dataset = load_uea(name)
        except Exception as error:
            unavailable[key] = f"{type(error).__name__}: {error}"
            print(f"{key}: UNAVAILABLE ({type(error).__name__})", flush=True)
            checkpoint()
            continue
        print(f"\n{key}: {dataset.summary()}", flush=True)

        for seed in range(args.seeds):
            train_idx, test_idx = make_split(dataset, seed)
            started = time.perf_counter()
            train, test = build_bank(dataset.X_series[train_idx],
                                     dataset.X_series[test_idx], seed)
            bank_seconds = time.perf_counter() - started
            y, y_test = dataset.y[train_idx], dataset.y[test_idx]

            scores = {}
            for arm, model in heads(seed).items():
                started = time.perf_counter()
                try:
                    model.fit(train, y)
                    score = balanced_accuracy(y_test, model.predict(test))
                except Exception as error:
                    print(f"  seed {seed} {arm}: FAILED {type(error).__name__}",
                          flush=True)
                    results.append({"dataset": key, "model": f"bank_{arm}",
                                    "n_train": len(train_idx), "seed": seed,
                                    "failed": f"{type(error).__name__}: {error}",
                                    "metrics": {}})
                    continue
                scores[arm] = score
                results.append({"dataset": key, "model": f"bank_{arm}",
                                "n_train": len(train_idx), "seed": seed,
                                "fit_seconds": time.perf_counter() - started,
                                "bank_seconds": bank_seconds,
                                "metrics": {"balanced_accuracy": score}})
            checkpoint()
            print(f"  seed {seed} n={len(train_idx)}: "
                  + "  ".join(f"{a}={s:.3f}" for a, s in scores.items())
                  + f"  (bank {bank_seconds:.0f}s)", flush=True)

    checkpoint()
    print(f"\nwrote {args.out / 'results.json'} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
