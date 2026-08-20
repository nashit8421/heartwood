"""Run the pre-registered validation (VALIDATION.md) on real datasets.

    python validation/run_validation.py --datasets credit
    python validation/run_validation.py --datasets credit har --sizes 100 250 500 1000 0

Size ``0`` means "the whole training split".  Heartwood runs on library defaults
throughout; baselines get the same rounds/depth/learning rate.  Nothing here is
tuned per dataset, by rule.

Two corrections to the v0.3 run are folded in here and recorded in
``validation/CORRECTION.md``: the aggregate baselines no longer propagate NaN
(``benchmarks.baselines``), and an official train/test split no longer collapses
the small-data curve to a single subsample.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.baselines import build_design_matrix, make_baseline_model
from benchmarks.run_benchmarks import classification_metrics, regression_metrics, roc_auc
from heartwood import HeartwoodClassifier, HeartwoodRegressor
from validation.loaders import MIXED, load_uea

HEARTWOOD = "heartwood"
#: ``agg_naive``/``wagg8_naive`` are the NaN-propagating reductions this harness
#: originally used; they are carried alongside the fixed ones so every table
#: shows what that choice was worth.  See ``validation/CORRECTION.md``.
REPRESENTATIONS = ["static_only", "agg", "wagg4", "wagg8", "raw_flat",
                   "agg_naive", "wagg8_naive"]


# ------------------------------------------------------------------ metrics


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = [
        float((y_pred[y_true == c] == c).mean())
        for c in np.unique(y_true)
        if (y_true == c).any()
    ]
    return float(np.mean(recalls))


def evaluate(task: str, y_true, y_pred, scores) -> dict[str, float]:
    if task == "regression":
        return regression_metrics(y_true, y_pred)
    out = {"balanced_accuracy": balanced_accuracy(y_true, y_pred),
           "accuracy": float((y_pred == y_true).mean())}
    if task == "binary":
        out.update(classification_metrics(y_true, y_pred, scores))
    elif scores is not None and scores.ndim == 2:
        # one-vs-rest AUC, averaged over classes present in the test set
        aucs = [
            roc_auc((y_true == c).astype(int), scores[:, i])
            for i, c in enumerate(np.unique(y_true))
            if 0 < (y_true == c).sum() < len(y_true)
        ]
        out["roc_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    return out


# ------------------------------------------------------------------- splits


def make_split(dataset, seed: int, test_fraction: float = 0.3):
    """Official split when the dataset ships one; otherwise stratified, grouped."""
    n = len(dataset.y)
    official = getattr(dataset, "n_official_train", None)
    if official is not None:
        return np.arange(official), np.arange(official, n)

    rng = np.random.default_rng(seed)
    if dataset.groups is not None:
        groups = np.unique(dataset.groups)
        rng.shuffle(groups)
        n_test = max(1, int(round(test_fraction * len(groups))))
        test_groups = set(groups[:n_test].tolist())
        is_test = np.array([g in test_groups for g in dataset.groups])
        return np.nonzero(~is_test)[0], np.nonzero(is_test)[0]

    train, test = [], []
    for label in np.unique(dataset.y):
        idx = np.nonzero(dataset.y == label)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(test_fraction * len(idx))))
        test.append(idx[:cut])
        train.append(idx[cut:])
    return np.concatenate(train), np.concatenate(test)


def subsample(train_idx: np.ndarray, y: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Stratified subsample of the training split; ``size=0`` keeps everything."""
    if size <= 0 or size >= len(train_idx):
        return train_idx
    rng = np.random.default_rng(seed + 7919)
    labels = y[train_idx]
    chosen = []
    for label in np.unique(labels):
        pool = train_idx[labels == label]
        take = max(1, int(round(size * len(pool) / len(train_idx))))
        chosen.append(rng.choice(pool, size=min(take, len(pool)), replace=False))
    return np.concatenate(chosen)


# ---------------------------------------------------------------- one cell


@dataclass
class Row:
    dataset: str
    model: str
    n_train: int
    seed: int
    fit_seconds: float
    metrics: dict


def fit_minirocket(X_train, y_train, X_test, n_kernels: int = 2000):
    """MiniROCKET + ridge — a real time-series method, as the plan requires."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clean_train = np.nan_to_num(X_train)
    clean_test = np.nan_to_num(X_test)
    model = make_pipeline(
        MiniRocket(n_kernels=n_kernels, random_state=0),
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    model.fit(clean_train, y_train)
    predictions = model.predict(clean_test)
    decision = model.decision_function(clean_test)
    return predictions, decision


def run_cell(dataset, train_idx, test_idx, size, seed, config) -> list[Row]:
    rows: list[Row] = []
    tr = subsample(train_idx, dataset.y, size, seed)
    Xs, Xt, y = dataset.X_static[tr], dataset.X_series[tr], dataset.y[tr]
    Xs_te, Xt_te, y_te = (
        dataset.X_static[test_idx], dataset.X_series[test_idx], dataset.y[test_idx]
    )
    n_train = len(tr)

    def record(name, predictions, scores, elapsed):
        rows.append(Row(dataset.key, name, n_train, seed, elapsed,
                        evaluate(dataset.task, y_te, predictions, scores)))

    # --- Heartwood, library defaults
    estimator = HeartwoodRegressor if dataset.task == "regression" else HeartwoodClassifier
    started = time.perf_counter()
    model = estimator(
        n_estimators=config["rounds"], max_depth=config["depth"],
        learning_rate=config["learning_rate"], random_state=seed,
    ).fit(Xs if Xs.shape[1] else None, Xt, y)
    elapsed = time.perf_counter() - started
    static_arg = Xs_te if Xs.shape[1] else None
    predictions = model.predict(static_arg, Xt_te)
    scores = None
    if dataset.task != "regression":
        proba = model.predict_proba(static_arg, Xt_te)
        scores = proba[:, 1] if dataset.task == "binary" else proba
    record(HEARTWOOD, predictions, scores, elapsed)

    # --- the workarounds
    for name in config["representations"]:
        if name == "static_only" and Xs.shape[1] == 0:
            continue
        design = build_design_matrix(name, Xs, Xt)
        design_te = build_design_matrix(name, Xs_te, Xt_te)
        baseline, _ = make_baseline_model(
            "regression" if dataset.task == "regression" else "binary",
            config["rounds"], config["depth"], config["learning_rate"], seed,
        )
        started = time.perf_counter()
        baseline.fit(design, y)
        elapsed = time.perf_counter() - started
        predictions = baseline.predict(design_te)
        scores = None
        if dataset.task != "regression":
            proba = baseline.predict_proba(design_te)
            scores = proba[:, 1] if dataset.task == "binary" else proba
        record(name, predictions, scores, elapsed)

    # --- a real time-series method, at both kernel budgets
    #
    # 10,000 is MiniROCKET's published default; 2,000 is what this harness first
    # used.  Neither dominates -- on ICU the smaller bank is the stronger
    # baseline -- so both are recorded and any claim about beating MiniROCKET is
    # made against whichever won.
    if config["minirocket"] and dataset.task != "regression":
        for label, n_kernels in (("minirocket", 2000), ("minirocket10k", 10000)):
            try:
                started = time.perf_counter()
                predictions, decision = fit_minirocket(Xt, y, Xt_te, n_kernels)
                elapsed = time.perf_counter() - started
                scores = decision if dataset.task == "binary" else None
                record(label, predictions, scores, elapsed)
            except Exception as error:  # recorded, never silently skipped
                print(f"    {label} failed: {type(error).__name__}: {error}", flush=True)

    return rows


# -------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["credit"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 250, 500, 1000, 0])
    parser.add_argument("--seeds", type=int, default=5,
                        help="repeats of the subsample (VALIDATION.md §5 fixes this at 5)")
    parser.add_argument("--full-seeds", type=int, default=2,
                        help="repeats for the whole-training-split cell of an "
                             "official split, where only the model seed varies")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--no-minirocket", action="store_true")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    config = {
        "rounds": args.rounds, "depth": args.depth,
        "learning_rate": args.learning_rate, "representations": REPRESENTATIONS,
        "minirocket": not args.no_minirocket,
    }

    results: list[Row] = []
    unavailable: dict[str, str] = {}

    for key in args.datasets:
        try:
            dataset = load_uea(key.split(":", 1)[1]) if key.startswith("uea:") else MIXED[key]()
        except Exception as error:
            unavailable[key] = f"{type(error).__name__}: {error}"
            print(f"UNAVAILABLE {key}: {unavailable[key]}", flush=True)
            continue

        print(f"\n{dataset.summary()}", flush=True)
        official = getattr(dataset, "n_official_train", None)

        for seed in range(args.seeds):
            train_idx, test_idx = make_split(dataset, seed)
            for size in args.sizes:
                if size and size > len(train_idx):
                    continue
                whole = not size or size >= len(train_idx)
                # An official split fixes the *split*, not the *subsample*, so
                # the small-data curve still gets its repeats (VALIDATION.md §5).
                # An earlier version collapsed those to one seed, which is why
                # the v0.3 tables carry a meaningless "+/-0.000".  The one case
                # where extra seeds really are redundant is an official split
                # with no subsampling: every seed then sees identical rows and
                # only re-measures the learner's own randomness.
                if official is not None and whole and seed >= args.full_seeds:
                    continue
                started = time.perf_counter()
                cells = run_cell(dataset, train_idx, test_idx, size, seed, config)
                results.extend(cells)
                label = size or len(train_idx)
                print(f"  seed {seed} n={label}: "
                      + "  ".join(
                          f"{r.model}={r.metrics.get(dataset.headline, float('nan')):.3f}"
                          for r in cells
                      )
                      + f"  ({time.perf_counter() - started:.0f}s)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "unavailable": unavailable,
        "results": [asdict(r) for r in results],
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out / 'results.json'} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
