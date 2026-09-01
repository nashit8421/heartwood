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


# ------------------------------------------------- Heartwood arm definitions

#: ``""`` is the shipped default.  Everything else puts a ridge over the kernel
#: bank underneath the trees (V6); the name is what a results table shows.
#:
#: V15 and V23 ablated the four extras this library added to MiniROCKET's bank --
#: virtual channels, comparison splits, a window-statistic block and Levy areas.
#: All four are now deleted, so their arms and their flags are gone with them;
#: the results stand in ``RESULTS_V15.md`` and ``RESULTS_V23.md``.
VARIANTS: dict[str, dict] = {
    "": {},
    "rocket_static": {"dense_base": True, "dense_include_static": True,
                      "dense_static_interactions": False},
    "rocket_inter": {"dense_base": True, "dense_include_static": True,
                     "dense_static_interactions": True},
    # V8: the convolution base plus a chance floor on split acceptance.
    "rocket_null": {"dense_base": True, "selection_null": 1},
}

#: V16 (roadmap item 2a): per-node bagging over the temporal draws, on top of
#: the shipped rocket base so the arm differs from ``rocket_static`` in exactly
#: one setting.  Fractions are named in advance and none is added later.
_BAGGING_FRACTIONS = (0.5, 0.25, 0.125)
for _fraction in _BAGGING_FRACTIONS:
    VARIANTS[f"bag{int(_fraction * 1000):03d}"] = {
        **VARIANTS["rocket_static"], "candidate_colsample": _fraction,
    }

#: Bagging arm -> its fraction.  The report reads this rather than parsing names.
BAGGING_ARMS = {f"bag{int(f * 1000):03d}": f for f in _BAGGING_FRACTIONS}
BAGGING_BASELINE = "rocket_static"

#: V17 (roadmap item 2b): the analytic multiple-comparisons charge, swept over
#: three multipliers named in advance.  1.0 is the value the derivation implies;
#: the other two bracket it, because the additive term of the fit was not stable
#: across gradient regimes and this multiplier is where that slack lives.
_MC_PENALTIES = (0.5, 1.0, 2.0)
for _mc in _MC_PENALTIES:
    VARIANTS[f"mc{int(_mc * 100):03d}"] = {
        **VARIANTS["rocket_static"], "mc_penalty": _mc,
    }

MC_ARMS = {f"mc{int(m * 100):03d}": m for m in _MC_PENALTIES}
MC_BASELINE = "rocket_static"

#: V18 (roadmap item 2c): the out-of-fold bank pre-screen.  The knob swept is
#: the shortlist length, at a fixed 25% screening fold -- sweeping both would be
#: a two-dimensional search dressed as a study, and the fold size is the part
#: with a principled default (enough rows to rank on, few enough to spare).
_SCREEN_TOP_K = (4, 8, 16)
for _k in _SCREEN_TOP_K:
    VARIANTS[f"screen{_k:02d}"] = {
        **VARIANTS["rocket_static"], "screen_fraction": 0.25, "screen_top_k": _k,
    }

#: H-V18.2's control: the same fraction of rows per tree, no screening.  A
#: screening win this arm reproduces is not a screening win.
VARIANTS["sub075"] = {**VARIANTS["rocket_static"], "subsample": 0.75}

SCREEN_ARMS = {f"screen{k:02d}": k for k in _SCREEN_TOP_K}
SCREEN_BASELINE = "rocket_static"

#: V19 (roadmap item 2d): the recalibrated permutation null.  The knob swept is
#: the quantile; the permutation count is fixed at 16 because a 0.95 tail
#: estimated from 4 draws is biased low, which was measured before these arms
#: were written (see VALIDATION_V19.md §2).
_NULL_PERMUTATIONS = 16
_NULL_QUANTILES = (0.5, 0.9, 0.95)
for _q in _NULL_QUANTILES:
    VARIANTS[f"null_q{int(_q * 100):02d}"] = {
        **VARIANTS["rocket_static"],
        "selection_null": _NULL_PERMUTATIONS, "selection_null_quantile": _q,
    }

#: V8's arm exactly -- one permutation, the global maximum as the floor.  Kept
#: so V19 is a comparison against what actually failed rather than against a
#: reconstruction of it.
VARIANTS["null_v8"] = {**VARIANTS["rocket_static"],
                       "selection_null": 1, "selection_null_quantile": 1.0}

NULL_ARMS = {f"null_q{int(q * 100):02d}": q for q in _NULL_QUANTILES}
NULL_BASELINE = "rocket_static"

#: V20 (roadmap item 3): the no-regret guarantee, and the two components it
#: promises never to be much worse than.  ``comp_base`` is the ridge alone
#: (no trees); ``comp_trees`` is the trees alone (no ridge under them).
VARIANTS["comp_base"] = {**VARIANTS["rocket_static"], "n_estimators": 0}
VARIANTS["comp_trees"] = {**VARIANTS["rocket_static"], "dense_base": False}
VARIANTS["noregret"] = {**VARIANTS["rocket_static"], "no_regret": True}

#: Arm names the V20 report reads.  ``guarded`` is the model under test,
#: ``unguarded`` is the same architecture without the guarantee, and the
#: components are what the bar is measured against.
NO_REGRET_ARMS = {
    "guarded": "noregret",
    "unguarded": "rocket_static",
    "components": ("comp_base", "comp_trees"),
}

#: V21 (roadmap item 4): the nonlinear base.  The knob swept is the width of the
#: random-feature block at the default bandwidth; sweeping width and bandwidth
#: together would be a two-dimensional search dressed as a study, and the
#: bandwidth is the one with a principled default (the design's own width).
_NONLINEAR_WIDTHS = (256, 1024, 4096)
for _w in _NONLINEAR_WIDTHS:
    VARIANTS[f"rff{_w:04d}"] = {
        **VARIANTS["rocket_static"], "nonlinear_features": _w,
    }

NONLINEAR_ARMS = {f"rff{w:04d}": w for w in _NONLINEAR_WIDTHS}
NONLINEAR_BASELINE = "rocket_static"

#: V22 (roadmap item 5): magnitude products, by the two routes that can carry
#: them.  ``prod_split`` gives the trees a banked temporal feature crossed with
#: a static; ``prod_margin`` gives them the base's out-of-fold margin crossed
#: with a static; ``prod_both`` is the union.
VARIANTS["prod_split"] = {**VARIANTS["rocket_static"], "n_product_candidates": 4}
VARIANTS["prod_margin"] = {**VARIANTS["rocket_static"], "base_static_products": True}
VARIANTS["prod_both"] = {**VARIANTS["rocket_static"],
                         "n_product_candidates": 4, "base_static_products": True}

PRODUCT_ARMS = ("prod_split", "prod_margin", "prod_both")
PRODUCT_BASELINE = "rocket_static"

#: The four extras under test, and the arm that switches each one on.  Written
#: here rather than in the report so the reporting script cannot quietly change
#: what "the +virtual-channels arm" means after a score has been seen.

#: Extras that are vacuous on one channel: ``levy_area_columns`` returns an empty
#: block, so that arm is *identical* to ``abl_min`` there.  V15 counts an extra's
#: majority only over the datasets where it is live.


def variant_kwargs(variant: str) -> dict:
    """Estimator keyword arguments for a named arm.

    An unknown name is an error rather than a silent fallback.  It used to mean
    ``dense_features=<name>``, which addressed the window-statistic bank; that
    bank was deleted after V15 and a typo would otherwise now produce a default
    model wearing the wrong label in a results table.
    """
    if variant not in VARIANTS:
        raise SystemExit(
            f"unknown variant {variant!r}; known arms: {sorted(VARIANTS)}")
    return dict(VARIANTS[variant])


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

    # --- Heartwood, library defaults, plus any requested base variants
    estimator = HeartwoodRegressor if dataset.task == "regression" else HeartwoodClassifier
    static_arg = Xs_te if Xs.shape[1] else None
    for variant in config["variants"]:
        extra = variant_kwargs(variant)
        name = HEARTWOOD if not variant else f"{HEARTWOOD}_{variant}"
        started = time.perf_counter()
        # Merged rather than splatted: an arm is allowed to override the grid's
        # own settings (V20's base-only arm sets n_estimators=0), and ``**extra``
        # beside an explicit keyword would raise instead.
        settings = {"n_estimators": config["rounds"], "max_depth": config["depth"],
                    "learning_rate": config["learning_rate"], "random_state": seed}
        settings.update(extra)
        model = estimator(**settings).fit(Xs if Xs.shape[1] else None, Xt, y,
              groups=dataset.groups[tr] if dataset.groups is not None else None)
        elapsed = time.perf_counter() - started
        predictions = model.predict(static_arg, Xt_te)
        scores = None
        if dataset.task != "regression":
            proba = model.predict_proba(static_arg, Xt_te)
            scores = proba[:, 1] if dataset.task == "binary" else proba
        record(name, predictions, scores, elapsed)

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


def restrict_channels(dataset, channels: list[str]):
    """Keep only the named channels, in the order named. H-V14.1.

    Channel count is confounded with dataset in every result so far -- the two
    12-lead sets beat MiniROCKET, the two single-channel sets split.  Holding the
    dataset, task, split and seed fixed and varying only the width is the one way
    to ask whether the width causes the margin.

    Selection is by *name*, never by position, so a subset is reproducible from
    the results file alone and a renamed or reordered loader fails loudly instead
    of silently ablating a different lead.
    """
    names = list(dataset.channel_names)
    if len(names) != dataset.X_series.shape[1]:
        raise SystemExit(
            f"{dataset.key}: {len(names)} channel names for "
            f"{dataset.X_series.shape[1]} channels; cannot select by name")
    missing = [c for c in channels if c not in names]
    if missing:
        raise SystemExit(
            f"{dataset.key}: no such channel {missing}; available {names}")
    take = [names.index(c) for c in channels]
    dataset.X_series = dataset.X_series[:, take, :]
    dataset.channel_names = list(channels)
    dataset.notes += f" | channels restricted to {list(channels)}"
    return dataset


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
    parser.add_argument("--max-test", type=int, default=0,
                        help="cap the test split by stratified subsample. Sleep-EDF has "
                             "43k epochs, so ~13k land in test and prediction dominates "
                             "the cell: 2900s at n=100, where the fit itself is trivial. "
                             "Balanced accuracy over 4000 stratified rows is precise "
                             "enough that the extra 9k buys nothing but hours.")
    parser.add_argument("--channels", nargs="+", default=None,
                        help="restrict the series to these named channels "
                             "(H-V14.1 channel ablation)")
    parser.add_argument("--drop-static", action="store_true",
                        help="blank the static block for every model (V7 arm C): "
                             "isolates what the static covariates are actually worth")
    parser.add_argument("--variants", nargs="+", default=[""],
                        help='Heartwood configurations to run: "" is the shipped '
                             'default, "rocket"/"stats"/"both" put a ridge over that '
                             "feature bank underneath the trees, and the abl_* family "
                             "is the V15 bank ablation. Names come from VARIANTS.")
    parser.add_argument("--representations", nargs="+", default=REPRESENTATIONS,
                        help="baselines to run; a wide one (raw_flat on 12k columns) can "
                             "be split into its own pass so it cannot stall the grid")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    config = {
        "rounds": args.rounds, "depth": args.depth,
        "learning_rate": args.learning_rate, "representations": args.representations,
        "minirocket": not args.no_minirocket, "variants": args.variants,
    }

    results: list[Row] = []
    unavailable: dict[str, str] = {}

    args.out.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        """Write results after every cell, not once at the end.

        A PTB-XL grid is hours long, and an interrupted run used to lose every
        completed cell because the only write happened after the last dataset.
        Compute this expensive is worth flushing.
        """
        (args.out / "results.json").write_text(json.dumps({
            "config": {k: str(v) for k, v in vars(args).items()},
            "unavailable": unavailable,
            "results": [asdict(r) for r in results],
        }, indent=1))

    for key in args.datasets:
        try:
            dataset = load_uea(key.split(":", 1)[1]) if key.startswith("uea:") else MIXED[key]()
        except Exception as error:
            unavailable[key] = f"{type(error).__name__}: {error}"
            checkpoint()
            print(f"UNAVAILABLE {key}: {unavailable[key]}", flush=True)
            continue

        if args.channels:
            restrict_channels(dataset, args.channels)

        if args.drop_static:
            # Arm C of the H-V7.3 decomposition. Heartwood also beats MiniROCKET on
            # datasets with no static block at all, so "we use the statics" cannot
            # be the whole story; this measures the part that is.
            dataset.X_static = np.empty((len(dataset.y), 0))
            dataset.static_names = []
        print(f"\n{dataset.summary()}", flush=True)
        official = getattr(dataset, "n_official_train", None)

        for seed in range(args.seeds):
            train_idx, test_idx = make_split(dataset, seed)
            if args.max_test and len(test_idx) > args.max_test:
                test_idx = subsample(test_idx, dataset.y, args.max_test, seed)
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
                checkpoint()
                label = size or len(train_idx)
                print(f"  seed {seed} n={label}: "
                      + "  ".join(
                          f"{r.model}={r.metrics.get(dataset.headline, float('nan')):.3f}"
                          for r in cells
                      )
                      + f"  ({time.perf_counter() - started:.0f}s)", flush=True)

    checkpoint()
    print(f"\nwrote {args.out / 'results.json'} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
