"""Real dataset loaders.

Every preprocessing decision here is a degree of freedom that could be nudged to
flatter a result, so each one is written down in the docstring and fixed in code
before any score was computed (see VALIDATION.md §9).

Each loader returns ``(X_static, X_series, y, meta)`` where ``X_series`` is
``(n, C, T)`` and ``meta`` records what was decided and why.
"""

from __future__ import annotations

import ssl
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Dataset:
    key: str
    X_static: np.ndarray
    X_series: np.ndarray
    y: np.ndarray
    task: str  # 'binary' | 'multiclass' | 'regression'
    headline: str  # the metric named in VALIDATION.md §5, fixed in advance
    static_names: list[str] = field(default_factory=list)
    channel_names: list[str] = field(default_factory=list)
    groups: np.ndarray | None = None  # rows sharing a subject must not be split apart
    notes: str = ""

    def summary(self) -> str:
        n, c, t = len(self.y), self.X_series.shape[1], self.X_series.shape[2]
        missing = float(np.isnan(self.X_series).mean())
        return (
            f"{self.key}: n={n}, static={self.X_static.shape[1]}, series={c}x{t}, "
            f"task={self.task}, missing={missing:.1%}"
        )


def _download(url: str, filename: str) -> Path:
    path = DATA_DIR / filename
    if path.exists():
        return path
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120, context=context) as response:
        path.write_bytes(response.read())
    return path


# --------------------------------------------------------------- M1: credit


def load_credit() -> Dataset:
    """UCI Default of Credit Card Clients (Taiwan), n=30,000.

    Decisions, fixed in advance:

    * **Static block** is the five genuinely per-customer attributes: credit
      limit, sex, education, marriage, age.
    * **Series block** is the three monthly channels — repayment status, bill
      amount, payment amount — over six months.
    * **Time order is reversed from the file's.** The columns are named most-
      recent-first (``BILL_AMT1`` is September, ``BILL_AMT6`` is April), so they
      are flipped to run oldest → newest. Feeding a series backwards would make
      "slope" and "how early" mean the opposite of what they say.
    * ``PAY_0`` is the September repayment status; the dataset simply skips the
      name ``PAY_1``.

    This is deliberately a hard case for Heartwood: T=6 leaves almost no room for
    windows or shapes. It is kept because dropping it would be exactly the bias
    VALIDATION.md exists to prevent.
    """
    import pandas as pd

    archive = _download(
        "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "credit.zip",
    )
    extracted = DATA_DIR / "default of credit card clients.xls"
    if not extracted.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATA_DIR)

    frame = pd.read_excel(extracted, header=1)

    static_cols = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    # oldest -> newest
    pay_status = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]
    bill = [f"BILL_AMT{i}" for i in (6, 5, 4, 3, 2, 1)]
    paid = [f"PAY_AMT{i}" for i in (6, 5, 4, 3, 2, 1)]

    X_static = frame[static_cols].to_numpy(dtype=np.float64)
    X_series = np.stack(
        [frame[cols].to_numpy(dtype=np.float64) for cols in (pay_status, bill, paid)],
        axis=1,
    )
    y = frame["default payment next month"].to_numpy(dtype=np.int64)

    return Dataset(
        key="credit",
        X_static=X_static,
        X_series=X_series,
        y=y,
        task="binary",
        headline="roc_auc",
        static_names=static_cols,
        channel_names=["repay_status", "bill_amount", "payment_amount"],
        notes="6 monthly steps, reversed to oldest-first; short series by design",
    )


# ------------------------------------------------------------------ M3: HAR


def load_har() -> Dataset:
    """UCI Human Activity Recognition from smartphones, 6 classes.

    Decisions, fixed in advance:

    * **Series block** is the nine raw inertial channels (body acceleration,
      gyroscope, total acceleration on x/y/z) at 128 timesteps per window. The
      561 engineered features shipped with the dataset are deliberately *not*
      used as the series — they are precisely the hand-aggregation this project
      exists to avoid, and using them would beg the question.
    * **Static block**: this dataset has no true per-row static covariates, so
      the subject identifier is used as a single column. That is a weak static
      block and the results should be read accordingly.
    * The official train/test split is used, and it is already subject-disjoint.
    """
    archive = _download(
        "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "har.zip",
    )
    root = DATA_DIR / "har"
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
        inner = root / "UCI HAR Dataset.zip"
        if inner.exists():
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(root)

    base = next(p for p in root.rglob("UCI HAR Dataset") if p.is_dir())
    signals = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    blocks, labels, subjects, is_train = [], [], [], []
    for split in ("train", "test"):
        folder = base / split / "Inertial Signals"
        channels = [
            np.loadtxt(folder / f"{name}_{split}.txt", dtype=np.float64)
            for name in signals
        ]
        blocks.append(np.stack(channels, axis=1))
        labels.append(np.loadtxt(base / split / f"y_{split}.txt", dtype=np.int64))
        subjects.append(np.loadtxt(base / split / f"subject_{split}.txt", dtype=np.int64))
        is_train.append(np.full(len(labels[-1]), split == "train"))

    X_series = np.concatenate(blocks)
    # labels ship as 1..6; encode to 0-based, which several baselines require
    _, y = np.unique(np.concatenate(labels), return_inverse=True)
    subject = np.concatenate(subjects)
    n_train = int(np.concatenate(is_train).sum())

    dataset = Dataset(
        key="har",
        X_static=subject.astype(np.float64)[:, None],
        X_series=X_series,
        y=y.astype(np.int64),
        task="multiclass",
        headline="balanced_accuracy",
        static_names=["subject"],
        channel_names=signals,
        groups=subject,
        notes="official split; raw inertial signals only, engineered features unused",
    )
    dataset.n_official_train = n_train  # type: ignore[attr-defined]
    return dataset


# -------------------------------------------------------------- M2: ICU 48h


#: The six values recorded once at admission; everything else is time-varying.
_ICU_DESCRIPTORS = ("RecordID", "Age", "Gender", "Height", "ICUType", "Weight")
_ICU_STATIC = ("Age", "Gender", "Height", "ICUType", "Weight")


def _parse_icu_record(path: Path):
    """One patient file -> (record_id, static dict, list of (hour, parameter, value))."""
    static: dict[str, float] = {}
    observations: list[tuple[int, str, float]] = []
    record_id = None

    with path.open() as handle:
        next(handle, None)  # header
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            stamp, parameter, raw = parts
            try:
                value = float(raw)
            except ValueError:
                continue
            if parameter == "RecordID":
                record_id = int(value)
                continue
            hours, minutes = stamp.split(":")
            hour = int(hours)
            if parameter in _ICU_STATIC and hour == 0 and parameter not in static:
                # -1 is the dataset's sentinel for "not recorded"
                static[parameter] = np.nan if value < 0 else value
            else:
                observations.append((min(hour, 47), parameter, value))
    return record_id, static, observations


def load_physionet_icu() -> Dataset:
    """PhysioNet/CinC 2012: predict in-hospital death from the first 48 h in ICU.

    This is the archetypal shape the library was built for — a handful of
    admission facts plus a long, irregular, mostly-absent clinical trajectory.

    Decisions, fixed in advance:

    * **Split** is the challenge's own: set-a trains, set-b tests. Not re-drawn.
    * **Static block** is the five admission descriptors (age, gender, height,
      ICU type, weight). The dataset's ``-1`` sentinel becomes NaN rather than
      being treated as a real measurement.
    * **Series block** is *every* time-varying parameter, in sorted order — no
      hand-picking of "useful" vitals, which would be a modelling choice smuggled
      into preprocessing.
    * **Binning is hourly**, 48 bins, taking the **mean** of whatever was
      recorded within each hour. Observations past hour 47 are clipped into the
      last bin.
    * **Empty bins stay NaN.** No imputation, no forward-filling. Roughly nine in
      ten cells are empty, and how a model handles that is precisely what is
      under test; filling them would answer a different question.
    """
    files: dict[str, list[Path]] = {}
    for split, url in (
        ("set-a", "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz"),
        ("set-b", "https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz"),
    ):
        archive = _download(url, f"{split}.tar.gz")
        folder = DATA_DIR / split
        if not folder.exists():
            import tarfile

            with tarfile.open(archive) as tar:
                tar.extractall(DATA_DIR)
        files[split] = sorted(folder.glob("*.txt"))

    outcomes: dict[int, int] = {}
    for split in ("a", "b"):
        path = _download(
            f"https://physionet.org/files/challenge-2012/1.0.0/Outcomes-{split}.txt",
            f"Outcomes-{split}.txt",
        )
        for line in path.read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 6:
                outcomes[int(parts[0])] = int(parts[5])

    parsed = []
    for split in ("set-a", "set-b"):
        for path in files[split]:
            record_id, static, observations = _parse_icu_record(path)
            if record_id is None or record_id not in outcomes or not observations:
                continue
            parsed.append((split, record_id, static, observations))

    parameters = sorted({p for _, _, _, obs in parsed for _, p, _ in obs})
    index = {p: i for i, p in enumerate(parameters)}

    n, T = len(parsed), 48
    X_series = np.full((n, len(parameters), T), np.nan)
    totals = np.zeros_like(X_series)
    counts = np.zeros_like(X_series)
    X_static = np.full((n, len(_ICU_STATIC)), np.nan)
    y = np.empty(n, dtype=np.int64)
    train_mask = np.zeros(n, dtype=bool)

    for row, (split, record_id, static, observations) in enumerate(parsed):
        for i, name in enumerate(_ICU_STATIC):
            X_static[row, i] = static.get(name, np.nan)
        for hour, parameter, value in observations:
            channel = index[parameter]
            totals[row, channel, hour] += value
            counts[row, channel, hour] += 1
        y[row] = outcomes[record_id]
        train_mask[row] = split == "set-a"

    observed = counts > 0
    X_series[observed] = totals[observed] / counts[observed]

    # keep the official split contiguous: train rows first
    order = np.concatenate([np.nonzero(train_mask)[0], np.nonzero(~train_mask)[0]])
    dataset = Dataset(
        key="icu",
        X_static=X_static[order],
        X_series=X_series[order],
        y=y[order],
        task="binary",
        headline="roc_auc",
        static_names=list(_ICU_STATIC),
        channel_names=parameters,
        notes="set-a trains / set-b tests; hourly means; empty bins left missing",
    )
    dataset.n_official_train = int(train_mask.sum())  # type: ignore[attr-defined]
    return dataset


# ------------------------------------------------------ T1: UEA multivariate


def load_uea(name: str) -> Dataset:
    """One dataset from the UEA multivariate archive, via ``aeon``.

    No static block — ``X_static`` is a zero-width array and Heartwood runs in
    its pure time-series mode.  This arm exists to compare against real
    time-series methods, not to test the mixed claim.
    """
    from aeon.datasets import load_classification

    X_train, y_train = load_classification(name, split="train", extract_path=str(DATA_DIR))
    X_test, y_test = load_classification(name, split="test", extract_path=str(DATA_DIR))

    X_series = np.concatenate([X_train, X_test]).astype(np.float64)
    y_raw = np.concatenate([y_train, y_test])
    classes, y = np.unique(y_raw, return_inverse=True)

    dataset = Dataset(
        key=f"uea:{name}",
        X_static=np.empty((len(y), 0)),
        X_series=X_series,
        y=y.astype(np.int64),
        task="binary" if len(classes) == 2 else "multiclass",
        headline="balanced_accuracy",
        channel_names=[f"ch{i}" for i in range(X_series.shape[1])],
        notes=f"official UEA split: first {len(y_train)} rows are train",
    )
    dataset.n_official_train = len(y_train)  # type: ignore[attr-defined]
    return dataset


# ------------------------------------------------------- V5-A1: PTB-XL ECG


#: The five diagnostic superclasses PTB-XL's own scp_statements.csv defines.
_PTBXL_SUPERCLASSES = ("CD", "HYP", "MI", "NORM", "STTC")


def _read_wfdb16(header: Path) -> np.ndarray:
    """One WFDB format-16 record -> ``(n_signals, n_samples)`` in physical units.

    PTB-XL ships as WFDB, and ``wfdb`` is a heavy dependency for a format this
    small: a text header naming gain and baseline per signal, then interleaved
    little-endian int16 samples.  Parsing it here keeps the project's "numpy
    only" install promise intact.  Anything that is not format 16 raises rather
    than being guessed at.
    """
    lines = [
        line for line in header.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    fields = lines[0].split()
    n_signals, n_samples = int(fields[1]), int(fields[3])

    gains, baselines, data_file = [], [], None
    for line in lines[1:1 + n_signals]:
        parts = line.split()
        data_file = parts[0]
        if parts[1] != "16":
            raise ValueError(f"{header}: expected format 16, got {parts[1]}")
        spec = parts[2].split("/")[0]  # e.g. "1000.0(0)" or "200"
        if "(" in spec:
            gain, baseline = spec.split("(")
            baselines.append(float(baseline.rstrip(")")))
        else:
            gain = spec
            baselines.append(float(parts[4]))  # adc_zero
        gains.append(float(gain) or 200.0)

    raw = np.fromfile(header.parent / data_file, dtype="<i2")
    raw = raw.reshape(-1, n_signals).T[:, :n_samples].astype(np.float32)
    gain = np.asarray(gains, dtype=np.float32)[:, None]
    baseline = np.asarray(baselines, dtype=np.float32)[:, None]
    return (raw - baseline) / gain


def load_ptbxl() -> Dataset:
    """PTB-XL 1.0.3: 12-lead ECG plus real patient demographics.

    Chosen for V5 because no v0.3 dataset occupied the cell this library was
    designed for -- genuine static covariates *and* series that a global summary
    demonstrably loses.  ICU and credit have statics whose series summarise
    away; HAR has shape-regime series but a subject-id "static" block that is
    disjoint across the split.  ECG has both.

    Every decision below is fixed in VALIDATION_V5.md §3, written and committed
    before this function was run:

    * **Split** is the dataset's own ``strat_fold`` -- folds 1-8 train, fold 10
      test, fold 9 unused.  Patient-disjoint by construction, never re-drawn.
    * **Label** is the diagnostic superclass.  A record is kept only if its SCP
      codes map to exactly one; records mapping to none and to two or more are
      dropped and both counts are reported in ``notes``.
    * **Static block** is age, sex, height, weight.  PTB-XL encodes ages over 89
      as 300; those become 90.  Missing height/weight stay missing.
    * **Series** is ``records100``: 12 leads x 1000 samples, float32.  The 500 Hz
      records are not used.
    """
    import pandas as pd

    root = DATA_DIR / "ptbxl"
    if not root.exists():
        # physionet.org serves this at ~100 KB/s, which is five hours for the
        # full archive; the project's own S3 mirror carries the same bytes and
        # lets the per-record files be fetched in parallel.  fetch_ptbxl.py does
        # that.  The zip path stays here as the dependency-free fallback.
        archive = _download(
            "https://physionet.org/static/published-projects/ptb-xl/"
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
            "ptbxl.zip",
        )
        with zipfile.ZipFile(archive) as zf:
            wanted = [
                name for name in zf.namelist()
                if "records100/" in name or name.endswith(
                    ("ptbxl_database.csv", "scp_statements.csv")
                )
            ]
            zf.extractall(root, members=wanted)

    database = next(root.rglob("ptbxl_database.csv"))
    base = database.parent
    frame = pd.read_csv(database, index_col="ecg_id")
    statements = pd.read_csv(base / "scp_statements.csv", index_col=0)
    diagnostic = statements[statements.diagnostic == 1].diagnostic_class

    cache = DATA_DIR / "ptbxl_series.npy"

    def superclasses(codes: str) -> set[str]:
        import ast
        return {
            diagnostic[code] for code in ast.literal_eval(codes)
            if code in diagnostic.index
        }

    labels = frame.scp_codes.map(superclasses)
    sizes = labels.map(len)
    keep = sizes == 1
    frame = frame[keep].copy()
    frame["label"] = labels[keep].map(lambda s: next(iter(s)))
    dropped = f"dropped {(sizes == 0).sum()} unlabelled, {(sizes > 1).sum()} multi-label"

    # only the official train and test folds; fold 9 is the standard validation
    # fold and is deliberately left unused so nothing can leak through it
    frame = frame[frame.strat_fold.isin(list(range(1, 9)) + [10])]
    frame = frame.sort_values(["strat_fold"], kind="stable")
    is_train = (frame.strat_fold <= 8).to_numpy()
    frame = frame.iloc[np.argsort(~is_train, kind="stable")]  # train rows first

    if cache.exists() and len(np.load(cache, mmap_mode="r")) == len(frame):
        X_series = np.load(cache)
    else:
        X_series = np.empty((len(frame), 12, 1000), dtype=np.float32)
        for row, name in enumerate(frame.filename_lr):
            X_series[row] = _read_wfdb16(base / f"{name}.hea")
        np.save(cache, X_series)

    age = frame.age.to_numpy(dtype=np.float64)
    age[age > 89] = 90.0  # PTB-XL's sentinel for "older than 89"
    X_static = np.column_stack([
        age,
        frame.sex.to_numpy(dtype=np.float64),
        frame.height.to_numpy(dtype=np.float64),
        frame.weight.to_numpy(dtype=np.float64),
    ])

    classes, y = np.unique(frame.label.to_numpy(), return_inverse=True)
    dataset = Dataset(
        key="ptbxl",
        X_static=X_static,
        X_series=X_series,
        y=y.astype(np.int64),
        task="multiclass",
        headline="balanced_accuracy",
        static_names=["age", "sex", "height", "weight"],
        channel_names=["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"],
        notes=f"official strat_fold 1-8 train / 10 test; {dropped}; "
              f"classes={list(classes)}",
    )
    dataset.n_official_train = int((frame.strat_fold <= 8).sum())  # type: ignore[attr-defined]
    return dataset


MIXED = {
    "credit": load_credit,
    "har": load_har,
    "icu": load_physionet_icu,
    "ptbxl": load_ptbxl,
}
