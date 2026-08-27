"""Real dataset loaders.

Every preprocessing decision here is a degree of freedom that could be nudged to
flatter a result, so each one is written down in the docstring and fixed in code
before any score was computed (see VALIDATION.md §9).

Each loader returns ``(X_static, X_series, y, meta)`` where ``X_series`` is
``(n, C, T)`` and ``meta`` records what was decided and why.
"""

from __future__ import annotations

import re
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


# --------------------------------------------------- V7-M1: CPSC-2018 ECG


#: The nine CPSC-2018 classes, by SNOMED code, for readable reporting only --
#: selection uses record counts, never these names.
_CPSC_NAMES = {
    "426783006": "NSR", "164889003": "AF", "270492004": "IAVB",
    "164909002": "LBBB", "59118001": "RBBB", "284470004": "PAC",
    "164884008": "PVC", "429622005": "STD", "164931005": "STE",
}
_CPSC_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF",
               "V1", "V2", "V3", "V4", "V5", "V6")
_CPSC_MIN_CLASS = 200      # a class must hold this many records to be kept
_CPSC_SECONDS = 10         # first 10 s of each record
_CPSC_DECIMATE = 5         # 500 Hz -> 100 Hz, matching PTB-XL


def _read_cpsc_record(header: Path):
    """One CPSC record -> ``(series, age, sex, dx codes)``.

    The signal ships as a MATLAB v4 file whose WFDB header gives the byte offset
    and per-lead gain, so it reads as plain interleaved int16 -- no scipy, no
    wfdb.  Demographics live in the header's trailing comments.
    """
    lines = header.read_text().splitlines()
    fields = lines[0].split()
    n_signals, n_samples = int(fields[1]), int(fields[3])

    gains, data_file, offset = [], None, 0
    for line in lines[1:1 + n_signals]:
        parts = line.split()
        data_file = parts[0]
        spec = parts[1]                       # e.g. "16x1+24"
        if "+" in spec:
            offset = int(spec.split("+")[1])
        gains.append(float(parts[2].split("/")[0].split("(")[0]) or 1000.0)

    raw = np.fromfile(header.parent / data_file, dtype="<i2", offset=offset)
    usable = (raw.size // n_signals) * n_signals
    block = raw[:usable].reshape(-1, n_signals).T.astype(np.float64)
    block = block[:, :n_samples] / np.asarray(gains, dtype=np.float64)[:, None]

    age, sex, codes = np.nan, np.nan, []
    for line in lines[n_signals + 1:]:
        if line.startswith("# Age:"):
            token = line.split(":", 1)[1].strip()
            age = float(token) if token.replace(".", "").isdigit() else np.nan
        elif line.startswith("# Sex:"):
            token = line.split(":", 1)[1].strip().lower()
            sex = 1.0 if token.startswith("m") else (0.0 if token.startswith("f") else np.nan)
        elif line.startswith("# Dx:"):
            codes = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
    return block, age, sex, codes


def load_cpsc2018() -> Dataset:
    """CPSC-2018 12-lead ECG with age and sex, via PhysioNet/CinC 2020.

    V7-M1.  A *different* ECG cohort from PTB-XL -- different country, label set
    and sampling rate -- so it asks whether the V6 win replicates on data it has
    never seen, with modality held fixed.

    Decisions, fixed in VALIDATION_V7.md §2 before the data was parsed:

    * **Series** is all 12 leads, the first 10 s, brought from 500 Hz to 100 Hz
      to match PTB-XL's resolution.  The plan says "by decimation"; this averages
      each block of 5 samples rather than taking every 5th, because plain
      subsampling of a 500 Hz ECG aliases QRS energy straight into the band the
      model reads.  Same rate, one fewer artefact.
    * **Records shorter than 10 s are dropped**, and counted in ``notes``.
    * **Label** is the SNOMED ``Dx`` code.  A record is kept only if it maps to
      exactly one class holding at least 200 records; records with none or with
      several are dropped and both counts reported.
    * **Static block** is age and sex.  Unknown stays NaN -- no imputation.
    * **Split** is the harness's stratified 70/30, redrawn per seed, since this
      dataset ships no official split and each record is one patient.
    """
    root = DATA_DIR / "cpsc2018"
    headers = sorted(root.rglob("A*.hea"))
    if not headers:
        raise FileNotFoundError(
            f"no CPSC records under {root}; run `python validation/fetch_cpsc.py` first"
        )

    cache = DATA_DIR / "cpsc2018_series.npy"
    length = _CPSC_SECONDS * 500

    parsed, short, unreadable = [], 0, 0
    for header in headers:
        # A header can exist before its signal file does, or a transfer can be
        # truncated; neither should take down a study that runs for hours.
        try:
            block, age, sex, codes = _read_cpsc_record(header)
        except (OSError, ValueError, IndexError):
            unreadable += 1
            continue
        if block.shape[1] < length:
            short += 1
            continue
        parsed.append((header.stem, block[:, :length], age, sex, codes))

    counts: dict[str, int] = {}
    for _, _, _, _, codes in parsed:
        for code in codes:
            counts[code] = counts.get(code, 0) + 1
    frequent = {code for code, n in counts.items() if n >= _CPSC_MIN_CLASS}

    kept, unlabelled, multi = [], 0, 0
    for record, block, age, sex, codes in parsed:
        hits = [c for c in codes if c in frequent]
        if not hits:
            unlabelled += 1
        elif len(hits) > 1:
            multi += 1
        else:
            kept.append((record, block, age, sex, hits[0]))

    n = len(kept)
    if cache.exists() and len(np.load(cache, mmap_mode="r")) == n:
        X_series = np.load(cache)
    else:
        X_series = np.empty((n, 12, _CPSC_SECONDS * 100), dtype=np.float32)
        for row, (_, block, _, _, _) in enumerate(kept):
            # average each block of 5 samples: 500 Hz -> 100 Hz without aliasing
            X_series[row] = block.reshape(12, -1, _CPSC_DECIMATE).mean(axis=2)
        np.save(cache, X_series)

    X_static = np.array([[age, sex] for _, _, age, sex, _ in kept], dtype=np.float64)
    classes, y = np.unique([label for *_, label in kept], return_inverse=True)

    return Dataset(
        key="cpsc2018",
        X_static=X_static,
        X_series=X_series,
        y=y.astype(np.int64),
        task="multiclass",
        headline="balanced_accuracy",
        static_names=["age", "sex"],
        channel_names=list(_CPSC_LEADS),
        notes=(f"stratified 70/30 per seed; dropped {short} under {_CPSC_SECONDS}s, "
               f"{unreadable} unreadable, {unlabelled} unlabelled, "
               f"{multi} multi-label; classes="
               + ",".join(_CPSC_NAMES.get(c, c) for c in classes)),
    )


# ------------------------------------------------ V7-M2: Sleep-EDF cassette


#: AASM stages. 3 and 4 merge, per the modern standard; "?" and movement drop.
_SLEEP_STAGES = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
                 "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}
_SLEEP_NAMES = ("W", "N1", "N2", "N3", "REM")
_SLEEP_HZ = 50             # from 100 Hz; keeps delta and spindles, halves the fit
_SLEEP_EPOCH_SECONDS = 30
_SLEEP_WAKE_MARGIN = 60    # epochs of wake kept either side of sleep (30 min)


def _read_edf_header(path: Path) -> dict:
    """EDF header: a fixed 256-byte block, then 256 bytes per signal."""
    with path.open("rb") as handle:
        fixed = handle.read(256)
        n_signals = int(fixed[252:256])
        rest = handle.read(256 * n_signals)

    def field(offset: int, width: int, index: int) -> str:
        start = offset * n_signals + width * index
        return rest[start:start + width].decode("latin-1").strip()

    return {
        "patient": fixed[8:88].decode("latin-1").strip(),
        "n_records": int(fixed[236:244]),
        "duration": float(fixed[244:252]),
        "header_bytes": int(fixed[184:192]),
        "n_signals": n_signals,
        "labels": [field(0, 16, i) for i in range(n_signals)],
        # per-signal field offsets: label 0, transducer 16, physical dimension 96,
        # then min/max pairs at 104/112 and 120/128, prefiltering 136, samples 216
        "physical_min": [float(field(104, 8, i)) for i in range(n_signals)],
        "physical_max": [float(field(112, 8, i)) for i in range(n_signals)],
        "digital_min": [float(field(120, 8, i)) for i in range(n_signals)],
        "digital_max": [float(field(128, 8, i)) for i in range(n_signals)],
        "samples": [int(field(216, 8, i)) for i in range(n_signals)],
    }


def _read_edf_signal(path: Path, header: dict, index: int) -> np.ndarray:
    """One signal as ``(n_records, samples_per_record)`` in physical units."""
    per_record = header["samples"]
    stride = sum(per_record)
    raw = np.fromfile(path, dtype="<i2", offset=header["header_bytes"])
    usable = (raw.size // stride) * stride
    records = raw[:usable].reshape(-1, stride)
    start = sum(per_record[:index])
    block = records[:, start:start + per_record[index]].astype(np.float64)

    digital_span = header["digital_max"][index] - header["digital_min"][index]
    physical_span = header["physical_max"][index] - header["physical_min"][index]
    scale = physical_span / digital_span if digital_span else 1.0
    return (block - header["digital_min"][index]) * scale + header["physical_min"][index]


def _read_hypnogram(path: Path) -> list[tuple[float, float, str]]:
    """EDF+ annotations as ``(onset, duration, label)``.

    Annotations are stored as TALs — ``+onset\x15duration\x14label\x14`` — inside
    the data records of a signal named ``EDF Annotations``.
    """
    header = _read_edf_header(path)
    blob = path.read_bytes()[header["header_bytes"]:]
    pattern = re.compile(rb"([+-]\d+(?:\.\d+)?)\x15(\d+(?:\.\d+)?)\x14([^\x14\x00]*)\x14")
    return [(float(a), float(b), c.decode("latin-1"))
            for a, b, c in pattern.findall(blob)]


def _sleep_demographics(patient: str) -> tuple[float, float]:
    """Age and sex out of the EDF patient field, e.g. ``X F X Female_33yr``."""
    age = re.search(r"(\d+)\s*yr", patient)
    sex = np.nan
    lowered = patient.lower()
    if "female" in lowered:
        sex = 0.0
    elif "male" in lowered:
        sex = 1.0
    return (float(age.group(1)) if age else np.nan), sex


def load_sleepedf() -> Dataset:
    """Sleep-EDF cassette: EEG epochs plus the sleeper's age and sex.

    V7-M2, and the load-bearing one. Every other dataset in this project that
    has both real statics and shape-regime series is an ECG, so this is what
    decides whether V6 found something about time series or about hearts.

    Decisions, fixed in VALIDATION_V7.md §2 before the data was parsed:

    * **Subjects** are the first 40 by record id, on size alone.
    * **Series** is the Fpz-Cz EEG. Each 30 s EDF data record is exactly one
      scoring epoch, which is why no windowing is needed. Kept at 50 Hz rather
      than the native 100 -- that preserves delta and sleep spindles, the two
      things staging actually turns on, and halves a fit that would otherwise
      take hours per cell.
    * **Labels** are the AASM stages, with 3 and 4 merged as is now standard;
      ``?`` and movement epochs are dropped and counted.
    * **Wake trimming**: at most 30 minutes of wake either side of the sleep
      period, the usual convention for this dataset, applied before any score.
    * **Static block** is age and sex, read from the EDF patient field.
    * **Split** is subject-disjoint, which the harness enforces via ``groups``.
    """
    root = DATA_DIR / "sleepedf"
    psgs = sorted(root.glob("*-PSG.edf"))
    if not psgs:
        raise FileNotFoundError(
            f"no Sleep-EDF records under {root}; run `python validation/fetch_sleepedf.py`"
        )

    step = 100 // _SLEEP_HZ
    width = _SLEEP_EPOCH_SECONDS * _SLEEP_HZ
    blocks, labels, subjects, statics, dropped = [], [], [], [], 0

    for psg in psgs:
        matches = sorted(root.glob(f"{psg.name[:6]}*-Hypnogram.edf"))
        if not matches:
            continue
        header = _read_edf_header(psg)
        if "EEG Fpz-Cz" not in header["labels"]:
            continue
        signal = _read_edf_signal(psg, header, header["labels"].index("EEG Fpz-Cz"))

        stage = np.full(len(signal), -1, dtype=np.int64)
        for onset, duration, name in _read_hypnogram(matches[0]):
            code = _SLEEP_STAGES.get(name)
            if code is None:
                continue
            first = int(onset // _SLEEP_EPOCH_SECONDS)
            last = min(len(stage), first + max(1, int(duration // _SLEEP_EPOCH_SECONDS)))
            stage[first:last] = code

        keep = np.nonzero(stage >= 0)[0]
        dropped += len(stage) - len(keep)
        if not len(keep):
            continue
        asleep = np.nonzero(stage[keep] > 0)[0]
        if len(asleep):  # trim the long wake tails this dataset is known for
            lo = max(0, asleep[0] - _SLEEP_WAKE_MARGIN)
            hi = min(len(keep), asleep[-1] + _SLEEP_WAKE_MARGIN + 1)
            keep = keep[lo:hi]

        epochs = signal[keep]
        usable = (epochs.shape[1] // step) * step
        blocks.append(epochs[:, :usable].reshape(len(keep), -1, step).mean(axis=2)[:, :width])
        labels.append(stage[keep])
        subjects.append(np.full(len(keep), int(psg.name[2:6])))
        statics.append(np.tile(_sleep_demographics(header["patient"]), (len(keep), 1)))

    if not blocks:
        raise RuntimeError("Sleep-EDF parsed but produced no usable epochs")

    X_series = np.concatenate(blocks).astype(np.float32)[:, None, :]
    y = np.concatenate(labels)
    group = np.concatenate(subjects)
    X_static = np.concatenate(statics)

    return Dataset(
        key="sleepedf",
        X_static=X_static,
        X_series=X_series,
        y=y.astype(np.int64),
        task="multiclass",
        headline="balanced_accuracy",
        static_names=["age", "sex"],
        channel_names=["EEG Fpz-Cz"],
        groups=group,
        notes=(f"{len(psgs)} subjects, subject-disjoint split; {_SLEEP_HZ} Hz; "
               f"dropped {dropped} unscored epochs; classes=" + ",".join(_SLEEP_NAMES)),
    )


# ------------------------------------------------- V9-M1: Apnea-ECG minutes


_APNEA_HZ = 50            # from 100 Hz; respiratory modulation sits near 0.2-0.3 Hz
_APNEA_SECONDS = 60
#: WFDB annotation codes used by the .apn files: N (non-apnea), A (apnea).
_APNEA_CODES = {1: 0, 8: 1}


def _read_wfdb_annotations(path: Path) -> list[tuple[int, int]]:
    """WFDB annotation file -> ``(sample, code)`` pairs.

    Each 16-bit word packs ``code = word >> 10`` and ``delta = word & 0x3FF``.
    Code 59 escapes to a 32-bit interval in the next two words, 63 introduces an
    auxiliary string to skip, and 60-62 are per-annotation metadata with no time
    of their own.  Validated against the counts published in the dataset's own
    additional-information.txt.
    """
    raw = np.fromfile(path, dtype="<u2")
    out: list[tuple[int, int]] = []
    sample, index = 0, 0
    while index < len(raw):
        word = int(raw[index]); index += 1
        if word == 0:
            break
        code, delta = word >> 10, word & 0x3FF
        if code == 59:
            delta = (int(raw[index]) << 16) | int(raw[index + 1]); index += 2
            code = int(raw[index]) >> 10; index += 1
            sample += delta
            out.append((sample, code))
            continue
        if code in (60, 61, 62):
            continue
        if code == 63:
            index += (delta + 1) // 2
            continue
        sample += delta
        out.append((sample, code))
    return out


def load_apnea() -> Dataset:
    """PhysioNet Apnea-ECG: one-minute ECG segments plus the subject's body size.

    V9-M1, and the first fair test of this library's premise.  Every earlier
    dataset either had statics the signal itself encodes (age from an ECG) or a
    series a plain average already captures.  Body-mass index is the primary
    risk factor for obstructive sleep apnea, spans 19.2 to 41.7 here, and is not
    present in a one-minute single-lead ECG at any resolution.

    Decisions, fixed in VALIDATION_V9.md §3 before the data was parsed:

    * **Series** is the single ECG lead, one minute per row, 100 Hz averaged down
      to 50 Hz.  Apnea appears as respiratory modulation near 0.2-0.3 Hz and in
      heart-rate variability; both survive comfortably.
    * **Target** is the per-minute expert annotation, apnea against non-apnea.
    * **Static block** is age, sex, height, weight and BMI.  BMI is included
      deliberately: it is the established risk factor and a tree cannot form a
      ratio of two columns by splitting on them.
    * **Split** is subject-disjoint; statics are constant within a subject.
    """
    root = DATA_DIR / "apnea"
    info = root / "additional-information.txt"
    if not info.exists():
        raise FileNotFoundError(
            f"no Apnea-ECG under {root}; run `python validation/fetch_apnea.py` first"
        )

    table = re.findall(
        r"^([abc]\d\d)\t\d+\t.*?\t(\d+)\t([MF])\t(\d+)\t(\d+)",
        info.read_text(), re.M,
    )
    demographics = {
        record: (float(age), 1.0 if sex == "M" else 0.0, float(height), float(weight))
        for record, age, sex, height, weight in table
    }

    step = 100 // _APNEA_HZ
    width = _APNEA_SECONDS * _APNEA_HZ
    blocks, labels, groups, statics, skipped = [], [], [], [], 0

    for record, (age, sex, height, weight) in sorted(demographics.items()):
        header, data = root / f"{record}.hea", root / f"{record}.dat"
        annotations = root / f"{record}.apn"
        if not (header.exists() and data.exists() and annotations.exists()):
            skipped += 1
            continue
        fields = header.read_text().splitlines()[1].split()
        gain = float(fields[2].split("/")[0].split("(")[0]) or 200.0
        signal = np.fromfile(data, dtype="<i2").astype(np.float64) / gain

        minutes = [(s, _APNEA_CODES[c]) for s, c in _read_wfdb_annotations(annotations)
                   if c in _APNEA_CODES]
        per_minute = _APNEA_SECONDS * 100
        kept = []
        for start, label in minutes:
            if start + per_minute <= signal.size:
                kept.append((signal[start:start + per_minute], label))
        if not kept:
            skipped += 1
            continue

        raw = np.stack([segment for segment, _ in kept])
        usable = (raw.shape[1] // step) * step
        blocks.append(raw[:, :usable].reshape(len(kept), -1, step).mean(axis=2)[:, :width])
        labels.append(np.array([label for _, label in kept]))
        groups.append(np.full(len(kept), int(record[1:]) + (0 if record[0] == "a"
                                                            else 100 if record[0] == "b" else 200)))
        bmi = weight / (height / 100.0) ** 2
        statics.append(np.tile([age, sex, height, weight, bmi], (len(kept), 1)))

    X_series = np.concatenate(blocks).astype(np.float32)[:, None, :]
    y = np.concatenate(labels)
    return Dataset(
        key="apnea",
        X_static=np.concatenate(statics),
        X_series=X_series,
        y=y.astype(np.int64),
        task="binary",
        headline="roc_auc",
        static_names=["age", "sex", "height", "weight", "bmi"],
        channel_names=["ECG"],
        groups=np.concatenate(groups),
        notes=(f"{len(blocks)} subjects, subject-disjoint; {_APNEA_HZ} Hz; "
               f"{skipped} records skipped; apnea rate {y.mean():.3f}"),
    )


MIXED = {
    "credit": load_credit,
    "har": load_har,
    "icu": load_physionet_icu,
    "ptbxl": load_ptbxl,
    "cpsc2018": load_cpsc2018,
    "sleepedf": load_sleepedf,
    "apnea": load_apnea,
}
