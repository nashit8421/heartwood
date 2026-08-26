"""Fetch the first 40 Sleep-EDF cassette subjects from PhysioNet's S3 mirror.

    python validation/fetch_sleepedf.py

V7-M2.  A different modality and task from every other dataset in this project,
and therefore the one that decides whether V6 found something about time series
or something about electrocardiograms.

Subject subset is fixed in VALIDATION_V7.md §2: the first 40 by record id, on
size alone (the full cassette set is ~7.5 GB), recorded before any Sleep-EDF
number was computed.
"""

from __future__ import annotations

import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com"
PREFIX = "sleep-edfx/1.0.0/sleep-cassette/"
ROOT = Path(__file__).resolve().parent / "data" / "sleepedf"
N_SUBJECTS = 40
WORKERS = 12


def listing() -> list[tuple[str, str]]:
    """(psg key, hypnogram key) for each subject, ordered by record id."""
    url = f"{BASE}/?list-type=2&prefix={PREFIX}&max-keys=1000"
    with urllib.request.urlopen(url, timeout=60) as response:
        keys = re.findall(r"<Key>(.*?)</Key>", response.read().decode())
    paired: dict[str, dict[str, str]] = {}
    for key in keys:
        name = key.rsplit("/", 1)[-1]
        if name.endswith("-PSG.edf"):
            paired.setdefault(name[:6], {})["psg"] = key
        elif "Hypnogram" in name:
            paired.setdefault(name[:6], {})["hyp"] = key
    return [(v["psg"], v["hyp"]) for _, v in sorted(paired.items())
            if "psg" in v and "hyp" in v]


def grab(key: str) -> str | None:
    target = ROOT / key.rsplit("/", 1)[-1]
    if target.exists() and target.stat().st_size:
        return None
    last = ""
    for _ in range(3):
        try:
            with urllib.request.urlopen(f"{BASE}/{key}", timeout=300) as response:
                target.write_bytes(response.read())
            return None
        except Exception as error:  # noqa: BLE001
            last = f"{type(error).__name__}: {error}"
    return f"{key}: {last}"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    subjects = listing()[:N_SUBJECTS]
    wanted = [k for pair in subjects for k in pair]
    print(f"{len(subjects)} subjects, {len(wanted)} files", flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, failure in enumerate(pool.map(grab, wanted), 1):
            if failure:
                failures.append(failure)
            if done % 10 == 0:
                print(f"  {done}/{len(wanted)} ({len(failures)} failed)", flush=True)
    print(f"failures: {len(failures)}")
    for failure in failures[:5]:
        print(f"  {failure}")
    print("SLEEPEDF_FETCH_COMPLETE" if not failures else "SLEEPEDF_FETCH_INCOMPLETE")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
