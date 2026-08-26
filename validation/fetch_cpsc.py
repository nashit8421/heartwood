"""Fetch the CPSC-2018 ECG records (via PhysioNet/CinC 2020) in parallel.

    python validation/fetch_cpsc.py

V7-M1.  A different ECG cohort from PTB-XL -- different country, label set and
sampling rate -- so it tests whether the V6 win replicates on data it has never
seen while holding modality fixed.

There is no usable RECORDS index at this path, so record ids are enumerated:
A0001..A6877, with record ``i`` living in ``g{i // 1000 + 1}``, verified against
the boundaries rather than assumed.
"""

from __future__ import annotations

import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018"
ROOT = Path(__file__).resolve().parent / "data" / "cpsc2018"
LAST_RECORD = 6877
WORKERS = 24


def relative(index: int, extension: str) -> str:
    return f"g{index // 1000 + 1}/A{index:04d}.{extension}"


def grab(relative_path: str) -> str | None:
    target = ROOT / relative_path
    if target.exists() and target.stat().st_size:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    last = ""
    for _ in range(3):
        try:
            with urllib.request.urlopen(f"{BASE}/{relative_path}", timeout=60) as response:
                target.write_bytes(response.read())
            return None
        except Exception as error:  # noqa: BLE001 - retried, then reported
            last = f"{type(error).__name__}: {error}"
    return f"{relative_path}: {last}"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    wanted = [relative(i, ext) for i in range(1, LAST_RECORD + 1) for ext in ("hea", "mat")]
    print(f"{LAST_RECORD} records, {len(wanted)} files", flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, failure in enumerate(pool.map(grab, wanted), 1):
            if failure:
                failures.append(failure)
            if done % 2000 == 0:
                print(f"  {done}/{len(wanted)}  ({len(failures)} failed)", flush=True)

    print(f"failures: {len(failures)}")
    for failure in failures[:10]:
        print(f"  {failure}")
    print("CPSC_FETCH_COMPLETE" if not failures else "CPSC_FETCH_INCOMPLETE")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
