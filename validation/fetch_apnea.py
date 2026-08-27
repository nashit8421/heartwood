"""Fetch PhysioNet Apnea-ECG (V9-M1) from the S3 mirror.

    python validation/fetch_apnea.py

Only the 35 released recordings carry per-minute expert annotations, and only
those appear in additional-information.txt with age, sex, height and weight.
"""

from __future__ import annotations

import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com/apnea-ecg/1.0.0"
ROOT = Path(__file__).resolve().parent / "data" / "apnea"
WORKERS = 12


def grab(name: str) -> str | None:
    target = ROOT / name
    if target.exists() and target.stat().st_size:
        return None
    last = ""
    for _ in range(3):
        try:
            with urllib.request.urlopen(f"{BASE}/{name}", timeout=180) as response:
                target.write_bytes(response.read())
            return None
        except Exception as error:  # noqa: BLE001
            last = f"{type(error).__name__}: {error}"
    return f"{name}: {last}"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    if failure := grab("additional-information.txt"):
        print(f"FATAL {failure}")
        return 1
    text = (ROOT / "additional-information.txt").read_text()
    records = re.findall(r"^([abc]\d\d)\t", text, re.M)
    wanted = [f"{r}.{ext}" for r in records for ext in ("hea", "dat", "apn")]
    print(f"{len(records)} annotated records, {len(wanted)} files", flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, failure in enumerate(pool.map(grab, wanted), 1):
            if failure:
                failures.append(failure)
            if done % 20 == 0:
                print(f"  {done}/{len(wanted)} ({len(failures)} failed)", flush=True)
    print(f"failures: {len(failures)}")
    for failure in failures[:5]:
        print(f"  {failure}")
    print("APNEA_FETCH_COMPLETE" if not failures else "APNEA_FETCH_INCOMPLETE")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
