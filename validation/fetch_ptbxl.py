"""Fetch PTB-XL's 100 Hz records from PhysioNet's S3 mirror, in parallel.

    python validation/fetch_ptbxl.py

physionet.org serves the 1.7 GB archive at roughly 100 KB/s -- five hours -- and
the loader only needs ``records100`` plus two CSVs.  The project's own S3 mirror
holds the identical files and tolerates concurrent requests, so this pulls the
~44,000 small files directly.  ``load_ptbxl`` falls back to the zip if this has
not been run.
"""

from __future__ import annotations

import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3"
ROOT = Path(__file__).resolve().parent / "data" / "ptbxl"
WORKERS = 24


def grab(relative: str) -> str | None:
    """Download one file unless it is already present and non-empty."""
    target = ROOT / relative
    if target.exists() and target.stat().st_size:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            with urllib.request.urlopen(f"{BASE}/{relative}", timeout=60) as response:
                target.write_bytes(response.read())
            return None
        except Exception as error:  # noqa: BLE001 - retried, then reported
            last = f"{type(error).__name__}: {error}"
    return f"{relative}: {last}"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("ptbxl_database.csv", "scp_statements.csv", "RECORDS"):
        if failure := grab(name):
            print(f"FATAL {failure}")
            return 1

    # RECORDS ships without a newline between its last records100 entry and the
    # first records500 one, so a naive startswith() splice yields a concatenated
    # id that 404s. Match the exact shape instead.
    pattern = re.compile(r"records100/\d{5}/\d{5}_lr")
    text = (ROOT / "RECORDS").read_text()
    records = sorted(set(pattern.findall(text)))
    wanted = [f"{record}.{extension}" for record in records for extension in ("hea", "dat")]
    print(f"{len(records)} records, {len(wanted)} files", flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, failure in enumerate(pool.map(grab, wanted), 1):
            if failure:
                failures.append(failure)
            if done % 4000 == 0:
                print(f"  {done}/{len(wanted)}  ({len(failures)} failed)", flush=True)

    print(f"failures: {len(failures)}")
    for failure in failures[:10]:
        print(f"  {failure}")
    print("PTBXL_FETCH_COMPLETE" if not failures else "PTBXL_FETCH_INCOMPLETE")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
