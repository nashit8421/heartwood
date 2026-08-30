"""Fetch PAMAP2 Physical Activity Monitoring (roadmap item 7 candidate).

    python validation/fetch_pamap2.py

Chosen because the roadmap names its shape directly: wearable accelerometry
where "body weight is genuinely exogenous to a step signal".  Nine subjects wear
three IMUs and a heart-rate monitor through twelve activities, and the dataset
ships a subject description file with each one's age, sex, height, weight,
resting heart rate and maximum heart rate.

That is the combination four previous attempts did not have.  Sleep-EDF's
statics were at chance; CPSC's were age and sex on an ECG, which the ECG
encodes.  Body mass is not written into an accelerometer trace the way age is
written into a QRS complex -- but that is the hypothesis, not the finding, and
``validation/screen_dataset.py`` is what tests it before any study is committed
to.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
ROOT = Path(__file__).resolve().parent / "data" / "pamap2"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "pamap2.zip"
    if not archive.exists() or not archive.stat().st_size:
        print(f"downloading {URL}")
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=600) as response:
            archive.write_bytes(response.read())
    print(f"{archive} ({archive.stat().st_size / 1e6:.0f} MB)")

    # The distribution nests a second zip inside the first.
    with zipfile.ZipFile(archive) as outer:
        outer.extractall(ROOT)
    for inner in ROOT.rglob("*.zip"):
        if inner == archive:
            continue
        with zipfile.ZipFile(inner) as handle:
            handle.extractall(inner.parent)

    protocol = list(ROOT.rglob("Protocol/*.dat"))
    subjects = list(ROOT.rglob("subjectInformation.pdf")) + list(
        ROOT.rglob("*subject*ormation*"))
    print(f"{len(protocol)} protocol recordings, {len(subjects)} subject files")
    if not protocol:
        print("FATAL: no Protocol/*.dat found; the archive layout has changed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
