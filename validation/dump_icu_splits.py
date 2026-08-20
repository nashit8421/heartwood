"""H5: dump the ICU model's highest-gain splits, with clinical names.

    python validation/dump_icu_splits.py

VALIDATION.md §6 asks for this qualitatively -- "record whether they are
domain-plausible, reported verbatim, including when they are nonsense" -- and
the v0.3 write-up quoted the output without committing the code that produced
it.  This is that code, so the claim can be re-checked rather than trusted.

Nothing is filtered or reordered: the top splits are the top splits by gain, and
the only edit is replacing channel indices with the parameter names the loader
already carries.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartwood import HeartwoodClassifier
from validation.loaders import MIXED

TOP_SPLITS = 12
TOP_FAMILIES = 15


def main() -> int:
    dataset = MIXED["icu"]()
    n_train = dataset.n_official_train  # set-a; set-b is never touched here
    print(dataset.summary())

    model = HeartwoodClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=0
    ).fit(dataset.X_static[:n_train], dataset.X_series[:n_train], dataset.y[:n_train])

    def name(description: str) -> str:
        described = re.sub(
            r"series\[ch=(\d+)\]",
            lambda m: f"series[{dataset.channel_names[int(m.group(1))]}]",
            description,
        )
        for index, column in enumerate(dataset.static_names):
            described = described.replace(f"static[{index}]", column)
        return described

    splits = model.dump_splits()

    print(f"\ntop {TOP_SPLITS} splits by gain")
    for description, gain in splits[:TOP_SPLITS]:
        print(f"  gain={gain:7.1f}  {name(description)}")

    families: dict[str, float] = defaultdict(float)
    for description, gain in splits:
        families[name(description).split("[t=")[0]] += gain

    print(f"\ntop {TOP_FAMILIES} families by total gain")
    for family, gain in sorted(families.items(), key=lambda item: -item[1])[:TOP_FAMILIES]:
        print(f"  {gain:8.1f}  {family}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
