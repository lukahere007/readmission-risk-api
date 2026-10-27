"""CLI: PSI drift check of a new batch of encounters against the training profile.

    python -m readmission.monitor new_batch.csv [--profile models/reference_profile.json]
                                               [--fail-on-major]

The batch must contain the raw input columns (same names as the API / UCI file).
Exit code 1 with --fail-on-major if any feature has PSI >= 0.25.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data import ROOT, engineer_features
from .monitoring import MAJOR, drift_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch", type=Path)
    ap.add_argument("--profile", type=Path, default=ROOT / "models" / "reference_profile.json")
    ap.add_argument("--fail-on-major", action="store_true")
    args = ap.parse_args(argv)

    profile = json.loads(args.profile.read_text())
    batch = pd.read_csv(args.batch, na_values=["?"], keep_default_na=False, low_memory=False)
    report = drift_report(profile, engineer_features(batch))
    print(f"batch: {args.batch} ({len(batch)} rows)")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    n_major = int((report.psi >= MAJOR).sum())
    print(f"\n{n_major} feature(s) with major drift (PSI >= {MAJOR})")
    return 1 if (args.fail_on_major and n_major) else 0


if __name__ == "__main__":
    sys.exit(main())
