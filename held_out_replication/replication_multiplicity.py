"""Multiplicity sensitivity of the replication tests.

Within each tier, each supported test's bound is recomputed at the Bonferroni level 0.025/m (one-sided), from a
normal approximation to its bootstrap interval (SE = half-width on the criterion side / 1.96). Reads tests.csv of
replication_tests.py; writes multiplicity.csv and summary.json to results/replication_multiplicity/<run>/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit

REPLICATION_RUN = "results/replication_tests"


def criterion(text: str) -> tuple[str | None, float | None]:
    m = re.match(r"(lower|upper) bound ([<>]) (-?[0-9.]+)", str(text))
    return (m.group(1), float(m.group(3))) if m else (None, None)


def load_tests(root: Path) -> pd.DataFrame:
    t = pd.read_csv(root / REPLICATION_RUN / "tests.csv")
    t["supported"] = t["supported"].astype(str).str.lower() == "true"
    return t


def multiplicity(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier, part in t.groupby("tier", sort=False):
        m = len(part)
        z = float(norm.ppf(1 - 0.025 / m))
        for r in part.itertuples():
            side, thr = criterion(r.criterion)
            adj, survives = np.nan, None
            if side is not None and r.supported:
                if side == "lower":
                    se = (r.estimate - r.ci_low) / 1.96
                    adj = r.estimate - z * se
                    survives = bool(adj > thr)
                else:
                    se = (r.ci_high - r.estimate) / 1.96
                    adj = r.estimate + z * se
                    survives = bool(adj < thr)
            rows.append({"id": r.id, "tier": tier, "m": m, "z": z, "supported": r.supported, "criterion": r.criterion,
                         "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "adjusted_bound": adj,
                         "survives_bonferroni": survives, "power_at_realised_n": r.power_at_realised_n,
                         "n_children": r.n_children})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    paths = ProjectPaths.discover()
    commit = git_commit(paths.root)
    t = load_tests(paths.root)
    mult = multiplicity(t)
    out_dir = paths.results / "replication_multiplicity" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(mult, out_dir / "multiplicity.csv")
    sup = mult[mult["supported"]]
    summary = {"git_commit": commit, "n_tests": len(mult), "n_supported": int(mult["supported"].sum()),
               "n_supported_surviving_bonferroni": int(sup["survives_bonferroni"].fillna(False).astype(bool).sum()),
               "not_surviving": sup.loc[~sup["survives_bonferroni"].fillna(False).astype(bool), "id"].tolist()}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
