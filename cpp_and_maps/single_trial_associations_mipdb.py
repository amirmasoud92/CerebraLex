"""The within-child single-trial associations in MIPDB children.

Same measures, cohort rules (artifact-free epochs, at least 10 valid responses) and estimators as the HBN analysis
(``within_association`` of single_trial_associations.py). Writes results/single_trial_associations_mipdb/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import utc_timestamp, write_dataframe
from common.paths import ProjectPaths, data_location
from common.scripts import script_path
from state_models_and_trait_moderation.permutation import fdr_correct


def _associations():
    spec = importlib.util.spec_from_file_location("single_trial_associations", script_path("single_trial_associations"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=data_location("MIPDB"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    assoc = _associations()
    paths = ProjectPaths.discover()
    measures = pd.read_parquet(Path(args.root) / "features" / "epochs-v1" / "trial_measures.parquet")
    frame = measures[measures["window_valid"].astype(bool) & ~measures["artifact_window"].astype(bool)].copy()
    per_child = frame[frame["valid_response"].astype(bool)].groupby("subject").size()
    frame = frame[frame["subject"].isin(per_child[per_child >= 10].index)]
    frame["log_rt"] = np.log(frame["rt_s"].where(frame["valid_response"].astype(bool)))
    frame["lapse_f"] = frame["lapse"].astype(float)
    rows = [assoc.within_association(frame[frame["valid_response"].astype(bool)], m, "log_rt", args.n_bootstrap, args.seed)
            for m in assoc.MEASURES]
    rows += [assoc.within_association(frame, m, "lapse_f", args.n_bootstrap, args.seed) for m in assoc.LAPSE_MEASURES]
    table = pd.DataFrame(rows)
    from scipy import stats

    table["p_normal"] = 2 * stats.norm.sf(np.abs(table["z_for_p"]))
    table["p_bh"] = np.nan
    for _outcome, idx in table.groupby("outcome").groups.items():
        table.loc[idx, "p_bh"] = fdr_correct(table.loc[idx, "p_normal"].to_numpy())[0]
    out_dir = paths.results / "single_trial_associations_mipdb" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(table, out_dir / "within_child_mipdb.csv")
    print(json.dumps({"out_dir": str(out_dir), "n_children": int(frame["subject"].nunique()),
                      "n_trials": len(frame)}, indent=1))
    print(table[["measure", "outcome", "n_children", "r_within", "r_within_ci_low", "r_within_ci_high", "p_bh"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
