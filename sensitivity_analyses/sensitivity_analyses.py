"""Do the H1 and benchmark-criterion conclusions hold under each sensitivity analysis?

Usage:
    python -m sensitivity_analyses.sensitivity_analyses --primary <run> [<run> ...] \
        --variant S-antic=<run>[,<run>...] --variant S-art=<run> ...

Every run directory is a finished (or checkpointed) benchmark run. Runs of one analysis are combined after checking
that repeated models agree. For the primary analysis and every variant the report gives, per window, target and
model (cross-validation scope), the estimate, the paired difference from demographics and, for RT, the within-child
correlation; and two checks:

* criterion 1: the best classical model in the post-target window beats demographics (paired interval below zero)
  with a within-child r interval above 0.10;
* H1: the pre-stimulus band-power model beats demographics (paired interval below zero).

Each model of each variant is also compared with the same model of the primary analysis on the trials both contain
(``decoding.paired``): identical trials for the signal-only variants, the common trials otherwise.

Writes results/sensitivity_analyses/<run>/: comparison.csv, checks.csv, paired_vs_primary.csv and report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit
from decoding.decomposition import decompose_predictions
from decoding.gate import _KEYS, criterion_1, deduplicate
from decoding.paired import KEYS as TRIAL_KEYS
from decoding.paired import compare_predictions

COLUMNS = ["window", "target", "model", "n_subjects", "n_trials", "estimate", "ci_low", "ci_high",
           "delta_vs_demographics", "delta_vs_demographics_ci_low", "delta_vs_demographics_ci_high"]


def load_analysis(run_dirs: list[str], n_bootstrap: int,
                  seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries, decompositions, predictions = [], [], []
    for run in run_dirs:
        summaries.append(pd.read_csv(Path(run) / "summary.csv"))
        preds = pd.read_parquet(Path(run) / "predictions.parquet")
        predictions.append(preds)
        decompositions.append(decompose_predictions(preds, n_bootstrap=n_bootstrap, seed=seed))
    summary = deduplicate(pd.concat(summaries, ignore_index=True), _KEYS, "estimate")
    decomposition = deduplicate(pd.concat(decompositions, ignore_index=True), ["window", "scope", "model"],
                                "nrmse_within")
    combined = pd.concat(predictions, ignore_index=True).drop_duplicates(TRIAL_KEYS).reset_index(drop=True)
    return summary, decomposition, combined


def h1_check(summary: pd.DataFrame) -> tuple[bool, str]:
    rt = summary[(summary["window"] == "prestim") & (summary["target"] == "rt_s") & (summary["scope"] == "cv")]
    rt = rt.set_index("model")
    if "spectral" not in rt.index or "delta_vs_demographics_ci_high" not in rt:
        return False, "no pre-stimulus spectral RT result"
    row = rt.loc["spectral"]
    holds = bool(row["delta_vs_demographics"] < 0 and row["delta_vs_demographics_ci_high"] < 0)
    return holds, (f"spectral delta vs demographics {row['delta_vs_demographics']:+.4f} "
                   f"[{row['delta_vs_demographics_ci_low']:+.4f}, {row['delta_vs_demographics_ci_high']:+.4f}]")


def comparison_rows(name: str, summary: pd.DataFrame, decomposition: pd.DataFrame) -> pd.DataFrame:
    table = summary[summary["scope"] == "cv"][[c for c in COLUMNS if c in summary]].copy()
    within = decomposition[decomposition["scope"] == "cv"][["window", "model", "r_within", "r_within_ci_low",
                                                            "r_within_ci_high", "nrmse_within"]]
    rt = table[table["target"] == "rt_s"].merge(within, on=["window", "model"], how="left")
    return pd.concat([rt, table[table["target"] != "rt_s"]], ignore_index=True).assign(analysis=name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary", nargs="+", required=True)
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=RUN[,RUN...]")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    analyses = {"primary": args.primary}
    for item in args.variant:
        name, _, runs = item.partition("=")
        analyses[name] = [r for r in runs.split(",") if r]
    comparisons, checks, paired = [], [], []
    primary_predictions = None
    for name, runs in analyses.items():
        summary, decomposition, predictions = load_analysis(runs, args.n_bootstrap, args.seed)
        comparisons.append(comparison_rows(name, summary, decomposition))
        if name == "primary":
            primary_predictions = predictions
        else:
            paired.append(compare_predictions(predictions, primary_predictions, args.n_bootstrap, args.seed)
                          .assign(analysis=name))
        c1 = criterion_1(summary, decomposition)
        h1, h1_detail = h1_check(summary)
        checks.append({"analysis": name, "runs": ";".join(runs), "criterion_1_holds": c1.passed,
                       "criterion_1_detail": c1.detail, "h1_holds": h1, "h1_detail": h1_detail})
    out_dir = paths.results / "sensitivity_analyses" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.concat(comparisons, ignore_index=True)
    checks = pd.DataFrame(checks)
    write_dataframe(comparison, out_dir / "comparison.csv")
    write_dataframe(checks, out_dir / "checks.csv")
    if paired:
        write_dataframe(pd.concat(paired, ignore_index=True), out_dir / "paired_vs_primary.csv")
    write_json({"analyses": analyses, "git_commit": git_commit(paths.root)}, out_dir / "report.json")
    print(json.dumps({"out_dir": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
