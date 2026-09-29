"""Benchmark criteria and the within-child decomposition, from finished benchmark runs.

Usage: python -m decoding.benchmark_summary --runs results/benchmark-v1/<ts1> results/benchmark-v1/<ts2>
       [--drop results/benchmark-v1/<ts1>=eegnet]   (models of one run to leave out)

Writes results/benchmark_summary/<run>/: gate.json, all_results.csv and decomposition.csv (the within-child and
between-child parts of the RT predictions).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.io import read_json, utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit
from decoding.decomposition import decompose_predictions
from decoding.gate import evaluate_gate, secondary_table


def parse_drop(items: list[str]) -> dict[Path, set[str]]:
    """``RUN=MODEL[,MODEL]`` pairs as {resolved run directory: models to leave out}."""
    out: dict[Path, set[str]] = {}
    for item in items:
        run, _, models = item.partition("=")
        out.setdefault(Path(run).resolve(), set()).update(m for m in models.split(",") if m)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="benchmark run directories")
    parser.add_argument("--drop", action="append", default=[], metavar="RUN=MODEL[,MODEL]",
                        help="leave these models of one run out")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    drop = parse_drop(args.drop)
    summaries, failures, provenance, decompositions = [], [], {}, []
    for run in args.runs:
        run_dir = Path(run)
        excluded = drop.get(run_dir.resolve(), set())
        summary = pd.read_csv(run_dir / "summary.csv")
        summaries.append(summary[~summary["model"].isin(excluded)])
        predictions = pd.read_parquet(run_dir / "predictions.parquet")
        predictions = predictions[~predictions["model"].isin(excluded)]
        decompositions.append(decompose_predictions(predictions, n_bootstrap=args.n_bootstrap,
                                                    seed=args.seed))
        info = read_json(run_dir / "run.json")
        failures.extend(info.get("failures", []))
        prov = run_dir / "provenance.json"
        provenance[run_dir.name] = read_json(prov) if prov.is_file() else None
    summary = pd.concat(summaries, ignore_index=True)
    decomposition = pd.concat(decompositions, ignore_index=True)
    results = evaluate_gate(summary, failures, decomposition)
    out_dir = paths.results / "benchmark_summary" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"runs": args.runs, "dropped": args.drop, "provenance": provenance, "evaluated_with_commit": git_commit(paths.root),
               "failures": [{k: v for k, v in f.items() if k != "traceback"} for f in failures],
               "criteria": [r.__dict__ for r in results],
               "gate_passed": all(r.passed for r in results)}
    write_json(payload, out_dir / "gate.json")
    write_dataframe(secondary_table(summary), out_dir / "all_results.csv")
    write_dataframe(decomposition.drop_duplicates(["window", "scope", "model"]), out_dir / "decomposition.csv")
    print(json.dumps(payload["criteria"], indent=1))
    print(f"gate passed: {payload['gate_passed']}  ->  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
