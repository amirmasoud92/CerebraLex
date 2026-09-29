"""Why do the discovery and held-out slow/fast build-up ratios differ? (post hoc, exploratory)

(i)   Between-release heterogeneity: E1 (the ratio of child means) in each HBN release, a DerSimonian-Laird
      random-effects meta-analysis of the ratios on the linear scale (on the log scale bootstrap ratios of small
      releases reach 0), and the 95% predictive interval for the observed estimate of a new two-release sample
      (R10 and R11 pooled).
(ii)  Age: E1 in HBN reweighted to the held-out age distribution.
(iii) The calibration grid of cpp_onset_rate_calibration.py on R10/R11: which onset shares w fit the held-out
      ratio, and which fit both samples.
Reads the per-child tertile tables of a buildup_ratio.py run and the calibration summaries. Writes
results/buildup_ratio_heterogeneity/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import t as student_t

from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit
from common.scripts import script_path
from decoding.benchmark import demographics_table

ROOTS = {"HBN": data_location("HBN"), "MIPDB": data_location("MIPDB"),
         "R10R11": data_location("R10R11")}
PEOPLE = {"MIPDB": f"{data_location('MIPDB')}/manifest/participants.csv",
          "R10R11": f"{data_location('R10R11')}/manifest/participants.csv"}
HBN_GRID = "results/cpp_onset_rate_calibration/hbn_grid"
AGE_EDGES = np.arange(5.0, 23.0 + 1e-9, 2.0)
N_BOOT, SEED = 2000, 12345


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ratio(slow: np.ndarray, fast: np.ndarray, weights: np.ndarray | None = None) -> float:
    w = np.ones(len(slow)) if weights is None else weights
    return float((w @ slow) / (w @ fast))


def boot_ratio(frame: pd.DataFrame, weights: np.ndarray | None = None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """E1 (optionally weighted) with a child bootstrap: percentile interval and the SD of the log ratio."""
    slow, fast = frame["slow_rate_uv_per_s"].to_numpy(float), frame["fast_rate_uv_per_s"].to_numpy(float)
    w = np.ones(len(frame)) if weights is None else np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        for b in range(n_boot):
            i = rng.integers(0, len(frame), len(frame))
            draws[b] = ratio(slow[i], fast[i], w[i])
    finite = draws[np.isfinite(draws)]                  # a resample of zero-weight children only has no ratio
    low, high = np.percentile(finite, [2.5, 97.5])
    return {"estimate": ratio(slow, fast, w), "ci_low": float(low), "ci_high": float(high), "se": float(finite.std(ddof=1)),
            "se_log": float(np.log(finite).std(ddof=1)) if (finite > 0).all() else float("nan"),
            "n_children": len(frame), "n_boot_undefined": int(n_boot - len(finite))}


def dersimonian_laird(y: np.ndarray, v: np.ndarray) -> dict:
    w = 1.0 / v
    mean_fixed = (w @ y) / w.sum()
    q = float(w @ (y - mean_fixed) ** 2)
    k = len(y)
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = 1.0 / (v + tau2)
    mu = float((ws @ y) / ws.sum())
    return {"k": k, "mu": mu, "se_mu": float(np.sqrt(1.0 / ws.sum())), "tau2": tau2, "tau": float(np.sqrt(tau2)),
            "Q": q, "Q_p": float(chi2.sf(q, k - 1)), "I2": float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0}


def predictive_interval(meta: dict, between_var_factor: float, se_new: float) -> tuple[float, float]:
    """95% interval for the observed ratio of a new sample (Higgins, Thompson and Spiegelhalter 2009), with the
    between-release variance scaled by the sample's release composition and its own sampling variance added."""
    half = student_t.ppf(0.975, meta["k"] - 2) * np.sqrt(meta["tau2"] * between_var_factor + meta["se_mu"] ** 2 + se_new ** 2)
    return float(meta["mu"] - half), float(meta["mu"] + half)


def with_release(children: pd.DataFrame, root: str) -> pd.DataFrame:
    releases = pd.read_parquet(Path(root) / "trials-v1" / "trials_ccd.parquet", columns=["subject", "release"])
    return children.merge(releases.drop_duplicates("subject"), on="subject", how="left", validate="many_to_one")


def heterogeneity(children: dict[str, pd.DataFrame], ratio_set) -> tuple[pd.DataFrame, dict]:
    hbn = with_release(ratio_set(children["HBN"]), ROOTS["HBN"])
    conf = with_release(ratio_set(children["R10R11"]), ROOTS["R10R11"])
    assert hbn["release"].notna().all() and conf["release"].notna().all()
    rows = [{"sample": "HBN", "release": rel, **boot_ratio(part)} for rel, part in hbn.groupby("release", sort=True)]
    table = pd.DataFrame(rows)
    meta = dersimonian_laird(table["estimate"].to_numpy(float), table["se"].to_numpy(float) ** 2)
    meta["scale"] = "linear ratio"
    meta["pooled_ratio"] = meta["mu"]
    meta["pooled_ratio_ci"] = [meta["mu"] - 1.96 * meta["se_mu"], meta["mu"] + 1.96 * meta["se_mu"]]
    shares = conf["release"].value_counts(normalize=True)
    pooled = boot_ratio(conf)
    pi = predictive_interval(meta, float((shares ** 2).sum()), pooled["se"])
    rows = [{"sample": "R10R11", "release": "R10+R11", **pooled, "pi_low": pi[0], "pi_high": pi[1],
             "inside_pi": bool(pi[0] <= pooled["estimate"] <= pi[1])}]
    for rel, part in conf.groupby("release", sort=True):
        b = boot_ratio(part)
        lo, hi = predictive_interval(meta, 1.0, b["se"])
        rows.append({"sample": "R10R11", "release": rel, **b, "pi_low": lo, "pi_high": hi, "inside_pi": bool(lo <= b["estimate"] <= hi)})
    mip = boot_ratio(ratio_set(children["MIPDB"]))
    lo, hi = predictive_interval(meta, 1.0, mip["se"])
    rows.append({"sample": "MIPDB", "release": "MIPDB", **mip, "pi_low": lo, "pi_high": hi, "inside_pi": bool(lo <= mip["estimate"] <= hi)})
    table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    meta["held_out_release_shares"] = shares.round(4).to_dict()
    meta["rule_compatible_with_heterogeneity"] = bool(rows[0]["inside_pi"])
    return table, meta


def age_reweighted(children: dict[str, pd.DataFrame], ratio_set, paths: ProjectPaths) -> dict:
    hbn_people = demographics_table(pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False))
    conf_people = pd.read_csv(PEOPLE["R10R11"], dtype={"subject": str})
    hbn = ratio_set(children["HBN"]).merge(hbn_people[["subject", "age"]], on="subject", how="left")
    conf = ratio_set(children["R10R11"]).merge(conf_people[["subject", "age"]], on="subject", how="left")
    for frame in (hbn, conf):
        age = pd.to_numeric(frame["age"], errors="coerce")
        frame["bin"] = pd.cut(age, AGE_EDGES, right=False, labels=False)
    out = {"n_hbn_without_bin": int(hbn["bin"].isna().sum()), "n_conf_without_bin": int(conf["bin"].isna().sum())}
    hbn, conf = hbn.dropna(subset=["bin"]), conf.dropna(subset=["bin"])
    hbn_share = hbn["bin"].value_counts(normalize=True)
    conf_counts = conf["bin"].value_counts()
    out["n_conf_in_bins_without_hbn"] = int(conf_counts[~conf_counts.index.isin(hbn_share.index)].sum())
    conf_share = conf_counts[conf_counts.index.isin(hbn_share.index)]
    conf_share = conf_share / conf_share.sum()
    weights = hbn["bin"].map(conf_share / hbn_share).fillna(0.0).to_numpy(float)
    out["age_summary"] = {name: {"median": float(pd.to_numeric(f["age"]).median()),
                                 "q1": float(pd.to_numeric(f["age"]).quantile(0.25)),
                                 "q3": float(pd.to_numeric(f["age"]).quantile(0.75)), "n": len(f)}
                          for name, f in (("HBN", hbn), ("R10R11", conf))}
    out["bins"] = {f"{AGE_EDGES[int(b)]:g}-{AGE_EDGES[int(b)] + 2:g}": {"hbn_share": float(hbn_share.get(b, 0.0)),
                                                                        "conf_share": float(conf_share.get(b, 0.0))}
                   for b in sorted(set(hbn_share.index) | set(conf_counts.index))}
    out["hbn_unweighted"] = boot_ratio(hbn)
    out["hbn_age_weighted"] = boot_ratio(hbn, weights)
    out["conf"] = boot_ratio(conf)
    out["rule_age_possible_explanation"] = bool(out["hbn_age_weighted"]["ci_low"] <= 0.868 <= out["hbn_age_weighted"]["ci_high"])
    return out


def consistent_w(summary: dict) -> dict:
    """Per realistic-noise setting, the w whose simulated share and ratio lie inside the observed intervals, and the w
    whose simulated ratio alone does."""
    obs = summary["observed"]
    both, ratio_only = {}, {}
    for key, s in summary["simulated"].items():
        if not key.startswith("surrogate"):
            continue
        setting, w = key.rsplit("_w", 1)
        both.setdefault(setting, [])
        ratio_only.setdefault(setting, [])
        if s["consistent"]:
            both[setting].append(float(w))
        if obs["rate_ratio_ci"][0] <= s["rate_ratio"] <= obs["rate_ratio_ci"][1]:
            ratio_only[setting].append(float(w))
    return {"observed": {k: obs[k] for k in ("n_children", "share", "share_ci", "rate_ratio", "rate_ratio_ci")},
            "consistent_both": both, "consistent_ratio_only": ratio_only,
            "union_both": sorted({w for ws in both.values() for w in ws}),
            "union_ratio_only": sorted({w for ws in ratio_only.values() for w in ws})}


def calibration(paths: ProjectPaths, confirm_cal: Path) -> dict:
    hbn = consistent_w(json.loads((paths.root / HBN_GRID / "summary.json").read_text(encoding="utf-8")))
    conf_summary = json.loads((confirm_cal / "summary.json").read_text(encoding="utf-8"))
    assert Path(conf_summary["root"]).as_posix().rstrip("/").lower() == ROOTS["R10R11"].lower()
    conf = consistent_w(conf_summary)
    inter = sorted(set(hbn["union_both"]) & set(conf["union_both"]))
    return {"hbn_grid": HBN_GRID, "confirm_cal": str(confirm_cal), "HBN": hbn, "R10R11": conf, "intersection": inter,
            "rule": "only accounts with w in the intersection fit both samples" if inter
            else "the two samples support different mixtures of onset and rate",
            "R10R11_pure_onset_excluded": 1.0 not in conf["union_both"],
            "R10R11_pure_rate_excluded": 0.0 not in conf["union_both"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--buildup-ratio-run", required=True, help="run directory of buildup_ratio.py")
    parser.add_argument("--confirm-cal", default=None, help="run directory of cpp_onset_rate_calibration.py on R10/R11")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    commit = git_commit(paths.root)
    run = Path(args.buildup_ratio_run)
    ratio_set = sibling("buildup_ratio").ratio_set
    children = {name: pd.read_csv(run / f"onset_rate_children_{name}.csv", dtype={"subject": str}) for name in ROOTS}
    out_dir = Path(args.out_dir) if args.out_dir else paths.results / "buildup_ratio_heterogeneity" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, out_dir / "provenance.json")
    table, meta = heterogeneity(children, ratio_set)
    write_dataframe(table, out_dir / "release_ratios.csv")
    age = age_reweighted(children, ratio_set, paths)
    summary = {"git_commit": commit, "buildup_ratio_run": str(run), "meta": meta, "age": age}
    if args.confirm_cal:
        summary["calibration"] = calibration(paths, Path(args.confirm_cal))
    write_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
