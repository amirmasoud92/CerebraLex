"""Is the developmental null for the CPP build-up an amplitude-scaling artefact? (post hoc)

Each child's CPP measures (child_level_measures.csv of single_trial_associations.py: mean response-locked slope and
amplitude at the response) are divided by the child's pre-target CPP-cluster amplitude: the root mean square, over
trials and time, of the cluster mean from -1.0 to 0 s before the target after the 8 Hz low-pass and per-trial
demeaning (same cohort and trials). For raw and normalised measures: the age effect (natural spline, 4 df, plus sex;
F test) and the partial correlation across children with EZ drift rate given the age spline and sex (child
bootstrap, 2,000). Writes results/cpp_normalised_amplitude/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import DerivativesConfig, TrialConfig
from common.scripts import script_path
from features.clusters import cluster_channels
from features.windows import WindowStore

ASSOCIATIONS_RUN = "results/single_trial_associations"


def _associations():
    spec = importlib.util.spec_from_file_location("single_trial_associations", script_path("single_trial_associations"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pretarget_rms(cpp_uv: np.ndarray, zero: int) -> float:
    """RMS over trials and time of the demeaned pre-target segment (``cpp_uv``: trials x samples, filtered)."""
    pre = cpp_uv[:, :zero]
    pre = pre - pre.mean(axis=1, keepdims=True)
    return float(np.sqrt(np.mean(pre ** 2)))


def partial_r(y: np.ndarray, x: np.ndarray, covariates: np.ndarray) -> float:
    """Correlation of y and x after both are residualised on the covariates (with an intercept)."""
    c = np.column_stack([np.ones(len(y)), covariates])
    ry = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
    rx = x - c @ np.linalg.lstsq(c, x, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    assoc = _associations()
    paths = ProjectPaths.discover()
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    spec = load_config(paths.root / "features" / "epochs.yaml", TrialConfig).windows[0]
    array, index, meta = WindowStore(deriv.root / "epochs-v1", spec, 250.0).open()
    sfreq, zero = float(meta["sfreq"]), int(round(-spec.tmin_s * float(meta["sfreq"])))
    measures = pd.read_parquet(deriv.root / "features" / "epochs-v1" / "trial_measures.parquet")
    for key in ("block_key", "trial_in_block"):
        if not (index[key].to_numpy() == measures[key].to_numpy()).all():
            raise ValueError(f"trial measures and epoch store are not aligned on {key}")
    usable = measures["window_valid"].astype(bool) & ~measures["artifact_window"].astype(bool)
    children = pd.read_csv(paths.root / ASSOCIATIONS_RUN / "child_level_measures.csv")
    rows = index[usable.to_numpy() & measures["subject"].isin(children["subject"]).to_numpy()]
    picks = [meta["channels"].index(ch) for ch in cluster_channels()["cpp"]]
    sos = butter(4, 8.0, btype="low", fs=sfreq, output="sos")
    rms = {}
    for subject, part in rows.sort_values("row").groupby("subject", sort=True):
        cpp = sosfiltfilt(sos, np.asarray(array[part["row"].to_numpy()][:, picks], dtype=np.float64).mean(axis=1) * 1e6,
                          axis=-1)
        rms[subject] = pretarget_rms(cpp, zero)
    children["pretarget_rms_uv"] = children["subject"].map(rms)
    for m in ("cpp_slope_resp", "cpp_amp_resp"):
        children[f"{m}_norm"] = children[m] / children["pretarget_rms_uv"]
    grid = np.arange(5.5, 21.5, 0.5)
    results, curves = [], []
    rng = np.random.default_rng(args.seed)
    for m in ("pretarget_rms_uv", "cpp_slope_resp", "cpp_slope_resp_norm", "cpp_amp_resp", "cpp_amp_resp_norm"):
        s, c = assoc.age_trajectory(children, m, grid, args.n_bootstrap, args.seed)
        part = children.dropna(subset=[m, "ez_drift", "age", "sex_male"])
        knots = np.quantile(part["age"], np.linspace(0, 1, 5))
        cov = np.column_stack([assoc.natural_basis(part["age"].to_numpy(), knots), part["sex_male"]])
        y, x = part["ez_drift"].to_numpy(float), part[m].to_numpy(float)
        r = partial_r(y, x, cov)
        draws = [partial_r(y[i], x[i], cov[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(args.n_bootstrap))]
        low, high = np.percentile(draws, [2.5, 97.5])
        results.append({**s, "partial_r_ez_drift": r, "partial_r_ci_low": float(low), "partial_r_ci_high": float(high),
                        "n_children_drift": len(part)})
        curves.append(c)
    out_dir = paths.results / "cpp_normalised_amplitude" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results)
    write_dataframe(table, out_dir / "age_and_drift.csv")
    write_dataframe(pd.concat(curves, ignore_index=True), out_dir / "normalised_age_trajectories.csv")
    write_dataframe(children[["subject", "age", "sex_male", "pretarget_rms_uv", "cpp_slope_resp", "cpp_slope_resp_norm",
                              "cpp_amp_resp", "cpp_amp_resp_norm", "ez_drift"]], out_dir / "children.csv")
    write_json({"git_commit": git_commit(paths.root), "associations_run": ASSOCIATIONS_RUN}, out_dir / "provenance.json")
    print(json.dumps({"out_dir": str(out_dir)}))
    print(table[["measure", "n_children", "r2_age_given_sex", "p", "partial_r_ez_drift", "partial_r_ci_low",
                 "partial_r_ci_high"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
