"""Where and when does the within-child RT information lie?

Descriptive maps of the pooled within-child correlation between each benchmark feature and log RT (valid responses;
features and log RT centred within child):

* ``early``: time courses, channel x 100 ms bin, target to +1.0 s, trials with RT above 1.1 s;
* ``challenge``: time courses, +0.5 to +2.5 s (contains the response);
* ``prestim``: log band power, channel x band (the benchmark's band-power features).

Computed separately in the HBN cross-validation children, the R5 children and the external children, with the
benchmark cohort rules. Replication: the correlation of each map across its cells between cross-validation and R5
and between cross-validation and the external sample. Writes results/rt_information_maps/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit
from common.schema import (
    BenchmarkConfig,
    ComputeConfig,
    DerivativesConfig,
    PreprocessingConfig,
    TrialConfig,
)
from common.scripts import script_path
from decoding.benchmark import (
    BenchmarkRunner,
    FeatureCache,
    demographics_table,
    load_window_data,
)
from decoding.splits import HOLDOUT_FOLD, SplitAssignment
from features.spectral import DEFAULT_BANDS

MAPS = {  # map: (window config, store, feature, minimum RT in s)
    "early": ("windows_early.yaml", "windows-v1-early", "timecourse", 1.1),
    "challenge": ("trials_and_windows.yaml", "windows-v1", "timecourse", None),
    "prestim": ("trials_and_windows.yaml", "windows-v1", "spectral", None),
}


def _external_validation():
    spec = importlib.util.spec_from_file_location("external_validation",
                                                  script_path("external_validation"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def within_child_r(x: np.ndarray, y: np.ndarray, subjects: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Pooled within-child correlation of every column of ``x`` with ``y``."""
    order = np.argsort(subjects, kind="stable")
    s, ys = subjects[order], np.asarray(y, dtype=np.float64)[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    counts = np.diff(np.r_[starts, len(s)])
    yc = ys - np.repeat(np.add.reduceat(ys, starts) / counts, counts)
    out = np.empty(x.shape[1])
    for c0 in range(0, x.shape[1], chunk):
        xs = np.asarray(x[order, c0:c0 + chunk], dtype=np.float64)
        xc = xs - np.repeat(np.add.reduceat(xs, starts, axis=0) / counts[:, None], counts, axis=0)
        out[c0:c0 + chunk] = (xc * yc[:, None]).sum(axis=0) / np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    return out


def cell_labels(feature: str, n_columns: int, n_channels: int) -> pd.DataFrame:
    if feature == "timecourse":                      # channel-major: column = channel x n_bins + bin
        n_bins = n_columns // n_channels
        return pd.DataFrame({"channel": np.repeat(np.arange(n_channels), n_bins), "bin": np.tile(np.arange(n_bins), n_channels)})
    bands = [*DEFAULT_BANDS, "total"]                # band-major, then total 1-45 Hz power (SpectralFeatures.transform)
    return pd.DataFrame({"channel": np.tile(np.arange(n_channels), len(bands)), "band": np.repeat(bands, n_channels)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-config", default="decoding/benchmark.yaml")
    parser.add_argument("--external-root", default=data_location("MIPDB"))
    args = parser.parse_args()
    external = _external_validation()
    paths = ProjectPaths.discover()
    bcfg = load_config(paths.root / args.benchmark_config, BenchmarkConfig)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    preproc = load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    compute = load_config(paths.root / "common" / "compute.yaml", ComputeConfig)
    ext_root = Path(args.external_root)
    split = SplitAssignment.load(deriv.root / "splits" / bcfg.split_file)
    participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
    people = pd.read_csv(ext_root / "manifest" / "participants.csv")
    import mne

    positions = np.array(list(mne.channels.make_standard_montage(preproc.montage).get_positions()["ch_pos"].values()))
    ch_names = list(mne.channels.make_standard_montage(preproc.montage).ch_names)
    trials = pd.read_parquet(deriv.root / "trials-v1" / "trials_ccd.parquet")
    out_dir = paths.results / "rt_information_maps" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    cells, summary = [], {"git_commit": git_commit(paths.root), "argv": sys.argv, "maps": {}}
    for name, (config, store, feature, min_rt) in MAPS.items():
        spec = next(w for w in load_config(paths.root / "features" / config, TrialConfig).windows if w.name == name)
        hbn = load_window_data(deriv.root / store, spec, preproc.resample_hz)
        ext = load_window_data(ext_root / store, spec, preproc.resample_hz)
        bench = BenchmarkRunner(bcfg, compute, trials, {spec.name: hbn}, split, demographics_table(participants),
                                positions, out_dir, deriv.root / "features" / store)
        sources = {
            "hbn": (FeatureCache(bench.feature_root, hbn, bcfg.aperiodic_regions, positions, None,
                                 n_jobs=bcfg.feature_n_jobs), bench.cohort(hbn)),
            "mipdb": (FeatureCache(ext_root / "features" / store, ext, bcfg.aperiodic_regions, positions, None,
                                   n_jobs=bcfg.feature_n_jobs), external.external_cohort(ext, people, bcfg)),
        }
        maps, frames = {}, {}
        for source, (cache, cohort) in sources.items():
            x_all = cache.timecourse(bcfg.timecourse_bin_ms) if feature == "timecourse" else cache.spectral()
            rows = cohort[cohort["valid_response"].astype(bool)]
            if min_rt is not None:
                rows = rows[rows["rt_s"] > min_rt]
            scopes = ({"cv": rows[rows["fold"] != HOLDOUT_FOLD], "holdout_R5": rows[rows["fold"] == HOLDOUT_FOLD]}
                      if source == "hbn" else {"mipdb": rows})
            for scope, part in scopes.items():
                r = within_child_r(x_all[part["row"].to_numpy()], np.log(part["rt_s"].to_numpy(float)),
                                   part["subject"].to_numpy())
                maps[scope] = r
                labels = cell_labels(feature, len(r), len(ch_names))
                frames[scope] = labels.assign(map=name, scope=scope, r=r, n_trials=len(part),
                                              n_children=part["subject"].nunique(),
                                              channel_name=[ch_names[c] for c in labels["channel"]])
                cells.append(frames[scope])
                print(f"{name} {scope}: {len(part)} trials, {part['subject'].nunique()} children", flush=True)
        top = frames["cv"].assign(abs_r=lambda f: f["r"].abs()).nlargest(10, "abs_r")
        summary["maps"][name] = {
            "replication_cv_r5": float(np.corrcoef(maps["cv"], maps["holdout_R5"])[0, 1]),
            "replication_cv_mipdb": float(np.corrcoef(maps["cv"], maps["mipdb"])[0, 1]),
            "max_abs_r": {k: float(np.abs(v).max()) for k, v in maps.items()},
            "top_cells_cv": top.drop(columns=["abs_r"]).to_dict(orient="records"),
        }
    write_dataframe(pd.concat(cells, ignore_index=True), out_dir / "maps.csv")
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **{k: {kk: vv for kk, vv in v.items() if kk != "top_cells_cv"}
                                                   for k, v in summary["maps"].items()}}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
