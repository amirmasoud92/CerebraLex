"""How well does each EEG representation identify the child?

For a random subset of children, every trial's nearest neighbour (leave-one-out, standardised features reduced by
PCA) is checked for coming from the same child. Accuracy far above chance means the representation identifies
children, which a decoder could exploit if children were split across folds; the benchmark prevents that with
child-grouped folds, and the trait analysis is repeated on features standardised within child.

Every feature set (band power, band power plus aperiodic, Riemannian tangent vectors, 100 ms time course) is tested
as used by the benchmark and after standardisation within child (``transform`` column). REVE embeddings are never
tested: their licence forbids re-identification.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.log import configure_logging
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import DerivativesConfig, PreprocessingConfig, TrialConfig
from decoding.benchmark import load_window_data
from decoding.leakage import LeakageAuditor
from features.normalisation import standardise_within_groups


def tangent_vectors(upper: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """Same projection and tangent map as the benchmark's Riemannian baseline (GPU when available)."""
    import torch

    from decoding.baselines import TangentSpaceRidge

    mapper = TangentSpaceRidge(projection_rank=rank, mean_subsample=2000, seed=seed,
                               device="cuda" if torch.cuda.is_available() else None)
    return mapper.fit_mapping(upper).tangent_vectors(upper)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-name", default="windows-v1")
    parser.add_argument("--n-subjects", type=int, default=300)
    parser.add_argument("--pca", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    log = configure_logging("child-identification", paths.logs)
    commit = git_commit(paths.root)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    preproc = load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    trial_cfg = load_config(paths.root / "features" / "trials_and_windows.yaml", TrialConfig)
    rng = np.random.default_rng(args.seed)
    rows = []
    for spec in trial_cfg.windows:
        window = load_window_data(deriv.root / args.store_name, spec, preproc.resample_hz)
        cache_dir = deriv.root / "features" / args.store_name / f"{spec.name}_{window.fingerprint[:12]}"
        index = window.index[window.index["window_valid"] & ~window.index["artifact_window"].astype(bool)]
        subjects = rng.choice(np.array(sorted(index["subject"].unique())), size=args.n_subjects, replace=False)
        sel = index[index["subject"].isin(subjects)]
        row_idx = sel["row"].to_numpy()
        sets = {}
        spectral_path = cache_dir / "spectral.npy"
        if spectral_path.is_file():
            sets["spectral"] = np.asarray(np.load(spectral_path, mmap_mode="r")[row_idx])
        aperiodic_path = cache_dir / "aperiodic.npy"
        if aperiodic_path.is_file() and "spectral" in sets:
            sets["spectral_aperiodic"] = np.concatenate(
                [sets["spectral"], np.nan_to_num(np.asarray(np.load(aperiodic_path, mmap_mode="r")[row_idx]))], axis=1)
        cov_path = cache_dir / "covariance_upper.npy"
        if cov_path.is_file():
            sets["riemann"] = tangent_vectors(np.asarray(np.load(cov_path, mmap_mode="r")[row_idx]), 100, args.seed)
        timecourse_path = cache_dir / "timecourse_100ms.npy"
        if timecourse_path.is_file():
            sets["timecourse"] = np.asarray(np.load(timecourse_path, mmap_mode="r")[row_idx])
        subjects_sel = sel["subject"].to_numpy()
        for name, x in sets.items():
            for transform in ("none", "child_standardised"):
                features = x if transform == "none" else np.nan_to_num(standardise_within_groups(x, subjects_sel))
                result = LeakageAuditor.fingerprint_accuracy(features, subjects_sel, n_components=args.pca,
                                                             seed=args.seed)
                cross = LeakageAuditor.cross_block_fingerprint_accuracy(
                    features, subjects_sel, sel["block_key"].to_numpy(), n_components=args.pca, seed=args.seed)
                result.update({"window": spec.name, "feature_set": name, "transform": transform,
                               "pca_components": args.pca, "cross_block_accuracy": cross["accuracy"],
                               "cross_block_chance": cross["chance"], "cross_block_n_scored": cross["n_scored"]})
                log.info("%s %s (%s): accuracy %.3f (chance %.4f); cross-block %.3f (chance %.4f)", spec.name,
                         name, transform, result["accuracy"], result["chance"], cross["accuracy"], cross["chance"])
                rows.append(result)
    out_dir = paths.results / "child_identification" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    write_dataframe(frame, out_dir / "fingerprint_accuracy.csv")
    write_json({**vars(args), "git_commit_at_launch": commit}, out_dir / "args.json")
    print(json.dumps(frame.to_dict(orient="records"), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
