"""Is the pre-response prediction ceiling a limit of the linear decoder?

EEGNet with the benchmark settings on raw windows from the target to 1.3 s, on exactly the trials of horizon
decoder A at T = 1.3 s (margin 0.1 s), with the frozen folds and R5 held out; MIPDB and R10/R11 from a network
trained on all HBN children. Within-child r and paired differences against ridge A and the pre-stimulus decoder C
on the same trials. Needs a GPU. Writes results/prediction_horizon_eegnet/<run>/.
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
from common.schema import BenchmarkConfig, ComputeConfig, TrialConfig
from common.scripts import script_path
from decoding.base import DecoderInput
from decoding.benchmark import load_window_data
from decoding.decomposition import within_between
from decoding.leakage import LeakageAuditor
from decoding.splits import HOLDOUT_FOLD

ROOTS = {"HBN": data_location("HBN"), "MIPDB": data_location("MIPDB"),
         "R10R11": data_location("R10R11")}
SCOPE_SAMPLE = {"cv": "HBN", "holdout_R5": "HBN", "MIPDB": "MIPDB", "R10R11": "R10R11"}
HORIZON = "results/prediction_horizon"
T, MARGIN = 1.3, 0.1
SFREQ, CHAL_START_S = 250.0, 0.5
N_BOOT, SEED = 2000, 12345


class ConcatRowView:
    """Rows of [0, T] s windows: the early store (0 to 1.0 s) followed by challenge-store samples from 1.0 s to T."""

    def __init__(self, early: np.ndarray, chal: np.ndarray, rows_e: np.ndarray, rows_c: np.ndarray, t: float = T) -> None:
        self.early, self.chal = early, chal
        self.rows_e, self.rows_c = np.asarray(rows_e), np.asarray(rows_c)
        self.start = int(round((1.0 - CHAL_START_S) * SFREQ))
        self.stop = int(round((t - CHAL_START_S) * SFREQ))
        self.shape = (len(self.rows_e), early.shape[1], early.shape[2] + self.stop - self.start)

    def __len__(self) -> int:
        return len(self.rows_e)

    def __getitem__(self, item):
        a = np.asarray(self.early[self.rows_e[item]])
        b = np.asarray(self.chal[self.rows_c[item]])[..., self.start:self.stop]
        return np.concatenate([a, b], axis=-1)


def trial_rows(paths: ProjectPaths, specs: dict) -> dict:
    """Per sample: the horizon A trials at T with early and challenge row indices, and the saved A and C predictions."""
    h = pd.read_parquet(paths.root / HORIZON / "predictions.parquet")
    h = h[(h["margin"] == MARGIN) & (h["T"] == T) & h["decoder"].isin(["A", "C"])].copy()
    h["fold"] = h["fold"].fillna(-99).astype(int)          # external samples have no fold; keep them in the pivot
    wide = h.pivot_table(index=["scope", "subject", "block_key", "trial_index", "rt_s", "fold"], columns="decoder",
                         values="pred").reset_index().dropna(subset=["A", "C"])
    out = {}
    for sample in ("HBN", "MIPDB", "R10R11"):
        root = Path(ROOTS[sample])
        early = load_window_data(root / "windows-v1-early", specs["early"], SFREQ)
        chal = load_window_data(root / "windows-v1", specs["challenge"], SFREQ)
        part = wide[wide["scope"].map(SCOPE_SAMPLE) == sample]
        keys = ["subject", "block_key", "trial_index"]
        part = part.merge(early.index[[*keys, "row"]].rename(columns={"row": "row_e"}), on=keys, how="left", validate="one_to_one")
        part = part.merge(chal.index[[*keys, "row"]].rename(columns={"row": "row_c"}), on=keys, how="left", validate="one_to_one")
        assert part[["row_e", "row_c"]].notna().all().all()
        out[sample] = {"rows": part.sort_values("row_e").reset_index(drop=True), "early": early.array, "chal": chal.array}
    return out


def view(sample: dict, rows: pd.DataFrame) -> ConcatRowView:
    return ConcatRowView(sample["early"], sample["chal"], rows["row_e"].to_numpy(int), rows["row_c"].to_numpy(int))


def make_net(n_channels: int, n_times: int, bcfg: BenchmarkConfig, compute: ComputeConfig):
    from decoding.eegnet import EEGNetRegressor

    e = bcfg.eegnet
    return EEGNetRegressor(n_channels, n_times, compute=compute, max_epochs=e.max_epochs, batch_size=e.batch_size,
                           lr=e.lr, weight_decay=e.weight_decay, patience=e.patience, seeds=tuple(e.seeds),
                           decimate=e.decimate, kernel_length=e.kernel_length, fast_spatial=e.fast_spatial,
                           autocast=e.autocast_bf16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit-children", type=int, default=None, help="smoke tests only")
    parser.add_argument("--max-epochs", type=int, default=None, help="smoke tests only")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None, help="smoke tests: cpu")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    commit = git_commit(paths.root)
    bcfg = load_config(paths.root / "decoding" / "benchmark.yaml", BenchmarkConfig)
    if args.max_epochs:
        bcfg = bcfg.model_copy(update={"eegnet": bcfg.eegnet.model_copy(update={"max_epochs": args.max_epochs})})
    compute = load_config(paths.root / "common" / "compute.yaml", ComputeConfig)
    if args.device:
        compute = compute.model_copy(update={"device": args.device})
    specs = {"early": load_config(paths.root / "features" / "windows_early.yaml", TrialConfig).windows[0]}
    for w in load_config(paths.root / "features" / "trials_and_windows.yaml", TrialConfig).windows:
        specs[w.name] = w
    out_dir = Path(args.out_dir) if args.out_dir else paths.results / "prediction_horizon_eegnet" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, out_dir / "provenance.json")
    data = trial_rows(paths, specs)
    hbn = data["HBN"]["rows"]
    if args.limit_children:
        keep = set(sorted(hbn["subject"].unique())[:args.limit_children])
        hbn = data["HBN"]["rows"] = hbn[hbn["subject"].isin(keep)].reset_index(drop=True)
    shape = view(data["HBN"], hbn.iloc[:1]).shape
    fold = hbn["fold"].astype(int).to_numpy()
    frames = []
    for f in [*sorted(int(v) for v in np.unique(fold) if v != HOLDOUT_FOLD), HOLDOUT_FOLD]:
        test, train = fold == f, (fold != f) & (fold != HOLDOUT_FOLD)
        LeakageAuditor.assert_rows_grouped(hbn, train, test)
        net = make_net(shape[1], shape[2], bcfg, compute)
        tr, te = hbn[train].reset_index(drop=True), hbn[test].reset_index(drop=True)
        net.fit(DecoderInput(view(data["HBN"], tr), tr["subject"].to_numpy(), None), tr["rt_s"].to_numpy(float))
        frames.append(te.assign(eegnet=net.predict(DecoderInput(view(data["HBN"], te), te["subject"].to_numpy(), None))))
        print(json.dumps({"fold": f, "n_test": int(test.sum())}), flush=True)
    net = make_net(shape[1], shape[2], bcfg, compute)
    net.fit(DecoderInput(view(data["HBN"], hbn), hbn["subject"].to_numpy(), None), hbn["rt_s"].to_numpy(float))
    for sample in ("MIPDB", "R10R11"):
        rows = data[sample]["rows"]
        frames.append(rows.assign(eegnet=net.predict(DecoderInput(view(data[sample], rows), rows["subject"].to_numpy(), None))))
    preds = pd.concat(frames, ignore_index=True)
    write_dataframe(preds[["scope", "subject", "block_key", "trial_index", "rt_s", "fold", "A", "C", "eegnet"]],
                    out_dir / "predictions.parquet")
    horizon = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location(
        "prediction_horizon", script_path("prediction_horizon")))
    spec.loader.exec_module(horizon)
    rows = []
    for scope, part in preds.groupby("scope", sort=True):
        y, s = part["rt_s"].to_numpy(float), part["subject"].to_numpy()
        r = within_between(y, part["eegnet"].to_numpy(float), s, n_bootstrap=N_BOOT, seed=SEED)
        row = {"scope": scope, "n_trials": len(part), "n_children": r["n_children"], "r_eegnet": r["r_within"],
               "r_eegnet_ci_low": r["r_within_ci_low"], "r_eegnet_ci_high": r["r_within_ci_high"]}
        for other in ("A", "C"):
            d = horizon.paired_bootstrap(y, part["eegnet"].to_numpy(float), part[other].to_numpy(float), s, "difference",
                                         N_BOOT, SEED)
            row.update({f"minus_{other}": d["estimate"], f"minus_{other}_ci_low": d["ci_low"], f"minus_{other}_ci_high": d["ci_high"]})
        rows.append(row)
    table = pd.DataFrame(rows)
    write_dataframe(table, out_dir / "summary.csv")
    t = table.set_index("scope")
    better = all(s in t.index and t.loc[s, "minus_A_ci_low"] > 0 for s in ("cv", "holdout_R5"))
    summary = {"git_commit": commit, "rule": "EEGNet reported as the best pre-response estimate" if better
               else "ridge estimate stands; EEGNet is a supplementary robustness line", "eegnet_beats_ridge_cv_r5": better}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
