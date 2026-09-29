"""Latent state models fitted within folds, evaluated by RT and lapse decoding.

For every window and every frozen fold (ten CV folds and the R5 hold-out):

1. Features of the benchmark cohort's trials are centred within each block, standardised and
   reduced by PCA, all fitted on training children only.
2. The number of HMM states and of LGSSM latent dimensions is chosen by held-out log-likelihood
   per trial on training children held out by subject hash (never on test children).
3. The selected models are refitted on all training children; every child's sequences are then
   filtered causally (a trial's state uses that trial and earlier ones only).
4. The benchmark's ridge decoder (unpenalised demographic block, inner grouped CV) predicts RT
   and lapse from each feature set: the raw components of the current trial, a causal
   exponentially smoothed version (the trivial dynamic baseline), the HMM state posteriors, the
   LGSSM state means, and raw plus each state representation.

Psychopathology is never read here.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from common.hashing import hash_to_unit_interval, stable_subject_hash
from common.schema import StateModelConfig
from decoding.base import DecoderInput
from decoding.baselines import GroupedRidge
from decoding.benchmark import BenchmarkRunner, FeatureCache
from decoding.leakage import LeakageAuditor
from decoding.splits import HOLDOUT_FOLD
from state_models_and_trait_moderation.hmm import GaussianHMM
from state_models_and_trait_moderation.lgssm import LinearGaussianSSM
from state_models_and_trait_moderation.sequences import block_centre, build_sequences, ewma

_LOG = logging.getLogger(__name__)
FEATURE_SETS = ("raw", "ewma", "hmm", "lgssm", "raw_hmm", "raw_lgssm")


class StateModelRunner:
    def __init__(self, cfg: StateModelConfig, bench: BenchmarkRunner) -> None:
        self.cfg = cfg
        self.bench = bench

    def _features(self, cache: FeatureCache, rows: pd.DataFrame) -> np.ndarray:
        idx = rows["row"].to_numpy()
        if self.cfg.features == "spectral":
            return np.asarray(cache.spectral()[idx], dtype=np.float64)
        return np.asarray(cache.reve("attention_pool_all_layers")[idx], dtype=np.float64)

    def _inner_validation(self, subjects: np.ndarray) -> np.ndarray:
        salt = f"{self.cfg.name}-inner"
        scores = np.array([hash_to_unit_interval(stable_subject_hash(s, salt)) for s in subjects])
        return scores < self.cfg.inner_validation_fraction

    def _select(self, z: np.ndarray, rows: pd.DataFrame, train: np.ndarray) -> tuple[int, int, list[dict]]:
        subjects = rows["subject"].to_numpy()
        train_subjects = np.unique(subjects[train])
        val_subjects = set(train_subjects[self._inner_validation(train_subjects)])
        inner_val = train & np.isin(subjects, list(val_subjects))
        inner_fit = train & ~inner_val
        LeakageAuditor.assert_disjoint(subjects[inner_fit], subjects[inner_val])
        fit_seq = build_sequences(rows[inner_fit], z[inner_fit])
        val_seq = build_sequences(rows[inner_val], z[inner_val])
        records = []
        for k in self.cfg.hmm_states:
            model = GaussianHMM(k, n_restarts=self.cfg.hmm_restarts, max_iter=self.cfg.max_iter,
                                seed=self.cfg.seed).fit(fit_seq.y, fit_seq.mask, fit_seq.lengths)
            records.append({"model": "hmm", "size": k,
                            "val_loglik_per_trial": model.loglik_per_trial(val_seq.y, val_seq.mask, val_seq.lengths),
                            "restart_logliks": model.restart_logliks_})
        for q in self.cfg.lgssm_latents:
            model = LinearGaussianSSM(q, max_iter=self.cfg.max_iter, seed=self.cfg.seed).fit(
                fit_seq.y, fit_seq.mask, fit_seq.lengths)
            records.append({"model": "lgssm", "size": q,
                            "val_loglik_per_trial": model.loglik_per_trial(val_seq.y, val_seq.mask, val_seq.lengths),
                            "restart_logliks": None})
        table = pd.DataFrame(records)
        best = {m: int(g.loc[g["val_loglik_per_trial"].idxmax(), "size"]) for m, g in table.groupby("model")}
        return best["hmm"], best["lgssm"], records

    def _state_features(self, z, rows, train, k_hmm, q_lgssm) -> tuple[dict[str, np.ndarray], dict]:
        train_seq = build_sequences(rows[train], z[train])
        hmm = GaussianHMM(k_hmm, n_restarts=self.cfg.hmm_restarts, max_iter=self.cfg.max_iter,
                          seed=self.cfg.seed).fit(train_seq.y, train_seq.mask, train_seq.lengths)
        lgssm = LinearGaussianSSM(q_lgssm, max_iter=self.cfg.max_iter, seed=self.cfg.seed).fit(
            train_seq.y, train_seq.mask, train_seq.lengths)
        all_seq = build_sequences(rows, z)                     # every child, filtered causally
        hmm_feat = all_seq.gather(hmm.filter(all_seq.y, all_seq.mask, all_seq.lengths))
        lgssm_feat = all_seq.gather(lgssm.filter(all_seq.y, all_seq.mask, all_seq.lengths))
        ewma_feat = all_seq.gather(ewma(all_seq, self.cfg.ewma_alpha))
        sets = {"raw": z, "ewma": ewma_feat, "hmm": hmm_feat, "lgssm": lgssm_feat,
                "raw_hmm": np.hstack([z, hmm_feat]), "raw_lgssm": np.hstack([z, lgssm_feat])}
        info = {"hmm_train_loglik": hmm.train_loglik_, "hmm_restart_logliks": hmm.restart_logliks_,
                "hmm_transition_diagonal": np.diag(hmm.params_.trans).round(4).tolist(),
                "lgssm_A": lgssm.params_.A.round(4).tolist()}
        return sets, info

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        bcfg = self.bench.cfg
        device = "cuda" if self.bench.compute.device != "cpu" else None
        predictions, selections = [], []
        folds = list(self.bench.split.iter_folds())
        folds.append((HOLDOUT_FOLD, sorted(self.bench.split.cv_subjects["subject"]), self.bench.split.holdout_subjects))
        for window_name in self.cfg.windows:
            window = self.bench.windows[window_name]
            cache = FeatureCache(self.bench.feature_root, window, bcfg.aperiodic_regions, self.bench.positions,
                                 device, n_jobs=bcfg.feature_n_jobs, embedder_factory=self.bench.embedder_factory,
                                 reve_tag=self.bench.reve_tag)
            rows = self.bench.cohort(window).sort_values(["block_key", "trial_in_block"]).reset_index(drop=True)
            centred = block_centre(self._features(cache, rows), rows["block_key"].to_numpy())
            covariates = self.bench._covariates(rows)
            subjects = rows["subject"].to_numpy()
            for fold, train_subjects, test_subjects in folds:
                start = time.perf_counter()
                train = rows["subject"].isin(train_subjects).to_numpy()
                test = rows["subject"].isin(test_subjects).to_numpy()
                LeakageAuditor.assert_rows_grouped(rows, train, test)
                rng = np.random.default_rng(self.cfg.seed + fold)
                fit_idx = np.flatnonzero(train)
                if len(fit_idx) > self.cfg.pca_subsample:
                    fit_idx = np.sort(rng.choice(fit_idx, self.cfg.pca_subsample, replace=False))
                scaler = StandardScaler().fit(centred[fit_idx])
                pca = PCA(self.cfg.n_components, random_state=self.cfg.seed).fit(scaler.transform(centred[fit_idx]))
                z = pca.transform(scaler.transform(centred))
                k_hmm, q_lgssm, records = self._select(z, rows, train)
                sets, info = self._state_features(z, rows, train, k_hmm, q_lgssm)
                for record in records:
                    selections.append({"window": window_name, "fold": fold, **record,
                                       "chosen": (record["size"] == (k_hmm if record["model"] == "hmm" else q_lgssm))})
                selections.append({"window": window_name, "fold": fold, "model": "refit", "size": None,
                                   "val_loglik_per_trial": None, "restart_logliks": info["hmm_restart_logliks"],
                                   "chosen": None, **{k: v for k, v in info.items() if k != "hmm_restart_logliks"},
                                   "pca_explained": float(pca.explained_variance_ratio_.sum())})
                for target in self.cfg.targets:
                    keep = rows["valid_response"].to_numpy(dtype=bool) if target == "rt_s" else np.ones(len(rows), bool)
                    y = rows[target].to_numpy(dtype=float)
                    tr, te = train & keep, test & keep
                    for name in FEATURE_SETS:
                        model = GroupedRidge(name=name, alphas=tuple(bcfg.ridge_alphas),
                                             n_inner_folds=bcfg.inner_folds, seed=bcfg.seed)
                        model.fit(DecoderInput(sets[name][tr], subjects[tr], covariates[tr]), y[tr])
                        pred = model.predict(DecoderInput(sets[name][te], subjects[te], covariates[te]))
                        out = rows.loc[te, ["subject", "release", "block_key", "trial_index"]].copy()
                        out = out.assign(window=window_name, target=target, model=name, fold=fold,
                                         y_true=y[te], y_pred=pred, alpha=model.alpha_,
                                         k_hmm=k_hmm, q_lgssm=q_lgssm)
                        predictions.append(out)
                _LOG.info("%s | fold %s done: K=%d, q=%d (%.0f s)", window_name, fold, k_hmm, q_lgssm,
                          time.perf_counter() - start)
        return pd.concat(predictions, ignore_index=True), pd.DataFrame(selections)
