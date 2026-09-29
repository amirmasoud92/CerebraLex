"""Trial sequences within blocks for the latent state models.

A block's targets form one sequence ordered by ``trial_in_block``. Trials without an analysable
window (artifact or invalid) stay in the sequence as missing observations, so the latent state
evolves through them. Features are centred within each block before any model sees them: the
fingerprint audit showed that EEG features identify the child and the block, and a state model
fitted on uncentred features would spend its states on that identity instead of on within-block
fluctuations. Centring uses only the block's own EEG, never a label.

Tested alternative: per-block offsets estimated inside EM instead of centring. In
24-trial blocks a block that stays in one state cannot be told apart from a shifted baseline, so
the offsets absorbed part of the state: on simulated sticky two-state blocks with random
baselines the within-block correlation of the HMM state with the truth fell from 0.56 (centred)
to 0.51, and the LGSSM dynamics were biased (0.69 for a true 0.8). Centring was kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Sequences:
    y: np.ndarray            # (N, T, d)
    mask: np.ndarray         # (N, T) observed trials
    lengths: np.ndarray      # (N,)
    block_keys: np.ndarray   # (N,)
    subjects: np.ndarray     # (N,)
    seq_index: np.ndarray    # (rows,) sequence of each input row
    position: np.ndarray     # (rows,) position of each input row in its sequence

    def gather(self, states: np.ndarray) -> np.ndarray:
        """Per-row values of a (N, T, k) array, in the order of the rows used to build it."""
        return states[self.seq_index, self.position]


def block_centre(features: np.ndarray, block_keys: np.ndarray) -> np.ndarray:
    """Subtract each block's mean feature vector (over the block's rows)."""
    frame = pd.DataFrame(np.asarray(features, dtype=np.float64))
    means = frame.groupby(np.asarray(block_keys)).transform("mean").to_numpy()
    return np.asarray(features, dtype=np.float64) - means


def build_sequences(rows: pd.DataFrame, features: np.ndarray) -> Sequences:
    """Pack row-aligned features into padded block sequences.

    ``rows`` holds ``block_key``, ``subject`` and ``trial_in_block`` for every observed trial;
    missing positions inside a block (and padding after it) are unobserved.
    """
    features = np.asarray(features, dtype=np.float64)
    blocks = rows["block_key"].to_numpy()
    positions = rows["trial_in_block"].to_numpy(dtype=int)
    keys, seq_index = np.unique(blocks, return_inverse=True)
    lengths = np.zeros(len(keys), dtype=int)
    np.maximum.at(lengths, seq_index, positions + 1)
    t_max = int(lengths.max())
    y = np.zeros((len(keys), t_max, features.shape[1]))
    mask = np.zeros((len(keys), t_max), dtype=bool)
    y[seq_index, positions] = features
    mask[seq_index, positions] = True
    subjects = pd.Series(rows["subject"].to_numpy()).groupby(seq_index).first().to_numpy()
    return Sequences(y, mask, lengths, keys, subjects, seq_index, positions)


def ewma(seqs: Sequences, alpha: float) -> np.ndarray:
    """Causal exponentially weighted average of the observed trials in each sequence (N, T, d);
    an unobserved trial carries the previous value forward."""
    out = np.zeros_like(seqs.y)
    state = np.zeros(seqs.y[:, 0].shape)
    seen = np.zeros(len(seqs.y), dtype=bool)
    for t in range(seqs.y.shape[1]):
        obs = seqs.mask[:, t]
        first = obs & ~seen
        update = obs & seen
        state[first] = seqs.y[first, t]
        state[update] = alpha * seqs.y[update, t] + (1.0 - alpha) * state[update]
        seen |= obs
        out[:, t] = state
    return out
