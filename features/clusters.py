"""Electrode clusters for the single-trial signal measures.

Each 10-05 label is mapped to the nearest GSN-HydroCel-129 electrode of MNE's templates (Euclidean
distance between the template positions as distributed). A rigid fiducial alignment of the two
templates was tried and rejected: it moves Cz onto E7 and lengthens every posterior distance,
because the templates differ in shape, not only in frame. For the midline and posterior labels
used here the direct mapping gives CPz E55, Pz E62, POz E72, Oz E75, O1 E70 and O2 E83.
"""

from __future__ import annotations

import warnings

import numpy as np

CLUSTER_LABELS = {
    "cpp": ("CPz", "CP1", "CP2", "Pz"),
    "alpha": ("POz", "PO3", "PO4", "PO7", "PO8", "Oz", "O1", "O2"),
    "ssvep": ("Oz", "O1", "O2", "POz"),
}
EXTRA_LABELS = ("C3", "C4")          # motor sites, stored for later exploratory use only


def nearest_electrodes(labels, montage: str = "GSN-HydroCel-129") -> dict[str, str]:
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)             # standard_1005 renamed in MNE 1.14
        reference = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    target = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
    names = list(target)
    positions = np.array([target[n] for n in names])
    out = {}
    for label in labels:
        distance = np.linalg.norm(positions - np.asarray(reference[label]), axis=1)
        out[label] = names[int(np.argmin(distance))]
    return out


def cluster_channels() -> dict[str, list[str]]:
    mapping = nearest_electrodes({label for labels in CLUSTER_LABELS.values() for label in labels})
    return {cluster: [mapping[label] for label in labels] for cluster, labels in CLUSTER_LABELS.items()}


def store_channels() -> list[str]:
    """Channels of the epoch store: every cluster electrode plus the motor sites, in montage order."""
    mapping = nearest_electrodes(EXTRA_LABELS)
    wanted = {ch for chans in cluster_channels().values() for ch in chans} | set(mapping.values())
    return sorted(wanted, key=lambda ch: (ch != "Cz", int(ch[1:]) if ch.startswith("E") else 0))
