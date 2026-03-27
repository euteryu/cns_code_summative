"""
01_load_qc.py -- Data Loading and Quality Control

Loads SC (density) and FC (correlation) at Lausanne Scale 0 (83 nodes)
for SCZ and control groups from the HDF5 dataset.

Looking for outlier subjects that should be flagged before downstream analysis.

Inputs:
    dataset/27_SCHZ_CTRL_dataset.mat  - HDF5 (v7.3) connectome data

Outputs:
    arrays/SC_ctrl.npy, SC_schz.npy     - per-subject SC (27, 83, 83)
    arrays/FC_ctrl.npy, FC_schz.npy     - per-subject raw Pearson FC
    arrays/SC_ctrl_mean.npy, SC_schz_mean.npy
    arrays/FC_ctrl_mean.npy, FC_schz_mean.npy  - Fisher-z averaged, back-transformed
    arrays/FC_ctrl_raw_mean.npy, FC_schz_raw_mean.npy  - alias for above
    figures/fig01_group_matrices.png

Preprocessing:
    SC: log1p -> per-subject max-normalise -> diagonal = 0
    FC: diagonal = 0 (kept as raw Pearson r for coupling & module analysis)
    Group-mean FC: averaged in Fisher-z space, then back-transformed (tanh)

Runtime: ~10 seconds

Code inspired by / adapted from following sources:
    Weeks (3,4,5) Module Lab Practicals
"""

import os, sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PATHS, ensure_dirs
ensure_dirs()

ARR_DIR = PATHS['arrays']
FIG_DIR = PATHS['figures']
MAT_PATH = PATHS['mat']
SCALE_IDX = 0  # 83 nodes


def load_matrices(mat, path, scale_idx=SCALE_IDX):
    """Dereference HDF5 object reference into a NumPy array.

    The dataset is stored as (5_scales, 1) object references; each
    [scale, 0] resolves to a dataset of shape (n_subj, n_nodes, n_nodes).

    Args:
        mat: Open h5py File object.
        path: HDF5 group path (e.g. 'SC_FC_Connectomes/SC_density/ctrl').
        scale_idx: Lausanne atlas scale index (0 = 83 nodes).

    Returns:
        Array of shape (n_subj, n_nodes, n_nodes).
    """
    group = mat[path]
    return np.array(mat[group[scale_idx, 0]])


def preprocess_sc(sc_raw):
    """Apply SC preprocessing: log1p, per-subject max-normalise, diagonal = 0.

    Args:
        sc_raw: Raw SC array (n_subj, n_nodes, n_nodes).

    Returns:
        Preprocessed SC array with values in [0, 1].
    """
    sc = np.log1p(sc_raw)
    for i in range(sc.shape[0]):
        mx = sc[i].max()
        if mx > 0:
            sc[i] /= mx
        np.fill_diagonal(sc[i], 0)
    return sc


def qc_flag(matrices, label):
    """Flag subjects whose mean connectivity strength is >3 SD from the group mean.

    Args:
        matrices: Array (n_subj, n_nodes, n_nodes).
        label: Descriptive string for printing (e.g. 'SC ctrl').

    Returns:
        Array of flagged subject indices.
    """
    strengths = matrices.mean(axis=(1, 2))
    mu, sd = strengths.mean(), strengths.std()
    flagged = np.where(np.abs(strengths - mu) > 3 * sd)[0]
    if len(flagged):
        print(f"  QC WARNING: {label} outliers at indices {flagged.tolist()}")
    else:
        print(f"  QC OK: {label} - no outliers")
    return flagged


def main():
    print("Loading HDF5 dataset ...")
    mat = h5py.File(MAT_PATH, 'r')

    SC_ctrl_raw = load_matrices(mat, 'SC_FC_Connectomes/SC_density/ctrl')
    SC_schz_raw = load_matrices(mat, 'SC_FC_Connectomes/SC_density/schz')
    FC_ctrl_raw = load_matrices(mat, 'SC_FC_Connectomes/FC_correlation/ctrl')
    FC_schz_raw = load_matrices(mat, 'SC_FC_Connectomes/FC_correlation/schz')
    mat.close()

    n_ctrl, n_schz = SC_ctrl_raw.shape[0], SC_schz_raw.shape[0]
    n_nodes = SC_ctrl_raw.shape[1]
    print(f"  Controls: {n_ctrl} subjects, {n_nodes} nodes")
    print(f"  SCZ:      {n_schz} subjects, {n_nodes} nodes")

    # SC preprocessing
    SC_ctrl = preprocess_sc(SC_ctrl_raw)
    SC_schz = preprocess_sc(SC_schz_raw)

    # FC preprocessing: raw Pearson r with diagonal = 0
    FC_ctrl_raw_clean = FC_ctrl_raw.copy()
    FC_schz_raw_clean = FC_schz_raw.copy()
    for i in range(n_ctrl):
        np.fill_diagonal(FC_ctrl_raw_clean[i], 0)
    for i in range(n_schz):
        np.fill_diagonal(FC_schz_raw_clean[i], 0)

    # Group means
    SC_ctrl_mean = SC_ctrl.mean(axis=0)
    SC_schz_mean = SC_schz.mean(axis=0)

    # Fisher-z transform for averaging, then back-transform with tanh
    FC_ctrl_z = np.arctanh(np.clip(FC_ctrl_raw, -0.9999, 0.9999))
    FC_schz_z = np.arctanh(np.clip(FC_schz_raw, -0.9999, 0.9999))
    for i in range(n_ctrl):
        np.fill_diagonal(FC_ctrl_z[i], 0)
    for i in range(n_schz):
        np.fill_diagonal(FC_schz_z[i], 0)

    FC_ctrl_mean_bt = np.tanh(FC_ctrl_z.mean(axis=0))
    FC_schz_mean_bt = np.tanh(FC_schz_z.mean(axis=0))
    np.fill_diagonal(FC_ctrl_mean_bt, 0)
    np.fill_diagonal(FC_schz_mean_bt, 0)

    # Quality control
    print("\nSubject-level QC (>3 SD from group mean strength):")
    qc_flag(SC_ctrl, "SC ctrl")
    qc_flag(SC_schz, "SC schz")
    qc_flag(FC_ctrl_raw_clean, "FC ctrl")
    qc_flag(FC_schz_raw_clean, "FC schz")

    # Save arrays
    np.save(os.path.join(ARR_DIR, 'SC_ctrl.npy'), SC_ctrl)
    np.save(os.path.join(ARR_DIR, 'SC_schz.npy'), SC_schz)
    np.save(os.path.join(ARR_DIR, 'FC_ctrl.npy'), FC_ctrl_raw_clean)
    np.save(os.path.join(ARR_DIR, 'FC_schz.npy'), FC_schz_raw_clean)
    np.save(os.path.join(ARR_DIR, 'SC_ctrl_mean.npy'), SC_ctrl_mean)
    np.save(os.path.join(ARR_DIR, 'SC_schz_mean.npy'), SC_schz_mean)
    np.save(os.path.join(ARR_DIR, 'FC_ctrl_mean.npy'), FC_ctrl_mean_bt)
    np.save(os.path.join(ARR_DIR, 'FC_schz_mean.npy'), FC_schz_mean_bt)
    np.save(os.path.join(ARR_DIR, 'FC_ctrl_raw_mean.npy'), FC_ctrl_mean_bt)
    np.save(os.path.join(ARR_DIR, 'FC_schz_raw_mean.npy'), FC_schz_mean_bt)
    print(f"\nArrays saved to {ARR_DIR}")

    # Figure: group-mean heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = ['SC -- Controls', 'SC -- SCZ', 'FC -- Controls', 'FC -- SCZ']
    data = [SC_ctrl_mean, SC_schz_mean, FC_ctrl_mean_bt, FC_schz_mean_bt]
    cmaps = ['hot', 'hot', 'coolwarm', 'coolwarm']

    for ax, title, d, cmap in zip(axes.ravel(), titles, data, cmaps):
        im = ax.imshow(d, cmap=cmap, aspect='equal')
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig01_group_matrices.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
    print("\n[DONE] 01_load_qc.py completed successfully.")


if __name__ == '__main__':
    main()
