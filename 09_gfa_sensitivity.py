"""
09_gfa_sensitivity.py - gFA SC Measure Sensitivity Check

Reruns Steps 2 and 3 (global and modular SC-FC coupling) using gFA-weighted
SC matrices instead of streamline density.  Uses the SAME `module_labels.npy`
from the density analysis (no re-clustering).

If Module 2 disruption replicates with gFA, the effect reflects white-matter
microstructural differences rather than tractography streamline-counting
artefacts.

Inputs:
    dataset/27_SCHZ_CTRL_dataset.mat  (gFA SC matrices)
    arrays/FC_ctrl.npy, FC_schz.npy
    arrays/module_labels.npy
    results/modular_coupling.csv      (density results for comparison)

Outputs:
    results/gfa_global_coupling.csv
    results/gfa_modular_coupling.csv
    figures/fig10_gfa_modular_coupling.png

Code inspired by / adapted from following sources:
    https://github.com/johannaleapopp/SC_FC_Coupling_Task_Intelligence
    https://github.com/zijin-gu/scfc-coupling
    https://github.com/netneurolab/liu_meg-scfc
"""

import os, sys
import h5py
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (PATHS, ensure_dirs, cohens_d,
                   subject_scfc_coupling, within_module_mask,
                   within_module_coupling)
ensure_dirs()

ARR_DIR     = PATHS['arrays']
FIG_DIR     = PATHS['figures']
RESULTS_DIR = PATHS['results']
MAT_PATH    = PATHS['mat']

SCALE_IDX = 0  # 83 nodes


def load_matrices(mat, path, scale_idx=SCALE_IDX):
    """Load a stack of connectivity matrices from the HDF5 dataset.

    Args:
        mat: Open `h5py.File` handle for the .mat dataset.
        path: HDF5 group path (e.g. `'SC_FC_Connectomes/SC_gFA/ctrl'`).
        scale_idx: Lausanne scale index (0 = 83 nodes).

    Returns:
        numpy.ndarray: Array of shape `(n_subjects, n_nodes, n_nodes)`.
    """
    group = mat[path]
    return np.array(mat[group[scale_idx, 0]])


def preprocess_sc(sc_raw):
    """Apply the same preprocessing used for density SC matrices.

    Transforms raw SC values with `log1p`, normalises each subject by its
    maximum, and zeroes the diagonal.  This ensures gFA results are directly
    comparable to the density analysis.

    Args:
        sc_raw: Raw SC array of shape `(n_subjects, n_nodes, n_nodes)`.

    Returns:
        numpy.ndarray: Preprocessed SC array, same shape as input.
    """
    sc = np.log1p(sc_raw)
    for i in range(sc.shape[0]):
        mx = sc[i].max()
        if mx > 0:
            sc[i] /= mx
        np.fill_diagonal(sc[i], 0)
    return sc


def main():
    """Run the gFA sensitivity analysis for global and modular SC-FC coupling."""
    # - Load gFA SC matrices
    print("Loading gFA SC matrices ...")
    mat = h5py.File(MAT_PATH, 'r')
    gFA_ctrl_raw = load_matrices(mat, 'SC_FC_Connectomes/SC_gFA/ctrl')
    gFA_schz_raw = load_matrices(mat, 'SC_FC_Connectomes/SC_gFA/schz')
    mat.close()

    print(f"  gFA ctrl: {gFA_ctrl_raw.shape}, schz: {gFA_schz_raw.shape}")

    gFA_ctrl = preprocess_sc(gFA_ctrl_raw)
    gFA_schz = preprocess_sc(gFA_schz_raw)

    # Load empirical FC (same as density analysis)
    FC_ctrl = np.load(os.path.join(ARR_DIR, 'FC_ctrl.npy'))
    FC_schz = np.load(os.path.join(ARR_DIR, 'FC_schz.npy'))

    # Load module labels (from density analysis -- DO NOT re-cluster)
    module_labels = np.load(os.path.join(ARR_DIR, 'module_labels.npy'))
    n_modules = len(np.unique(module_labels))
    n_nodes = gFA_ctrl.shape[1]

    # - Step 2 replication: Global SC-FC coupling with gFA
    print("\n--- Global SC-FC Coupling (gFA) ---")
    coup_ctrl = subject_scfc_coupling(gFA_ctrl, FC_ctrl)
    coup_schz = subject_scfc_coupling(gFA_schz, FC_schz)

    print(f"  Controls: {coup_ctrl.mean():.4f} +/- {coup_ctrl.std():.4f}")
    print(f"  SCZ:      {coup_schz.mean():.4f} +/- {coup_schz.std():.4f}")

    t_stat, t_p = stats.ttest_ind(coup_ctrl, coup_schz)
    d_global = cohens_d(coup_ctrl, coup_schz)
    print(f"  t-test: t={t_stat:.4f}, p={t_p:.4f}, Cohen's d={d_global:.4f}")

    df_global = pd.DataFrame({
        'subject': list(range(len(coup_ctrl))) + list(range(len(coup_schz))),
        'group': ['ctrl'] * len(coup_ctrl) + ['schz'] * len(coup_schz),
        'gfa_scfc_coupling': np.concatenate([coup_ctrl, coup_schz])
    })
    df_global.to_csv(os.path.join(RESULTS_DIR, 'gfa_global_coupling.csv'),
                     index=False)

    # - Step 3 replication: Modular coupling with gFA 
    print("\n--- Modular SC-FC Coupling (gFA) ---")

    # Load density results for comparison
    density_df = pd.read_csv(os.path.join(RESULTS_DIR, 'modular_coupling.csv'))

    gfa_results = []
    for m in range(n_modules):
        mask = within_module_mask(module_labels, m, n_nodes)

        if mask.sum() < 3:
            continue

        mc_ctrl = within_module_coupling(gFA_ctrl, FC_ctrl, mask)
        mc_schz = within_module_coupling(gFA_schz, FC_schz, mask)

        stat_val, p_val = stats.mannwhitneyu(mc_ctrl, mc_schz,
                                              alternative='two-sided')
        d = cohens_d(mc_ctrl, mc_schz)

        gfa_results.append({
            'module': m, 'n_nodes': int(np.sum(module_labels == m)),
            'ctrl_mean': mc_ctrl.mean(), 'ctrl_std': mc_ctrl.std(),
            'schz_mean': mc_schz.mean(), 'schz_std': mc_schz.std(),
            'U_stat': stat_val, 'p_value': p_val, 'cohens_d': d,
        })

    gfa_df = pd.DataFrame(gfa_results)
    gfa_df.to_csv(os.path.join(RESULTS_DIR, 'gfa_modular_coupling.csv'),
                  index=False)

    # - Comparison table 
    print("\nComparison: density vs gFA module disruption")
    print(f"{'Module':>6} | {'d (density)':>11} | {'d (gFA)':>9} | "
          f"{'Consistent?':>12}")
    print("-" * 50)
    for _, gfa_row in gfa_df.iterrows():
        m = int(gfa_row['module'])
        d_gfa = gfa_row['cohens_d']
        density_row = density_df[density_df['module'] == m]
        if len(density_row) > 0:
            d_dens = density_row['cohens_d'].values[0]
            consistent = ("YES" if (d_dens > 0 and d_gfa > 0)
                          or (d_dens < 0 and d_gfa < 0) else "NO")
            print(f"{m:>6} | {d_dens:>11.3f} | {d_gfa:>9.3f} | "
                  f"{consistent:>12}")

    # Module 2 specifically
    m2_gfa = gfa_df[gfa_df['module'] == 2]
    if len(m2_gfa) > 0:
        d_m2 = m2_gfa['cohens_d'].values[0]
        replicates = "YES" if d_m2 > 0.3 else "NO"
        print(f"\nModule 2 disruption replicates with gFA: "
              f"{replicates} (d={d_m2:.3f})")

    # - Figure: bar chart matching Step 3 format 
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(gfa_results))
    width = 0.35

    ctrl_means = [r['ctrl_mean'] for r in gfa_results]
    schz_means = [r['schz_mean'] for r in gfa_results]
    ctrl_stds  = [r['ctrl_std'] for r in gfa_results]
    schz_stds  = [r['schz_std'] for r in gfa_results]
    mod_labels = [f"M{r['module']}" for r in gfa_results]

    ax.bar(x - width / 2, ctrl_means, width, yerr=ctrl_stds,
           label='Controls', color='steelblue', alpha=0.8, capsize=3)
    ax.bar(x + width / 2, schz_means, width, yerr=schz_stds,
           label='SCZ', color='salmon', alpha=0.8, capsize=3)

    ax.set_xlabel('Module', fontsize=13)
    ax.set_ylabel('Within-Module SC(gFA)-FC Coupling (r)', fontsize=13)
    ax.set_title('Modular SC-FC Coupling (gFA): Controls vs SCZ', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(mod_labels)
    ax.legend()

    fig_path = os.path.join(FIG_DIR, 'fig10_gfa_modular_coupling.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {fig_path}")

    print("\n[DONE] 09_gfa_sensitivity.py completed successfully.")


if __name__ == '__main__':
    main()
