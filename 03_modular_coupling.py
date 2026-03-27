"""
03_modular_coupling.py - Module-Specific SC-FC Coupling

Detects network modules via Ward hierarchical clustering on the combined
group-mean FC, then computes within-module SC-FC coupling per subject.

Scientific question: Which brain network modules from above clustering
show the largest SC-FC coupling disruption in SCZ?

Inputs:
    arrays/SC_ctrl.npy, SC_schz.npy, FC_ctrl.npy, FC_schz.npy
    arrays/FC_ctrl_mean.npy, FC_schz_mean.npy

Outputs:
    arrays/module_labels.npy
    results/modular_coupling.csv
    figures/fig03_modular_coupling.png

Runtime: ~10 seconds

Code inspired by / adapted from following sources:
    https://github.com/johannaleapopp/SC_FC_Coupling_Task_Intelligence
    https://github.com/zijin-gu/scfc-coupling
    https://github.com/netneurolab/liu_meg-scfc
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (PATHS, ensure_dirs, cohens_d,
                   within_module_mask, within_module_coupling)
ensure_dirs()

ARR_DIR     = PATHS['arrays']
FIG_DIR     = PATHS['figures']
RESULTS_DIR = PATHS['results']

N_MODULES = 5


def main():
    SC_ctrl = np.load(os.path.join(ARR_DIR, 'SC_ctrl.npy'))
    SC_schz = np.load(os.path.join(ARR_DIR, 'SC_schz.npy'))
    FC_ctrl = np.load(os.path.join(ARR_DIR, 'FC_ctrl.npy'))
    FC_schz = np.load(os.path.join(ARR_DIR, 'FC_schz.npy'))
    FC_ctrl_mean = np.load(os.path.join(ARR_DIR, 'FC_ctrl_mean.npy'))
    FC_schz_mean = np.load(os.path.join(ARR_DIR, 'FC_schz_mean.npy'))

    # Combined group-mean FC for module detection (avoids circular analysis)
    FC_combined = (FC_ctrl_mean + FC_schz_mean) / 2.0
    n_nodes = FC_combined.shape[0]

    # Module detection: Ward clustering on FC
    # FC is in raw Pearson r space (back-transformed from Fisher-z mean),
    # so |FC| <= 1 and distances are naturally non-negative
    FC_dist = np.clip(1.0 - np.abs(FC_combined), 0, None)
    np.fill_diagonal(FC_dist, 0)
    FC_dist = (FC_dist + FC_dist.T) / 2.0
    condensed = squareform(FC_dist, checks=False)

    Z = linkage(condensed, method='ward')
    labels = fcluster(Z, t=N_MODULES, criterion='maxclust') - 1  # 0-indexed

    np.save(os.path.join(ARR_DIR, 'module_labels.npy'), labels)
    print(f"Module sizes: {[int(np.sum(labels == m)) for m in range(N_MODULES)]}")

    # Within-module SC-FC coupling per module x group
    results = []
    p_values = []

    for m in range(N_MODULES):
        mask = within_module_mask(labels, m, n_nodes)
        if mask.sum() < 3:
            print(f"  Module {m}: too few edges ({mask.sum()}), skipping")
            continue

        coup_ctrl = within_module_coupling(SC_ctrl, FC_ctrl, mask)
        coup_schz = within_module_coupling(SC_schz, FC_schz, mask)

        from scipy import stats
        stat_val, p_val = stats.mannwhitneyu(coup_ctrl, coup_schz,
                                              alternative='two-sided')
        d = cohens_d(coup_ctrl, coup_schz)
        p_values.append(p_val)

        results.append({
            'module': m,
            'n_nodes': int(np.sum(labels == m)),
            'n_edges': int(mask.sum()),
            'ctrl_mean': coup_ctrl.mean(),
            'ctrl_std': coup_ctrl.std(),
            'schz_mean': coup_schz.mean(),
            'schz_std': coup_schz.std(),
            'U_stat': stat_val,
            'p_value': p_val,
            'cohens_d': d,
        })

    # FDR correction (Benjamini-Hochberg)
    raw_ps = np.array(p_values)
    try:
        from scipy.stats import false_discovery_control
        fdr_ps = false_discovery_control(raw_ps, method='bh')
    except ImportError:
        fdr_ps = np.minimum(raw_ps * len(raw_ps), 1.0)
        print("  (Using Bonferroni correction -- scipy too old for FDR)")

    for i, r in enumerate(results):
        r['p_fdr'] = fdr_ps[i]

    # Print and save
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('cohens_d', key=abs, ascending=False)
    print("\nModule disruption ranked by |Cohen's d|:")
    print(df_results.to_string(index=False))

    csv_path = os.path.join(RESULTS_DIR, 'modular_coupling.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Figure: bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results))
    width = 0.35

    ctrl_means = [r['ctrl_mean'] for r in results]
    schz_means = [r['schz_mean'] for r in results]
    ctrl_stds  = [r['ctrl_std'] for r in results]
    schz_stds  = [r['schz_std'] for r in results]
    mod_labels = [f"M{r['module']}" for r in results]

    ax.bar(x - width/2, ctrl_means, width, yerr=ctrl_stds,
           label='Controls', color='steelblue', alpha=0.8, capsize=3)
    ax.bar(x + width/2, schz_means, width, yerr=schz_stds,
           label='SCZ', color='salmon', alpha=0.8, capsize=3)

    for i, r in enumerate(results):
        if r['p_fdr'] < 0.05:
            ymax = max(r['ctrl_mean'] + r['ctrl_std'],
                       r['schz_mean'] + r['schz_std'])
            ax.text(i, ymax + 0.02, '*', ha='center', fontsize=16,
                    fontweight='bold')

    ax.set_xlabel('Module', fontsize=13)
    ax.set_ylabel('Within-Module SC-FC Coupling (r)', fontsize=13)
    ax.set_title('Modular SC-FC Coupling: Controls vs SCZ', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(mod_labels)
    ax.legend()

    fig_path = os.path.join(FIG_DIR, 'fig03_modular_coupling.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
    print("\n[DONE] 03_modular_coupling.py completed successfully.")


if __name__ == '__main__':
    main()
