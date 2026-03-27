"""
02_scfc_coupling.py - Global SC-FC Coupling Analysis

Computes whole-brain structural-functional connectivity coupling for each
subject and compares groups statistically.

Scientific question: Is global SC-FC coupling significantly different
between SCZ patients and healthy controls?

Inputs:
    arrays/SC_ctrl.npy, SC_schz.npy
    arrays/FC_ctrl.npy, FC_schz.npy

Outputs:
    results/global_coupling.csv
    figures/fig02_global_coupling.png
    arrays/coupling_ctrl.npy, coupling_schz.npy

Runtime: ~5 seconds

Code inspired by / adapted from following sources:
    https://github.com/johannaleapopp/SC_FC_Coupling_Task_Intelligence
    https://github.com/zijin-gu/scfc-coupling
    https://github.com/netneurolab/liu_meg-scfc
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (PATHS, ensure_dirs, cohens_d, choose_test,
                   subject_scfc_coupling)
ensure_dirs()

ARR_DIR     = PATHS['arrays']
FIG_DIR     = PATHS['figures']
RESULTS_DIR = PATHS['results']


def main():
    SC_ctrl = np.load(os.path.join(ARR_DIR, 'SC_ctrl.npy'))
    SC_schz = np.load(os.path.join(ARR_DIR, 'SC_schz.npy'))
    FC_ctrl = np.load(os.path.join(ARR_DIR, 'FC_ctrl.npy'))
    FC_schz = np.load(os.path.join(ARR_DIR, 'FC_schz.npy'))

    # Per-subject coupling
    coup_ctrl = subject_scfc_coupling(SC_ctrl, FC_ctrl)
    coup_schz = subject_scfc_coupling(SC_schz, FC_schz)

    print(f"SC-FC coupling - Controls: {coup_ctrl.mean():.4f} +/- {coup_ctrl.std():.4f}")
    print(f"SC-FC coupling - SCZ:      {coup_schz.mean():.4f} +/- {coup_schz.std():.4f}")

    # Group comparison
    stat_val, p_val, test_name = choose_test(coup_ctrl, coup_schz)
    d = cohens_d(coup_ctrl, coup_schz)
    print(f"\n{test_name}: stat={stat_val:.4f}, p={p_val:.4f}")
    print(f"Cohen's d = {d:.4f}")

    # Save
    df = pd.DataFrame({
        'subject': list(range(len(coup_ctrl))) + list(range(len(coup_schz))),
        'group': ['ctrl'] * len(coup_ctrl) + ['schz'] * len(coup_schz),
        'scfc_coupling': np.concatenate([coup_ctrl, coup_schz])
    })
    csv_path = os.path.join(RESULTS_DIR, 'global_coupling.csv')
    df.to_csv(csv_path, index=False)
    np.save(os.path.join(ARR_DIR, 'coupling_ctrl.npy'), coup_ctrl)
    np.save(os.path.join(ARR_DIR, 'coupling_schz.npy'), coup_schz)
    print(f"\nCoupling values saved to {csv_path}")

    # Figure: violin plot
    fig, ax = plt.subplots(figsize=(5, 6))
    sns.violinplot(data=df, x='group', y='scfc_coupling', hue='group',
                   palette={'ctrl': 'steelblue', 'schz': 'salmon'},
                   inner=None, ax=ax, alpha=0.6, legend=False)
    sns.stripplot(data=df, x='group', y='scfc_coupling',
                  color='k', size=5, jitter=0.15, ax=ax, alpha=0.7)
    ax.set_xlabel('Group', fontsize=13)
    ax.set_ylabel('SC-FC Coupling (Pearson r)', fontsize=13)
    ax.set_title(f'Global SC-FC Coupling\n{test_name}: p={p_val:.4f}, d={d:.2f}',
                 fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Controls', 'SCZ'])

    fig_path = os.path.join(FIG_DIR, 'fig02_global_coupling.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
    print("\n[DONE] 02_scfc_coupling.py completed successfully.")


if __name__ == '__main__':
    main()
