"""
07_individual_fits.py - Individual SC-FC Model Fits

Uses each subject's OWN SC matrix with group-optimal WC parametres,
then measures how well the model fits that subject's OWN FC.

Scientific question: Is the group-optimal model uniformly poor for SCZ,
or do some patients show good fits (suggesting heterogeneity)?

Inputs:
    arrays/SC_ctrl.npy, SC_schz.npy, FC_ctrl.npy, FC_schz.npy
    arrays/opt_params_ctrl.npy, opt_params_schz.npy
    arrays/coupling_ctrl.npy, coupling_schz.npy

Outputs:
    results/individual_fits.csv
    figures/fig08a_individual_fits_violin.png
    figures/fig08b_fit_vs_coupling.png

Runtime: ~3 minutes with parallelisation

Code inspired by / adapted from following sources:
    https://github.com/johannaleapopp/SC_FC_Coupling_Task_Intelligence
    https://github.com/zijin-gu/scfc-coupling
    https://github.com/netneurolab/liu_meg-scfc
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PATHS, ensure_dirs, cohens_d
from wc_model import simulate_fc, fc_fit
ensure_dirs()

ARR_DIR     = PATHS['arrays']
FIG_DIR     = PATHS['figures']
RESULTS_DIR = PATHS['results']

SIM_DURATION = 60 * 1000


def fit_one_subject(SC_subj, FC_subj, K_gl, c_excinh):
    """Run WC on one subject's SC and return fit to that subject's FC.

    Args:
        SC_subj: Individual SC matrix (n_nodes, n_nodes).
        FC_subj: Individual FC matrix (n_nodes, n_nodes).
        K_gl: Global coupling parametre.
        c_excinh: E->I coupling parametre.

    Returns:
        Float: Pearson r between simulated and empirical FC upper triangles.
    """
    sim_fc, success = simulate_fc(SC_subj, K_gl=K_gl, c_excinh=c_excinh,
                                   duration=SIM_DURATION)
    if not success:
        return np.nan
    return fc_fit(sim_fc, FC_subj)


def main():
    """Run individual-level model fits and produce summary statistics and figures."""
    SC_ctrl = np.load(os.path.join(ARR_DIR, 'SC_ctrl.npy'))
    SC_schz = np.load(os.path.join(ARR_DIR, 'SC_schz.npy'))
    FC_ctrl = np.load(os.path.join(ARR_DIR, 'FC_ctrl.npy'))
    FC_schz = np.load(os.path.join(ARR_DIR, 'FC_schz.npy'))
    opt_ctrl = np.load(os.path.join(ARR_DIR, 'opt_params_ctrl.npy'))
    opt_schz = np.load(os.path.join(ARR_DIR, 'opt_params_schz.npy'))
    coup_ctrl = np.load(os.path.join(ARR_DIR, 'coupling_ctrl.npy'))
    coup_schz = np.load(os.path.join(ARR_DIR, 'coupling_schz.npy'))

    n_ctrl, n_schz = SC_ctrl.shape[0], SC_schz.shape[0]

    print(f"Ctrl-optimal parametres: K_gl={opt_ctrl[0]:.4f}, c_excinh={opt_ctrl[1]:.4f}")
    print(f"SCZ-optimal parametres:  K_gl={opt_schz[0]:.4f}, c_excinh={opt_schz[1]:.4f}")

    # - Controls: own SC + ctrl-optimal parametres
    print(f"\nFitting {n_ctrl} ctrl subjects with ctrl-optimal parametres ...")
    fit_ctrl_own = np.array(Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_one_subject)(SC_ctrl[i], FC_ctrl[i], opt_ctrl[0], opt_ctrl[1])
        for i in range(n_ctrl)
    ))

    # - SCZ: own SC + schz-optimal parametres
    print(f"\nFitting {n_schz} SCZ subjects with SCZ-optimal parametres ...")
    fit_schz_own = np.array(Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_one_subject)(SC_schz[i], FC_schz[i], opt_schz[0], opt_schz[1])
        for i in range(n_schz)
    ))

    # - SCZ: own SC + ctrl-optimal parametres (cross-group)
    print(f"\nFitting {n_schz} SCZ subjects with CTRL-optimal parametres (cross-group) ...")
    fit_schz_cross = np.array(Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_one_subject)(SC_schz[i], FC_schz[i], opt_ctrl[0], opt_ctrl[1])
        for i in range(n_schz)
    ))

    # - Results
    rows = []
    for i in range(n_ctrl):
        rows.append({'subject': i, 'group': 'ctrl',
                     'fit_own_model': fit_ctrl_own[i], 'fit_cross_model': np.nan})
    for i in range(n_schz):
        rows.append({'subject': i, 'group': 'schz',
                     'fit_own_model': fit_schz_own[i],
                     'fit_cross_model': fit_schz_cross[i]})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'individual_fits.csv')
    df.to_csv(csv_path, index=False)

    # - Statistics
    valid_c = fit_ctrl_own[~np.isnan(fit_ctrl_own)]
    valid_s = fit_schz_own[~np.isnan(fit_schz_own)]

    print(f"\nIndividual model fit (own-group parametres):")
    print(f"  Controls: {valid_c.mean():.4f} +/- {valid_c.std():.4f}")
    print(f"  SCZ:      {valid_s.mean():.4f} +/- {valid_s.std():.4f}")

    t_stat, t_p = stats.ttest_ind(valid_c, valid_s)
    d = cohens_d(valid_c, valid_s)
    print(f"  t-test: t={t_stat:.4f}, p={t_p:.4f}, Cohen's d={d:.4f}")

    # Variance ratio F-test (higher SCZ variance = heterogeneity)
    nx, ny = len(valid_c), len(valid_s)
    var_c, var_s = valid_c.var(ddof=1), valid_s.var(ddof=1)
    f_stat = var_s / var_c if var_c > 0 else np.nan
    f_p = 2 * min(stats.f.cdf(f_stat, ny - 1, nx - 1),
                  1 - stats.f.cdf(f_stat, ny - 1, nx - 1))
    print(f"  Variance: ctrl={var_c:.6f}, schz={var_s:.6f}")
    print(f"  F-test (variance ratio): F={f_stat:.4f}, p={f_p:.4f}")

    valid_sx = fit_schz_cross[~np.isnan(fit_schz_cross)]
    print(f"\nSCZ fit with ctrl-optimal parametres: {valid_sx.mean():.4f} +/- {valid_sx.std():.4f}")
    print(f"\nResults saved to {csv_path}")

    # - Figure 08a: violin
    df_plot = df.dropna(subset=['fit_own_model'])
    fig, ax = plt.subplots(figsize=(5, 6))
    sns.violinplot(data=df_plot, x='group', y='fit_own_model', hue='group',
                   palette={'ctrl': 'steelblue', 'schz': 'salmon'},
                   inner=None, ax=ax, alpha=0.6, legend=False)
    sns.stripplot(data=df_plot, x='group', y='fit_own_model',
                  color='k', size=5, jitter=0.15, ax=ax, alpha=0.7)
    ax.set_xlabel('Group', fontsize=13)
    ax.set_ylabel('Model Fit (r: simFC vs own FC)', fontsize=13)
    ax.set_title(f'Individual Model Fits\nt-test p={t_p:.4f}, d={d:.2f}\n'
                 f'Var ratio F={f_stat:.2f}, p={f_p:.4f}', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Controls', 'SCZ'])

    fig_path = os.path.join(FIG_DIR, 'fig08a_individual_fits_violin.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")

    # - Figure 08b: fit vs coupling 
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(coup_ctrl, fit_ctrl_own, s=50, c='steelblue',
               edgecolors='k', alpha=0.7, label='Controls', zorder=3)
    ax.scatter(coup_schz, fit_schz_own, s=50, c='salmon',
               edgecolors='k', alpha=0.7, label='SCZ', zorder=3)

    valid_mask_c = ~np.isnan(fit_ctrl_own)
    valid_mask_s = ~np.isnan(fit_schz_own)

    if valid_mask_c.sum() >= 5:
        rho_c, p_c = stats.spearmanr(coup_ctrl[valid_mask_c],
                                      fit_ctrl_own[valid_mask_c])
        ax.annotate(f'Ctrl: rho={rho_c:.3f}, p={p_c:.3f}',
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    va='top', fontsize=10, color='steelblue')

    if valid_mask_s.sum() >= 5:
        rho_s, p_s = stats.spearmanr(coup_schz[valid_mask_s],
                                      fit_schz_own[valid_mask_s])
        ax.annotate(f'SCZ: rho={rho_s:.3f}, p={p_s:.3f}',
                    xy=(0.02, 0.91), xycoords='axes fraction',
                    va='top', fontsize=10, color='salmon')

    ax.set_xlabel('Individual SC-FC Coupling (empirical)', fontsize=12)
    ax.set_ylabel('Individual Model Fit (simFC vs ownFC)', fontsize=12)
    ax.set_title('Model Fit vs Empirical SC-FC Coupling', fontsize=13)
    ax.legend()

    fig_path = os.path.join(FIG_DIR, 'fig08b_fit_vs_coupling.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
    print("\n[DONE] 07_individual_fits.py completed successfully.")


if __name__ == '__main__':
    main()
