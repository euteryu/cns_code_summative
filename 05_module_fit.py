"""
05_module_fit.py - Module-Specific Model Fit

Runs optimal WC models (from Step 4) for each group with longer simulation
(120 s), then evaluates model fit within each module.

Scientific question: Do modules with the worst empirical SC-FC disruption
also show the largest model fit gap? If so, this supports module-
heterogeneous E/I dynamics (Yang et al. 2016 hierarchy account).

Inputs:
    arrays/SC_ctrl_mean.npy, SC_schz_mean.npy
    arrays/FC_ctrl_raw_mean.npy, FC_schz_raw_mean.npy
    arrays/module_labels.npy, opt_params_ctrl.npy, opt_params_schz.npy
    results/modular_coupling.csv

Outputs:
    results/module_fit.csv
    figures/fig06_module_fit.png

Runtime: ~2 minutes

Code inspired by / adapted from following sources:
    https://github.com/johannaleapopp/SC_FC_Coupling_Task_Intelligence
    https://github.com/neurolib-dev/neurolib
    https://github.com/OpenSourceBrain/WilsonCowan
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PATHS, ensure_dirs
from wc_model import simulate_fc
ensure_dirs()

ARR_DIR     = PATHS['arrays']
FIG_DIR     = PATHS['figures']
RESULTS_DIR = PATHS['results']

LONG_DURATION = 120 * 1000  # 120 seconds for stable module-level FC


# - Helper functions

def within_module_fc_fit(sim_fc, emp_fc, labels, module_id):
    """Pearson r between simulated and empirical FC edges within one module.

    Args:
        sim_fc: Simulated FC matrix (n_nodes, n_nodes).
        emp_fc: Empirical FC matrix (n_nodes, n_nodes).
        labels: Module assignment array (n_nodes,).
        module_id: Which module to evaluate.

    Returns:
        Float: Pearson r within the module's upper-triangle edges.
    """
    nodes = np.where(labels == module_id)[0]
    if len(nodes) < 2:
        return np.nan
    rows, cols = [], []
    for i_idx in range(len(nodes)):
        for j_idx in range(i_idx + 1, len(nodes)):
            rows.append(nodes[i_idx])
            cols.append(nodes[j_idx])
    if len(rows) < 3:
        return np.nan
    sim_vals = sim_fc[rows, cols]
    emp_vals = emp_fc[rows, cols]
    if sim_vals.std() == 0 or emp_vals.std() == 0:
        return 0.0
    return np.corrcoef(sim_vals, emp_vals)[0, 1]


# - Main

def main():
    """Run optimal models and evaluate module-level fit."""
    SC_ctrl_mean = np.load(os.path.join(ARR_DIR, 'SC_ctrl_mean.npy'))
    SC_schz_mean = np.load(os.path.join(ARR_DIR, 'SC_schz_mean.npy'))
    FC_ctrl_emp  = np.load(os.path.join(ARR_DIR, 'FC_ctrl_raw_mean.npy'))
    FC_schz_emp  = np.load(os.path.join(ARR_DIR, 'FC_schz_raw_mean.npy'))
    labels       = np.load(os.path.join(ARR_DIR, 'module_labels.npy'))
    opt_ctrl     = np.load(os.path.join(ARR_DIR, 'opt_params_ctrl.npy'))
    opt_schz     = np.load(os.path.join(ARR_DIR, 'opt_params_schz.npy'))
    mod_df       = pd.read_csv(os.path.join(RESULTS_DIR, 'modular_coupling.csv'))
    n_modules    = len(mod_df)

    # - Run optimal models
    print(f"Running ctrl optimal model (K_gl={opt_ctrl[0]:.2f}, "
          f"c_excinh={opt_ctrl[1]:.2f}, {LONG_DURATION/1000:.0f} s) ...")
    sim_fc_ctrl, ok_c = simulate_fc(SC_ctrl_mean, K_gl=opt_ctrl[0],
                                     c_excinh=opt_ctrl[1],
                                     duration=LONG_DURATION)
    print(f"  Success: {ok_c}")

    print(f"Running SCZ optimal model (K_gl={opt_schz[0]:.2f}, "
          f"c_excinh={opt_schz[1]:.2f}, {LONG_DURATION/1000:.0f} s) ...")
    sim_fc_schz, ok_s = simulate_fc(SC_schz_mean, K_gl=opt_schz[0],
                                     c_excinh=opt_schz[1],
                                     duration=LONG_DURATION)
    print(f"  Success: {ok_s}")

    # - Module-level fits
    results = []
    for _, row in mod_df.iterrows():
        m = int(row['module'])
        fit_ctrl = within_module_fc_fit(sim_fc_ctrl, FC_ctrl_emp, labels, m)
        fit_schz = within_module_fc_fit(sim_fc_schz, FC_schz_emp, labels, m)
        fit_gap = fit_ctrl - fit_schz
        emp_disruption = abs(row['cohens_d'])

        results.append({
            'module': m, 'n_nodes': int(row['n_nodes']),
            'fit_ctrl': fit_ctrl, 'fit_schz': fit_schz,
            'fit_gap': fit_gap, 'empirical_disruption_abs_d': emp_disruption,
        })
        print(f"  Module {m}: fit_ctrl={fit_ctrl:.3f}, fit_schz={fit_schz:.3f}, "
              f"gap={fit_gap:.3f}, emp|d|={emp_disruption:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'module_fit.csv'), index=False)

    # - Disruption vs fit-gap correlation
    valid = df.dropna(subset=['fit_gap', 'empirical_disruption_abs_d'])
    r_all, p_all = np.nan, np.nan
    if len(valid) >= 3:
        r_all, p_all = stats.pearsonr(valid['empirical_disruption_abs_d'],
                                       valid['fit_gap'])
        print(f"\nDisruption vs fit-gap (all {len(valid)} modules): "
              f"r={r_all:.4f}, p={p_all:.4f}")

    # Leave-one-out robustness check (n=5 makes single outliers dominant)
    print("\nLeave-one-out robustness check:")
    for drop_m in valid['module'].values:
        subset = valid[valid['module'] != drop_m]
        if len(subset) >= 3:
            r_loo, p_loo = stats.pearsonr(subset['empirical_disruption_abs_d'],
                                           subset['fit_gap'])
            print(f"  Without M{int(drop_m)}: r={r_loo:.4f}, p={p_loo:.4f}")

    if len(valid) >= 3:
        if abs(r_all) > 0.5 and p_all < 0.1:
            print("\n-> Tentative: modules with worst empirical decoupling show "
                  "largest model fit gap (supports heterogeneous E/I).")
            print("   CAVEAT: n=5 modules; interpret with caution.")
        else:
            print("\n-> Global E/I shift appears to explain modules equally.")

    # - Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(n_modules)
    width = 0.35
    axes[0].bar(x - width/2, df['fit_ctrl'], width, label='Controls',
                color='steelblue', alpha=0.8)
    axes[0].bar(x + width/2, df['fit_schz'], width, label='SCZ',
                color='salmon', alpha=0.8)
    axes[0].set_xlabel('Module', fontsize=12)
    axes[0].set_ylabel('Model Fit (r: simFC vs empFC)', fontsize=12)
    axes[0].set_title('Within-Module Model Fit', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'M{m}' for m in df['module']])
    axes[0].legend()
    axes[0].axhline(0, color='k', linewidth=0.5)

    if len(valid) >= 3:
        axes[1].scatter(valid['empirical_disruption_abs_d'], valid['fit_gap'],
                        s=100, c='purple', edgecolors='k', zorder=3)
        for _, row in valid.iterrows():
            axes[1].annotate(f"M{int(row['module'])}",
                             (row['empirical_disruption_abs_d'], row['fit_gap']),
                             textcoords='offset points', xytext=(8, 5),
                             fontsize=10)
        slope, intercept = np.polyfit(valid['empirical_disruption_abs_d'],
                                       valid['fit_gap'], 1)
        x_line = np.linspace(valid['empirical_disruption_abs_d'].min(),
                              valid['empirical_disruption_abs_d'].max(), 50)
        axes[1].plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5)
        axes[1].set_title(f'Disruption vs Fit Gap\n'
                          f'(r={r_all:.2f}, p={p_all:.3f}, n={len(valid)})',
                          fontsize=13)

    axes[1].set_xlabel('Empirical SC-FC Disruption (|Cohen d|)', fontsize=12)
    axes[1].set_ylabel('Model Fit Gap (ctrl - schz)', fontsize=12)
    axes[1].axhline(0, color='k', linewidth=0.5, alpha=0.3)

    fig_path = os.path.join(FIG_DIR, 'fig06_module_fit.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
    print("\n[DONE] 05_module_fit.py completed successfully.")


if __name__ == '__main__':
    main()
