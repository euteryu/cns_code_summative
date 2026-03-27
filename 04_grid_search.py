"""
04_grid_search.py - 2D Parametre Grid Search: K_gl x c_excinh

Identifies the optimal Wilson-Cowan parametres for each group by searching
a 2D parametre space. Runs in two stages:

  Stage 1 (coarse): K_gl in [1.0, 8.0], c_excinh in [5.0, 25.0], 12x12 grid.
                    Identifies approximate optimum region.
  Stage 2 (fine):   10x10 grid centred on coarse optima (+/- 1.0 for K_gl,
                    +/- 3.0 for c_excinh). Confirms whether parametre
                    differences are robust to grid resolution.

Scientific question: Do SCZ and controls require different Wilson-Cowan
parametres (global coupling vs local E/I balance) to reproduce their
empirical FC?

Inputs:
    arrays/SC_ctrl_mean.npy, SC_schz_mean.npy
    arrays/FC_ctrl_raw_mean.npy, FC_schz_raw_mean.npy

Outputs:
    arrays/grid_ctrl.npy, grid_schz.npy           - coarse grids
    arrays/fine_grid_ctrl.npy, fine_grid_schz.npy  - fine grids
    arrays/opt_params_ctrl.npy, opt_params_schz.npy - fine optima
    results/grid_optima.csv, grid_fine_optima.csv
    figures/fig05_grid_heatmaps.png
    figures/fig05b_fine_grid_heatmaps.png

Runtime: ~12 minutes with parallelisation

Code inspired by / adapted from following sources:
    https://github.com/neurolib-dev/neurolib
    https://github.com/OpenSourceBrain/WilsonCowan
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PATHS, ensure_dirs
from wc_model import simulate_fc, fc_fit
ensure_dirs()

ARR_DIR     = PATHS['arrays']
FIG_DIR     = PATHS['figures']
RESULTS_DIR = PATHS['results']

SIM_DURATION = 60 * 1000  # 60 seconds per simulation


# - Helper functions

def run_one(SC_mean, FC_emp, k_gl, c_ei):
    """Run a single WC simulation and return fit to empirical FC.

    Args:
        SC_mean: Group-mean structural connectivity matrix.
        FC_emp: Group-mean empirical functional connectivity.
        k_gl: Global coupling strength.
        c_ei: Excitatory-to-inhibitory coupling weight.

    Returns:
        Float: Pearson r between simulated and empirical FC, or NaN on failure.
    """
    sim_fc, success = simulate_fc(SC_mean, K_gl=k_gl, c_excinh=c_ei,
                                   duration=SIM_DURATION)
    if not success:
        return np.nan
    return fc_fit(sim_fc, FC_emp)


def grid_search(SC_mean, FC_emp, K_range, C_range, label):
    """Run a full 2D grid search for one group.

    Args:
        SC_mean: Group-mean SC matrix.
        FC_emp: Group-mean empirical FC (raw Pearson r).
        K_range: Array of K_gl values to sweep.
        C_range: Array of c_excinh values to sweep.
        label: Descriptive string for progress messages.

    Returns:
        2D array of fit values, shape (len(K_range), len(C_range)).
    """
    n_combos = len(K_range) * len(C_range)
    print(f"\n  Grid search for {label} ({len(K_range)}x{len(C_range)} = "
          f"{n_combos} simulations) ...")

    jobs = [(i, j, k, c) for i, k in enumerate(K_range)
                          for j, c in enumerate(C_range)]

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_one)(SC_mean, FC_emp, k, c)
        for _, _, k, c in jobs
    )

    grid = np.full((len(K_range), len(C_range)), np.nan)
    for idx, (i, j, _, _) in enumerate(jobs):
        grid[i, j] = results[idx]
    return grid


def find_optimum(grid, K_range, C_range, label):
    """Find the peak fit in a grid and warn if it sits at the boundary.

    Args:
        grid: 2D array of fit values.
        K_range: Array of K_gl values corresponding to grid rows.
        C_range: Array of c_excinh values corresponding to grid columns.
        label: Descriptive string for the group.

    Returns:
        Tuple of (k_opt, c_opt, r_opt).
    """
    valid = np.nan_to_num(grid, nan=-999)
    i_opt, j_opt = np.unravel_index(valid.argmax(), grid.shape)
    k_opt = K_range[i_opt]
    c_opt = C_range[j_opt]
    r_opt = grid[i_opt, j_opt]
    at_boundary = (i_opt == 0 or i_opt == len(K_range)-1 or
                   j_opt == 0 or j_opt == len(C_range)-1)
    boundary_warn = " ** AT BOUNDARY **" if at_boundary else ""
    print(f"  {label} optimum: K_gl={k_opt:.4f}, c_excinh={c_opt:.4f}, "
          f"r={r_opt:.4f}{boundary_warn}")
    return k_opt, c_opt, r_opt


# - Plotting grids

def plot_grids(grid_ctrl, grid_schz, K_range, C_range,
               k_ctrl, c_ctrl, k_schz, c_schz, fig_name):
    """Save a 3-panel heatmap figure (ctrl, SCZ, difference).

    Args:
        grid_ctrl: 2D fit array for controls.
        grid_schz: 2D fit array for SCZ.
        K_range: Array of K_gl values.
        C_range: Array of c_excinh values.
        k_ctrl: Optimal K_gl for controls.
        c_ctrl: Optimal c_excinh for controls.
        k_schz: Optimal K_gl for SCZ.
        c_schz: Optimal c_excinh for SCZ.
        fig_name: Output filename.
    """
    grid_diff = grid_ctrl - grid_schz
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    grids  = [grid_ctrl, grid_schz, grid_diff]
    titles = ['Controls', 'SCZ', 'Ctrl - SCZ (difference)']
    cmaps  = ['viridis', 'viridis', 'RdBu_r']

    for ax, g, title, cmap in zip(axes, grids, titles, cmaps):
        if 'diff' in title.lower():
            vmax = np.nanmax(np.abs(g))
            vmin = -vmax
        else:
            vmin, vmax = np.nanmin(g), np.nanmax(g)
        im = ax.imshow(g, origin='lower', aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       extent=[C_range[0], C_range[-1],
                               K_range[0], K_range[-1]])
        ax.set_xlabel('c_excinh (E->I)', fontsize=12)
        ax.set_ylabel('K_gl (global coupling)', fontsize=12)
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, label='Pearson r (simFC vs empFC)')

    axes[0].plot(c_ctrl, k_ctrl, 'r*', markersize=15)
    axes[1].plot(c_schz, k_schz, 'r*', markersize=15)

    fig_path = os.path.join(FIG_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved to {fig_path}")


# - Main run

def main():
    """Execute the two-stage parametre grid search."""
    SC_ctrl_mean = np.load(os.path.join(ARR_DIR, 'SC_ctrl_mean.npy'))
    SC_schz_mean = np.load(os.path.join(ARR_DIR, 'SC_schz_mean.npy'))
    FC_ctrl_emp  = np.load(os.path.join(ARR_DIR, 'FC_ctrl_raw_mean.npy'))
    FC_schz_emp  = np.load(os.path.join(ARR_DIR, 'FC_schz_raw_mean.npy'))

    # - Stage 1: Coarse grid
    K_GL_COARSE     = np.linspace(1.0, 8.0, 12)
    C_EXCINH_COARSE = np.linspace(5.0, 25.0, 12)

    print("=== Stage 1: Coarse grid ===")
    grid_ctrl_c = grid_search(SC_ctrl_mean, FC_ctrl_emp,
                              K_GL_COARSE, C_EXCINH_COARSE, "Controls")
    grid_schz_c = grid_search(SC_schz_mean, FC_schz_emp,
                              K_GL_COARSE, C_EXCINH_COARSE, "SCZ")

    np.save(os.path.join(ARR_DIR, 'grid_ctrl.npy'), grid_ctrl_c)
    np.save(os.path.join(ARR_DIR, 'grid_schz.npy'), grid_schz_c)

    print("\nCoarse grid optima:")
    k_ctrl_c, c_ctrl_c, r_ctrl_c = find_optimum(
        grid_ctrl_c, K_GL_COARSE, C_EXCINH_COARSE, "Controls")
    k_schz_c, c_schz_c, r_schz_c = find_optimum(
        grid_schz_c, K_GL_COARSE, C_EXCINH_COARSE, "SCZ")

    df_coarse = pd.DataFrame({
        'group': ['ctrl', 'schz'],
        'K_gl_opt': [k_ctrl_c, k_schz_c],
        'c_excinh_opt': [c_ctrl_c, c_schz_c],
        'best_fit_r': [r_ctrl_c, r_schz_c]
    })
    df_coarse.to_csv(os.path.join(RESULTS_DIR, 'grid_optima.csv'), index=False)

    plot_grids(grid_ctrl_c, grid_schz_c, K_GL_COARSE, C_EXCINH_COARSE,
               k_ctrl_c, c_ctrl_c, k_schz_c, c_schz_c,
               'fig05_grid_heatmaps.png')

    # - Stage 2: Fine grid centred on coarse optima
    # Use the average of both groups' coarse optima as the centre
    k_centre = (k_ctrl_c + k_schz_c) / 2.0
    c_centre = (c_ctrl_c + c_schz_c) / 2.0
    K_GL_FINE     = np.linspace(k_centre - 1.0, k_centre + 1.0, 10)
    C_EXCINH_FINE = np.linspace(c_centre - 3.0, c_centre + 3.0, 10)

    print("\n=== Stage 2: Fine grid ===")
    print(f"  K_gl range:     [{K_GL_FINE[0]:.2f}, {K_GL_FINE[-1]:.2f}]")
    print(f"  c_excinh range: [{C_EXCINH_FINE[0]:.2f}, {C_EXCINH_FINE[-1]:.2f}]")

    grid_ctrl_f = grid_search(SC_ctrl_mean, FC_ctrl_emp,
                              K_GL_FINE, C_EXCINH_FINE, "Controls")
    grid_schz_f = grid_search(SC_schz_mean, FC_schz_emp,
                              K_GL_FINE, C_EXCINH_FINE, "SCZ")

    np.save(os.path.join(ARR_DIR, 'fine_grid_ctrl.npy'), grid_ctrl_f)
    np.save(os.path.join(ARR_DIR, 'fine_grid_schz.npy'), grid_schz_f)

    print("\nFine grid optima:")
    k_ctrl_f, c_ctrl_f, r_ctrl_f = find_optimum(
        grid_ctrl_f, K_GL_FINE, C_EXCINH_FINE, "Controls")
    k_schz_f, c_schz_f, r_schz_f = find_optimum(
        grid_schz_f, K_GL_FINE, C_EXCINH_FINE, "SCZ")

    # Key comparison
    step_c = C_EXCINH_FINE[1] - C_EXCINH_FINE[0]
    c_same = abs(c_ctrl_f - c_schz_f) < step_c
    print(f"\n  c_excinh* same at fine resolution? "
          f"{'YES' if c_same else 'NO'} "
          f"(ctrl={c_ctrl_f:.4f}, schz={c_schz_f:.4f}, "
          f"diff={abs(c_ctrl_f - c_schz_f):.4f}, step={step_c:.4f})")
    print(f"  K_gl* difference: {abs(k_ctrl_f - k_schz_f):.4f}")

    df_fine = pd.DataFrame({
        'group': ['ctrl', 'schz'],
        'K_gl_opt': [k_ctrl_f, k_schz_f],
        'c_excinh_opt': [c_ctrl_f, c_schz_f],
        'best_fit_r': [r_ctrl_f, r_schz_f]
    })
    df_fine.to_csv(os.path.join(RESULTS_DIR, 'grid_fine_optima.csv'),
                   index=False)

    # Save fine optima for downstream scripts
    np.save(os.path.join(ARR_DIR, 'opt_params_ctrl.npy'),
            np.array([k_ctrl_f, c_ctrl_f]))
    np.save(os.path.join(ARR_DIR, 'opt_params_schz.npy'),
            np.array([k_schz_f, c_schz_f]))

    plot_grids(grid_ctrl_f, grid_schz_f, K_GL_FINE, C_EXCINH_FINE,
               k_ctrl_f, c_ctrl_f, k_schz_f, c_schz_f,
               'fig05b_fine_grid_heatmaps.png')

    print("\n[DONE] 04_grid_search.py completed successfully.")


if __name__ == '__main__':
    main()
