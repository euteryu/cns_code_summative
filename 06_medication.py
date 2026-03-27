"""
06_medication.py - Medication Moderator Analysis

Checks whether antipsychotic medication (CPZ equivalents if available)
correlates with individual SC-FC coupling scores in the SCZ group.

Scientific question: Could medication dosage be a confound driving the
observed SC-FC coupling patterns?

Inputs:
    dataset/27_SCHZ_CTRL_demographics.xlsx
    arrays/coupling_schz.npy, FC_schz.npy

Outputs:
    results/medication_analysis.csv
    figures/fig07_medication.png

Runtime: ~5 seconds
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PATHS, ensure_dirs
ensure_dirs()

ARR_DIR     = PATHS['arrays']
FIG_DIR     = PATHS['figures']
RESULTS_DIR = PATHS['results']
DEMOG_PATH  = PATHS['demog']


# - Helpers

def _find_medication_column(demog):
    """Identify the most likely medication column in the demographics table.

    Args:
        demog: DataFrame of participant demographics.

    Returns:
        Column name (str) or None if no suitable column is found.
    """
    med_candidates = [c for c in demog.columns
                      if any(kw in c.lower() for kw in
                             ['med', 'cpz', 'chlorpromazine', 'dose',
                              'antipsychotic', 'drug', 'rx'])]

    med_col = None
    for candidate in med_candidates:
        vals = pd.to_numeric(demog[candidate], errors='coerce')
        if vals.notna().sum() > 5:
            med_col = candidate
            break
    if med_col is None and med_candidates:
        med_col = med_candidates[0]
    return med_col


def _find_group_column(demog):
    """Identify the most likely group/diagnosis column.

    Args:
        demog: DataFrame of participant demographics.

    Returns:
        Column name (str) or None if no suitable column is found.
    """
    group_candidates = [c for c in demog.columns
                        if any(kw in c.lower() for kw in
                               ['group', 'diag', 'status', 'type'])]
    return group_candidates[0] if group_candidates else None


# - Main run

def main():
    """Run medication moderator analysis for the SCZ group."""
    # - Load demographics
    demog = pd.read_excel(DEMOG_PATH)
    print("Demographics columns:", demog.columns.tolist())
    print("\nFirst 5 rows:")
    print(demog.head())
    print(f"\nTotal rows: {len(demog)}")

    # - Identify medication column
    med_col = _find_medication_column(demog)

    # - Identify group column
    group_col = _find_group_column(demog)

    # - Load SCZ data
    coup_schz = np.load(os.path.join(ARR_DIR, 'coupling_schz.npy'))
    FC_schz = np.load(os.path.join(ARR_DIR, 'FC_schz.npy'))
    fc_strength_schz = FC_schz.mean(axis=(1, 2))
    n_schz = len(coup_schz)

    if med_col is None:
        print("\nWARNING: No medication data found.")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(coup_schz, bins=10, color='salmon', edgecolor='k', alpha=0.7)
        ax.set_xlabel('SC-FC Coupling', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('SCZ SC-FC Coupling Distribution\n'
                      '(No medication data available)', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig07_medication.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("\n[DONE] 06_medication.py completed (no medication column found).")
        return

    # - Extract SCZ medication values
    if group_col is not None:
        groups = demog[group_col].astype(str).str.lower()
        schz_mask = groups.str.contains('schz|scz|sz|patient|1', regex=True)
        schz_rows = demog[schz_mask].reset_index(drop=True)
    else:
        schz_rows = demog.iloc[n_schz:2*n_schz].reset_index(drop=True)

    med_values = pd.to_numeric(schz_rows[med_col], errors='coerce').values
    n_valid = min(len(med_values), n_schz)
    med_values = med_values[:n_valid]
    coup_vals = coup_schz[:n_valid]
    fc_vals = fc_strength_schz[:n_valid]

    valid_mask = ~np.isnan(med_values)
    med_valid = med_values[valid_mask]
    coup_valid = coup_vals[valid_mask]
    fc_valid = fc_vals[valid_mask]

    print(f"\nMedication column: '{med_col}'")
    print(f"Valid medication values: {len(med_valid)} / {n_valid}")

    if len(med_valid) < 5:
        print("Too few valid medication values for correlation analysis.")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5,
                f'Insufficient medication data\n({len(med_valid)} valid values)',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(os.path.join(FIG_DIR, 'fig07_medication.png'), dpi=150)
        plt.close()
        print("\n[DONE] 06_medication.py completed (insufficient data).")
        return

    # - Spearman correlations
    rho_coup, p_coup = stats.spearmanr(med_valid, coup_valid)
    rho_fc, p_fc = stats.spearmanr(med_valid, fc_valid)
    print(f"\nMedication vs SC-FC coupling: Spearman rho={rho_coup:.4f}, "
          f"p={p_coup:.4f}")
    print(f"Medication vs mean FC strength: Spearman rho={rho_fc:.4f}, "
          f"p={p_fc:.4f}")

    df_results = pd.DataFrame({
        'analysis': ['medication_vs_coupling', 'medication_vs_fc_strength'],
        'spearman_rho': [rho_coup, rho_fc],
        'p_value': [p_coup, p_fc],
        'n': [len(med_valid), len(med_valid)]
    })
    df_results.to_csv(os.path.join(RESULTS_DIR, 'medication_analysis.csv'),
                      index=False)

    # - Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(med_valid, coup_valid, s=60, c='salmon',
                    edgecolors='k', alpha=0.7)
    slope, intercept = np.polyfit(med_valid, coup_valid, 1)
    x_line = np.linspace(med_valid.min(), med_valid.max(), 50)
    axes[0].plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5)
    axes[0].set_xlabel(f'Medication ({med_col})', fontsize=12)
    axes[0].set_ylabel('SC-FC Coupling', fontsize=12)
    axes[0].set_title(f'Medication vs Coupling\n'
                      f'(rho={rho_coup:.3f}, p={p_coup:.3f})', fontsize=13)

    axes[1].scatter(med_valid, fc_valid, s=60, c='mediumpurple',
                    edgecolors='k', alpha=0.7)
    slope2, intercept2 = np.polyfit(med_valid, fc_valid, 1)
    axes[1].plot(x_line, slope2 * x_line + intercept2, 'k--', alpha=0.5)
    axes[1].set_xlabel(f'Medication ({med_col})', fontsize=12)
    axes[1].set_ylabel('Mean FC Strength', fontsize=12)
    axes[1].set_title(f'Medication vs FC Strength\n'
                      f'(rho={rho_fc:.3f}, p={p_fc:.3f})', fontsize=13)

    fig_path = os.path.join(FIG_DIR, 'fig07_medication.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
    print("\n[DONE] 06_medication.py completed successfully.")


if __name__ == '__main__':
    main()
