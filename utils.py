"""
utils.py - Shared utilities for the SCZ SC-FC pipeline

Centralises path constants, statistical helper functions, and connectivity
analysis utilities used across all pipeline scripts.

Import with:
    from utils import PATHS, ensure_dirs, cohens_d, choose_test
    from utils import subject_scfc_coupling, within_module_mask

Code inspired by / adapted from following sources:
    https://github.com/johannaleapopp/SC_FC_Coupling_Task_Intelligence
    https://github.com/zijin-gu/scfc-coupling
    https://github.com/netneurolab/liu_meg-scfc
    https://github.com/manishrami/Cohens_d
"""

import os
import numpy as np
from scipy import stats


# All scripts import PATHS rather than defining their own path constants.
# DEAR EXAMINER: Please change root to match wherever you download; see directory_structure.txt for project hierarchy.
PROJECT_ROOT = r'C:\Users\minse\Downloads\cns_research'

PATHS = {
    'project':  PROJECT_ROOT,
    'data':     os.path.join(PROJECT_ROOT, 'dataset'),
    'code':     os.path.join(PROJECT_ROOT, 'code'),
    'results':  os.path.join(PROJECT_ROOT, 'code', 'results'),
    'figures':  os.path.join(PROJECT_ROOT, 'code', 'results', 'figures'),
    'arrays':   os.path.join(PROJECT_ROOT, 'code', 'results', 'arrays'),
    'mat':      os.path.join(PROJECT_ROOT, 'dataset', '27_SCHZ_CTRL_dataset.mat'),
    'demog':    os.path.join(PROJECT_ROOT, 'dataset', '27_SCHZ_CTRL_demographics.xlsx'),
    'parc':     os.path.join(PROJECT_ROOT, 'dataset', 'ParcellationLausanne2008.xls'),
    'nii_33':   os.path.join(PROJECT_ROOT, 'dataset', 'ROI_nii', 'ROIv_scale33.nii.gz'),
}


def ensure_dirs():
    """Create output dirs if they do not already exist."""
    for key in ('results', 'figures', 'arrays'):
        os.makedirs(PATHS[key], exist_ok=True)


# Stats helpers

def cohens_d(x, y):
    """Compute pooled-SD Cohen's d btwn two independent groups.

    Positive d means group x (typically controls) has a higher mean than
    group y (typically SCZ).

    Args:
        x: Array of values for group 1 (e.g. controls).
        y: Array of values for group 2 (e.g. SCZ).

    Returns:
        Float: Cohen's d effect size (pooled standard deviation).
    """
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 +
                          (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def choose_test(x, y, alpha=0.05):
    """Choose and run the appropriate two-sample comparison test.

    Both groups must be normal at the given alpha level for the parametric
    t-test to be used; otherwise falls back to Mann-Whitney U.

    Args:
        x: Array of values for group 1.
        y: Array of values for group 2.
        alpha: Significance threshold for the Shapiro-Wilk normality check.

    Returns:
        Tuple of (test_statistic, p_value, test_name).
    """
    _, p_x = stats.shapiro(x)
    _, p_y = stats.shapiro(y)
    both_normal = (p_x > alpha) and (p_y > alpha)

    if both_normal:
        stat, p = stats.ttest_ind(x, y)
        return stat, p, "Independent t-test"
    else:
        stat, p = stats.mannwhitneyu(x, y, alternative='two-sided')
        return stat, p, "Mann-Whitney U"


# Connectivity analysis

def subject_scfc_coupling(SC_stack, FC_stack):
    """Compute per-subject SC-FC coupling as Pearson r of upper-triangle edges.

    Both matrices must be in raw Pearson r / normalised space (not Fisher
    z-transformed) for a valid like-for-like comparison.

    Args:
        SC_stack: Structural connectivity, shape (n_subj, n_nodes, n_nodes).
        FC_stack: Functional connectivity, shape (n_subj, n_nodes, n_nodes).

    Returns:
        Array of shape (n_subj,) with one coupling value per subject.
    """
    n_subj = SC_stack.shape[0]
    triu_idx = np.triu_indices(SC_stack.shape[1], k=1)
    couplings = np.zeros(n_subj)
    for i in range(n_subj):
        couplings[i] = np.corrcoef(SC_stack[i][triu_idx],
                                    FC_stack[i][triu_idx])[0, 1]
    return couplings


def within_module_mask(module_labels, module_id, n_nodes):
    """Build an upper-triangle boolean mask for edges within a module.

    Args:
        module_labels: Integer array (n_nodes,) assigning each node to a module.
        module_id: Which module to select.
        n_nodes: Total number of nodes.

    Returns:
        Boolean array (n_nodes, n_nodes), True only for upper-triangle edges
        where both nodes belong to the specified module.
    """
    nodes = np.where(module_labels == module_id)[0]
    mask = np.zeros((n_nodes, n_nodes), dtype=bool)
    for i_idx in range(len(nodes)):
        for j_idx in range(i_idx + 1, len(nodes)):
            mask[nodes[i_idx], nodes[j_idx]] = True
    return mask


def within_module_coupling(SC_stack, FC_stack, mask):
    """Compute per-subject Pearson r between SC and FC edges within a module.

    Args:
        SC_stack: Structural connectivity, shape (n_subj, n_nodes, n_nodes).
        FC_stack: Functional connectivity, shape (n_subj, n_nodes, n_nodes).
        mask: Boolean mask from within_module_mask(), selecting edges.

    Returns:
        Array of shape (n_subj,) with within-module coupling per subject.
    """
    idx = np.where(mask)
    n_subj = SC_stack.shape[0]
    couplings = np.zeros(n_subj)
    for i in range(n_subj):
        sc_vals = SC_stack[i][idx]
        fc_vals = FC_stack[i][idx]
        if sc_vals.std() == 0 or fc_vals.std() == 0:
            couplings[i] = 0.0
        else:
            couplings[i] = np.corrcoef(sc_vals, fc_vals)[0, 1]
    return couplings


def fc_upper_triangle_fit(sim_fc, emp_fc):
    """Pearson r between upper triangles of simulated and empirical FC.

    Args:
        sim_fc: Simulated FC matrix (n_nodes, n_nodes).
        emp_fc: Empirical FC matrix (n_nodes, n_nodes).

    Returns:
        Float: Pearson correlation between vectorised upper-triangle edges.
    """
    triu = np.triu_indices(sim_fc.shape[0], k=1)
    return np.corrcoef(sim_fc[triu], emp_fc[triu])[0, 1]


# Alias for backwards compatibility from legacy code iterations just in case...!
fc_fit = fc_upper_triangle_fit
