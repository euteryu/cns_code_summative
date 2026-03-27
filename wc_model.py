"""
wc_model.py - Wilson-Cowan Neural Mass Model

Shared simulation library. Import with:
    from wc_model import simulate_fc, fc_fit

Implements a whole-brain Wilson-Cowan model using neurolib's WCModel,
with low-pass filtered (<50 Hz) excitatory timeseries as the simulated
FC proxy.

The model takes a structural connectivity (SC) matrix and produces a
simulated functional connectivity (FC) matrix. Key design decisions:

  - sigma_ou = 0.01 (low noise): empirical testing showed that the default
    noise level (0.14) drowns out SC-driven structure in the FC. Low noise
    allows structural connectivity to shape the dynamics, yielding r ~ 0.3
    fits to empirical FC.

  - FC from excitatory rates, not BOLD: the Balloon-Windkessel BOLD model
    in neurolib produces too few timepoints at practical simulation lengths
    (5 points for 10s). Excitatory timeseries correlation is the standard
    approach in the professor's reference code and the wider literature.

  - Low-pass filter at 50 Hz: removes unstructured high-frequency noise
    whilst preserving the SC-driven oscillatory dynamics (~5-50 Hz).
    Cutoffs below ~10 Hz destroy the structured signal because the WC
    model's coupling-dependent dynamics operate at neural, not BOLD,
    timescales.

  - Uniform Dmat with signalV=0: w/o real fibre length data, a uniform
    distance matrix is used and propagation delays are disabled. This is
    equivalent to assuming instantaneous signal transmission.

References:
    Wilson & Cowan (1972) - original model.
    Hansen et al. (2015) - whole-brain modelling framework.
    Kim et al. (2013) - WC applied to SCZ oscillations.

Code inspired by / adapted from following sources:
    https://github.com/zijin-gu/scfc-coupling
    https://github.com/neurolib-dev/neurolib
    https://github.com/OpenSourceBrain/WilsonCowan
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from neurolib.models.wc import WCModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PATHS, ensure_dirs, fc_upper_triangle_fit

# Alias for convenience
fc_fit = fc_upper_triangle_fit

np.random.seed(42)

# Default WC parametres (neurolib convention)
DEFAULT_PARAMS = {
    'duration': 60 * 1000,    # 60 s in ms - long enough for stable FC
    'dt': 0.1,                # integration timestep (ms)
    'K_gl': 4.0,              # global coupling strength
    'c_excexc': 16.0,         # E->E self-excitation (wEE)
    'c_inhexc': 12.0,         # I->E suppression (wEI)
    'c_excinh': 15.0,         # E->I drive (wIE) - swept for E/I hypothesis
    'c_inhinh': 3.0,          # I->I self-inhibition (wII)
    'exc_ext': 0.65,          # external input to excitatory population
    'inh_ext': 0.0,           # external input to inhibitory population
    'sigma_ou': 0.01,         # LOW noise - critical for SC to shape FC
    'signalV': 0,             # disable propagation delays (no real Dmat)
}

# - Simulation constants
# ~4 seconds of simulation time at dt=0.1 ms, chosen to balance stability
# against stochastic variation in the FC estimate.
FC_TAIL = 40_000


# - Low-pass filter

def lowpass_filter(timeseries, cutoff_hz=50.0, fs_hz=10_000.0, order=4):
    """Apply a Butterworth low-pass filter to each row of a timeseries.

    Removes unstructured high-frequency noise whilst preserving the
    SC-driven oscillatory dynamics (~5-50 Hz). Cutoffs below ~10 Hz
    destroy the structured signal because the WC model's coupling-
    dependent dynamics operate at neural timescales.

    Args:
        timeseries: Array of shape (n_nodes, n_timepoints).
        cutoff_hz: Filter cutoff frequency in Hz.
        fs_hz: Sampling rate, 1/(dt in seconds) = 1/0.0001 = 10 000 Hz.
        order: Butterworth filter order.

    Returns:
        Filtered array with the same shape as input.
    """
    nyq = fs_hz / 2.0
    norm_cutoff = cutoff_hz / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, timeseries, axis=1)


# - Main simulation function

def simulate_fc(SC_matrix, K_gl=None, c_excinh=None, duration=None,
                params_override=None):
    """Run Wilson-Cowan on the given SC matrix and return simulated FC.

    FC is computed from low-pass filtered (<50 Hz) excitatory timeseries
    (last FC_TAIL timepoints).

    Args:
        SC_matrix: Structural connectivity (n_nodes, n_nodes), normalised
            with values in [0, 1] and diagonal = 0.
        K_gl: Global coupling strength (overrides default).
        c_excinh: E->I coupling (overrides default).
        duration: Simulation duration in ms (overrides default).
        params_override: Dict of any additional parametre overrides.

    Returns:
        Tuple of (sim_fc, success):
            sim_fc: Simulated FC matrix (n_nodes, n_nodes), Pearson
                correlation with diagonal set to 0.
            success: Bool indicating whether the simulation produced
                valid (non-NaN) output.

    Notes:
        The BOLD vs excitatory timeseries trade-off: neurolib's built-in
        Balloon-Windkessel BOLD model produces too few timepoints at
        practical simulation lengths (~5 points for 10 s). We therefore
        use Pearson correlation of filtered excitatory rates, which is the
        standard approach in the reference code and produces r~0.3 fits
        to empirical FC.
    """
    N = SC_matrix.shape[0]

    # Uniform distance matrix (no real fibre lengths available)
    Dmat = np.ones((N, N)) * 250.0
    np.fill_diagonal(Dmat, 0)

    Cmat = SC_matrix.copy()
    np.fill_diagonal(Cmat, 0)

    model = WCModel(Cmat=Cmat, Dmat=Dmat)

    for key, val in DEFAULT_PARAMS.items():
        model.params[key] = val

    if K_gl is not None:
        model.params['K_gl'] = K_gl
    if c_excinh is not None:
        model.params['c_excinh'] = c_excinh
    if duration is not None:
        model.params['duration'] = duration
    if params_override:
        for key, val in params_override.items():
            model.params[key] = val

    try:
        model.run(chunkwise=True)
        exc = model.exc

        n_tp = exc.shape[1]
        tail = min(FC_TAIL, n_tp // 2)
        exc_used = exc[:, -tail:]

        if exc_used.shape[1] < 100:
            return np.zeros((N, N)), False

        exc_filtered = lowpass_filter(exc_used)
        sim_fc = np.corrcoef(exc_filtered)

        if np.any(np.isnan(sim_fc)):
            return np.zeros((N, N)), False

        np.fill_diagonal(sim_fc, 0)
        return sim_fc, True

    except Exception as e:
        print(f"  Simulation failed: {e}")
        return np.zeros((N, N)), False


# - Sanity check (runs when executed directly)

def main():
    """Run a single sanity-check simulation with group-mean ctrl SC."""
    ensure_dirs()
    ARR_DIR = PATHS['arrays']
    FIG_DIR = PATHS['figures']

    SC_ctrl_mean = np.load(os.path.join(ARR_DIR, 'SC_ctrl_mean.npy'))
    FC_emp = np.load(os.path.join(ARR_DIR, 'FC_ctrl_raw_mean.npy'))

    print("Running sanity-check simulation (group-mean ctrl SC, 60 s) ...")
    sim_fc, success = simulate_fc(SC_ctrl_mean, K_gl=4.0, c_excinh=15.0)

    if not success:
        print("WARNING: simulation did not produce valid FC.")
        return

    r = fc_fit(sim_fc, FC_emp)
    print(f"Simulated FC vs Empirical FC (raw): r = {r:.4f}")

    triu = np.triu_indices(sim_fc.shape[0], k=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(sim_fc, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Simulated FC', fontsize=13)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(FC_emp, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Empirical FC (ctrl mean, raw)', fontsize=13)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].scatter(FC_emp[triu], sim_fc[triu], s=3, alpha=0.3, color='steelblue')
    axes[2].set_xlabel('Empirical FC (raw Pearson r)', fontsize=12)
    axes[2].set_ylabel('Simulated FC', fontsize=12)
    axes[2].set_title(f'Edge-wise comparison (r={r:.3f})', fontsize=13)
    lims = [min(axes[2].get_xlim()[0], axes[2].get_ylim()[0]),
            max(axes[2].get_xlim()[1], axes[2].get_ylim()[1])]
    axes[2].plot(lims, lims, 'k--', alpha=0.3)

    fig_path = os.path.join(FIG_DIR, 'fig04_sanity_check.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
    print("\n[DONE] wc_model.py sanity check completed successfully.")


if __name__ == '__main__':
    main()
