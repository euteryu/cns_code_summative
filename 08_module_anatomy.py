"""
08_module_anatomy.py - Anatomical Labelling of Modules

Identifies which brain regions comprise each module, with Module 2
highlighted as the primary locus of SC-FC disruption in schizophrenia.

Coordinates are extracted from the Scale33 NIfTI parcellation and cached
for subsequent runs. Glass-brain visualisations are produced via nilearn
where available, with a matplotlib scatter fallback.

Inputs:
    arrays/module_labels.npy
    dataset/ParcellationLausanne2008.xls
    dataset/ROI_nii/ROIv_scale33.nii.gz

Outputs:
    results/module2_anatomy.txt
    figures/fig09a_module2_glass_brain.png
    figures/fig09b_all_modules_glass_brain.png

Code inspired by / adapted from following sources:
    https://github.com/netneurolab/liu_meg-scfc
    https://github.com/neurolib-dev/neurolib
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PATHS, ensure_dirs
ensure_dirs()

ARR_DIR     = PATHS['arrays']
FIG_DIR     = PATHS['figures']
RESULTS_DIR = PATHS['results']
DATA_DIR    = PATHS['data']
PARC_PATH   = PATHS['parc']
NII_SCALE33 = PATHS['nii_33']

N_MODULES = 5


def load_region_labels():
    """Extract Scale33 (83-node) region labels from the parcellation spreadsheet.

    The LABELS sheet uses row 0 as a header row with scale names.
    Scale33 labels reside in:
      - Right hemisphere: column index 9
      - Left hemisphere:  column index 19
    Data starts at row index 1 (after the sub-header row).

    Returns:
        numpy.ndarray: String array of region labels prefixed with `R_` or
        `L_` for hemisphere identification.
    """
    df = pd.read_excel(PARC_PATH, sheet_name='LABELS', header=None)
    # Row 0 is the sub-header; data from row 1 onward
    right_labels = df.iloc[1:, 9].dropna().astype(str)
    left_labels  = df.iloc[1:, 19].dropna().astype(str)

    # Prefix with R_/L_ for hemisphere identification
    right_labels = ('R_' + right_labels).values
    left_labels  = ('L_' + left_labels).values

    # Right hemisphere first, then left (matches Lausanne atlas convention)
    labels = np.concatenate([right_labels, left_labels])
    print(f"  Loaded {len(labels)} region labels "
          f"({len(right_labels)} R + {len(left_labels)} L)")
    return labels


def get_coordinates():
    """Obtain MNI coordinates for the 83 Scale33 nodes.

    Coordinates are computed from the centre of mass of each ROI in the
    NIfTI parcellation file, then cached to ``arrays/coords_scale33.npy``
    to avoid repeated extraction.

    Returns:
        numpy.ndarray: Array of shape ``(n_nodes, 3)`` with MNI coordinates.
    """
    coords_path = os.path.join(ARR_DIR, 'coords_scale33.npy')
    if os.path.exists(coords_path):
        print("  Loading cached coordinates ...")
        return np.load(coords_path)

    print("  Extracting coordinates from NIfTI (may take ~1 min) ...")
    from nilearn.image import load_img
    from scipy import ndimage

    img = load_img(NII_SCALE33)
    data = np.asarray(img.dataobj)
    affine = img.affine

    # Obtain unique non-zero ROI labels
    roi_ids = np.unique(data)
    roi_ids = roi_ids[roi_ids > 0]

    coords = []
    for roi_id in sorted(roi_ids):
        mask = data == roi_id
        com = ndimage.center_of_mass(mask)
        # Convert voxel coordinates to MNI space
        mni = affine @ np.array([com[0], com[1], com[2], 1.0])
        coords.append(mni[:3])

    coords = np.array(coords)
    np.save(coords_path, coords)
    print(f"  Extracted {len(coords)} coordinates, cached to {coords_path}")
    return coords


def main():
    """Run the full anatomical labelling pipeline for all modules."""
    module_labels = np.load(os.path.join(ARR_DIR, 'module_labels.npy'))
    n_nodes = len(module_labels)
    print(f"Module labels loaded: {n_nodes} nodes, {N_MODULES} modules")

    # -Step A: Region labels
    print("\nLoading region labels ...")
    region_names = load_region_labels()

    # Trim or pad to match node count
    if len(region_names) > n_nodes:
        region_names = region_names[:n_nodes]
    elif len(region_names) < n_nodes:
        extra = [f'Region_{i}' for i in range(len(region_names), n_nodes)]
        region_names = np.concatenate([region_names, extra])

    # - Step B: Coordinates
    print("\nGetting MNI coordinates ...")
    coords = get_coordinates()
    if len(coords) > n_nodes:
        coords = coords[:n_nodes]
    elif len(coords) < n_nodes:
        pad = np.zeros((n_nodes - len(coords), 3))
        coords = np.vstack([coords, pad])

    # - Step C: Module anatomy table
    output_lines = []
    for m in range(N_MODULES):
        nodes = np.where(module_labels == m)[0]
        names = sorted(region_names[nodes])
        header = f"Module {m} ({len(nodes)} nodes):"
        output_lines.append(header)
        output_lines.append("-" * len(header))
        for name in names:
            output_lines.append(f"  {name}")

        if m == 2:
            # Hemisphere distribution (labels prefixed with R_ / L_)
            n_left  = sum(1 for n in names if n.startswith('L_'))
            n_right = sum(1 for n in names if n.startswith('R_'))
            n_other = len(names) - n_left - n_right
            output_lines.append(
                f"  Hemisphere: L={n_left}, R={n_right}, unclear={n_other}")
        output_lines.append("")

    # Print and save
    anatomy_text = "\n".join(output_lines)
    print("\n" + anatomy_text)

    txt_path = os.path.join(RESULTS_DIR, 'module2_anatomy.txt')
    with open(txt_path, 'w') as f:
        f.write(anatomy_text)
    print(f"Anatomy table saved to {txt_path}")

    # - Step D: Module 2 interpretation
    m2_nodes = np.where(module_labels == 2)[0]
    m2_names = [region_names[i].lower() for i in m2_nodes]
    m2_text = " ".join(m2_names)

    association_kw = ['frontal', 'parietal', 'temporal', 'prefrontal',
                      'cingul', 'precuneus', 'angular', 'supramarginal',
                      'inferior_parietal', 'superior_frontal', 'middle_frontal']
    sensory_kw = ['occipital', 'calcarine', 'cuneus', 'lingual',
                  'precentral', 'postcentral', 'paracentral']

    n_assoc = sum(1 for kw in association_kw if kw in m2_text)
    n_sens  = sum(1 for kw in sensory_kw if kw in m2_text)

    print("\nModule 2 anatomical interpretation:")
    if n_assoc > n_sens:
        print("  Module 2 nodes are predominantly association cortex "
              "(prefrontal/parietal).")
        print("  This is CONSISTENT with Yang et al.'s cortical hierarchy "
              "account.")
    elif n_sens > n_assoc:
        print("  Module 2 nodes are predominantly sensorimotor/occipital.")
        print("  This points to a DIFFERENT disruption profile than "
              "Yang et al.'s hierarchy.")
    else:
        print("  Module 2 has a mixed composition. Interpret with caution.")

    # - Step E: Glass-brain figures
    try:
        from nilearn import plotting
        from matplotlib.colors import ListedColormap

        # Fig 09a: Module 2 highlighted
        m2_vals  = np.array([1.0 if m == 2 else 0.0 for m in module_labels])
        m2_sizes = np.array([60 if m == 2 else 20 for m in module_labels])
        cmap_m2  = ListedColormap(['#BBBBBB', 'red'])

        display_a = plotting.plot_markers(
            node_values=m2_vals,
            node_coords=coords,
            node_size=m2_sizes,
            node_cmap=cmap_m2,
            node_vmin=0, node_vmax=1,
            display_mode='ortho',
            title='Module 2 (red) highlighted -- 83 nodes, Scale33',
            colorbar=False,
        )
        fig_path_a = os.path.join(FIG_DIR, 'fig09a_module2_glass_brain.png')
        display_a.savefig(fig_path_a, dpi=150)
        display_a.close()
        print(f"Figure saved to {fig_path_a}")

        # Fig 09b: All modules coloured
        mod_vals = module_labels.astype(float)
        module_colours = ListedColormap(
            ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd'])

        display_b = plotting.plot_markers(
            node_values=mod_vals,
            node_coords=coords,
            node_size=40,
            node_cmap=module_colours,
            node_vmin=0, node_vmax=4,
            display_mode='ortho',
            title='All 5 modules -- 83 nodes, Scale33',
            colorbar=False,
        )
        fig_path_b = os.path.join(FIG_DIR, 'fig09b_all_modules_glass_brain.png')
        display_b.savefig(fig_path_b, dpi=150)
        display_b.close()
        print(f"Figure saved to {fig_path_b}")

    except Exception as e:
        print(f"\nNilearn glass brain failed ({e}). Using fallback scatter plot.")

        # Fallback: MNI scatter coloured by module; in case returning to legacy from earlier iteration...
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        module_cmap = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd']

        for m in range(N_MODULES):
            mask = module_labels == m
            c = 'red' if m == 2 else '#BBBBBB'
            s = 80 if m == 2 else 30
            axes[0].scatter(coords[mask, 0], coords[mask, 1], c=c, s=s,
                            edgecolors='k', linewidths=0.5, label=f'M{m}',
                            zorder=5 if m == 2 else 1)
        axes[0].set_xlabel('X (MNI)')
        axes[0].set_ylabel('Y (MNI)')
        axes[0].set_title('Module 2 highlighted (axial view)')

        for m in range(N_MODULES):
            mask = module_labels == m
            axes[1].scatter(coords[mask, 0], coords[mask, 1],
                            c=module_cmap[m], s=50, edgecolors='k',
                            linewidths=0.5, label=f'M{m}')
        axes[1].set_xlabel('X (MNI)')
        axes[1].set_ylabel('Y (MNI)')
        axes[1].set_title('All 5 modules (axial view)')
        axes[1].legend(fontsize=9)

        fig_path = os.path.join(FIG_DIR, 'fig09a_module2_glass_brain.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Fallback figure saved to {fig_path}")

    print("\n[DONE] 08_module_anatomy.py completed successfully.")


if __name__ == '__main__':
    main()
