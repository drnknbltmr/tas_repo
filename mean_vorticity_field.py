import os
import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas

dx = 0.0004
dy = 0.0004


def vorticity_heatmap_png(case, vorticity_data, foil_extent, dpi,vmin,vmax):
    # Create output folder if it doesn't exist
    output_folder = 'vorticity_graphs'
    os.makedirs(output_folder, exist_ok=True)

    # === Vorticity Heatmap ===
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.jet.copy()
    cmap.set_bad('white')

    im = ax.imshow(vorticity_data, extent=foil_extent, aspect='equal', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'Case {case} - Vorticity')
    ax.set_xlabel('Arc Length (m)')
    ax.set_ylabel('Distance from Airfoil (m)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.4)
    cbar.set_label('Vorticity (1/s)')

    plt.tight_layout()

    # Save inside 'vorticity graphs' folder
    save_path = os.path.join(output_folder, f'vorticity_case_{case}.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

for case in [1,2,3,4,5]:
    if case == 1:
        u_data, v_data, foil_extent = tas.read_npz(case, 'data_files/dewarped_data.npz')
        vorticity = tas.mean_vorticity(u_data, v_data, dx, dy)
        vorticity = tas.outlier_filter(vorticity)
        vorticity[vorticity > 200] = np.nan
        vorticity_masked = np.ma.masked_invalid(vorticity)
        vmin = vorticity_masked.min()
        vmax = vorticity_masked.max()
    else:
        u_data, v_data, foil_extent = tas.read_npz_loop(case, 'data_files/dewarped_data.npz')
        vorticity = tas.mean_vorticity(u_data, v_data, dx, dy)
        vorticity = tas.outlier_filter(vorticity)
        vorticity[vorticity > 200] = np.nan
        vorticity_masked = np.ma.masked_invalid(vorticity)
        if vorticity_masked.min()<vmin:
            vmin = vorticity_masked.min()
        if vorticity_masked.max()>vmax:
            vmax = vorticity_masked.max()
    # choose which timeframe you want
for case in [1,2,3,4,5]:
    u_data, v_data, foil_extent = tas.read_npz_loop(case, 'data_files/dewarped_data.npz')
    vorticity = tas.mean_vorticity(u_data, v_data, dx, dy)
    vorticity = tas.outlier_filter(vorticity)
    vorticity[vorticity > 200] = np.nan
    vorticity_heatmap_png(case, vorticity, foil_extent, 1000,vmin,vmax)









