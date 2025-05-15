import os
import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas

dx = 0.0004
dy = 0.0004

cases = [1, 2, 3, 4, 5]
frequencies = [0, 4, 8, 16, 300]
vorticity_all = []
foil_extents = []

# First pass: get global vmin and vmax
vmin, vmax = float('inf'), float('-inf')

for case in cases:
    if case == 1:
        u_data, v_data, foil_extent = tas.read_npz(case, 'data_files/dewarped_data.npz')
    else:
        u_data, v_data, foil_extent = tas.read_npz_loop(case, 'data_files/dewarped_data.npz')

    vorticity = tas.mean_vorticity(u_data, v_data, dx, dy)
    vorticity = tas.outlier_filter(vorticity)
    vorticity[vorticity > 200] = np.nan
    vorticity_masked = np.ma.masked_invalid(vorticity)

    vorticity_all.append(vorticity_masked)
    foil_extents.append(foil_extent)

    if vorticity_masked.min() < vmin:
        vmin = vorticity_masked.min()
    if vorticity_masked.max() > vmax:
        vmax = vorticity_masked.max()

# Plot in 2 columns, filling down columns first
n_rows = (len(cases) + 1) // 2
n_cols = 2
fig, axs = plt.subplots(n_rows, n_cols, figsize=(25, 5 * n_rows))
axs = axs.T.flatten()  # Transpose to fill columns vertically

cmap = plt.cm.jet.copy()
cmap.set_bad('white')

for idx, case in enumerate(cases):
    ax = axs[idx]
    im = ax.imshow(vorticity_all[idx], extent=foil_extents[idx], aspect='equal',
                   origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'Frequency - {frequencies[idx]} Hz', fontsize=12)
    ax.set_xlabel('Arc Length (m)')
    ax.set_ylabel('Distance from Airfoil (m)')

# Hide unused subplot if any
for ax in axs[len(cases):]:
    ax.axis('off')

# Adjust spacing
plt.subplots_adjust(hspace=0.01, wspace=0.1, bottom=0.1)

# Shared horizontal colorbar (moved lower)
cbar = fig.colorbar(im, ax=axs[:len(cases)], orientation='horizontal', shrink=0.8, pad=0.08)
cbar.set_label('Vorticity (1/s)')

# Save and send
os.makedirs('vorticity_graphs', exist_ok=True)
plt.savefig('vorticity_graphs/combined_vorticity_cases.png', dpi=1000)
tas.send_plot('meanvorticity')
