import numpy as np
import matplotlib.pyplot as plt
import os
import tas_functions as tas
from matplotlib.colors import Normalize
from scipy.interpolate import CloughTocher2DInterpolator

def main(frame):
    u_data, v_data, foil_extent = tas.read_npz(3, 'data_files/dewarped_data.npz')
    u_data, v_data = tas.frame_process(u_data, v_data, frame)
    return u_data, v_data, foil_extent


def fill_nans_near_wall(u, y_coords, max_wall_distance=0.0002):
    """Fill NaNs near the wall (y=0) with first valid velocity value"""
    u_filled = u.copy()
    for i in range(u.shape[1]):  # Loop over x-columns
        col = u[:, i]
        valid_idx = np.where(~np.isnan(col))[0]
        if len(valid_idx) == 0:
            continue
        first_valid = valid_idx[0]
        wall_region = (y_coords < max_wall_distance) & (y_coords < y_coords[first_valid])
        u_filled[wall_region, i] = col[first_valid]  # Extrapolate first valid value
    return u_filled

def replace_low_nans(matrix):
    num_rows, num_cols = matrix.shape
    for col_idx in range(num_cols):
        col = matrix[:, col_idx]
        not_nan_mask = ~np.isnan(col)
        if np.any(not_nan_mask):
            first_valid = np.argmax(not_nan_mask)
            matrix[:first_valid, col_idx] = 0
    return matrix

u_data, v_data, foil_extent = main(-1)
u_data_original = u_data

print('reading npz complete')
#Extract domain extents
x_min, x_max = foil_extent[0], foil_extent[1]
y_min, y_max = foil_extent[2], foil_extent[3]

# Compute grid spacing (assuming uniform grid)
Ny, Nx = u_data.shape
dx = (x_max - x_min) / (Nx - 1)
dy = (y_max - y_min) / (Ny - 1)

x_coords = np.linspace(x_min, x_max, Nx)
y_coords = np.linspace(y_min, y_max, Ny)


with np.errstate(invalid='ignore'):
    psi_int_y = np.nancumsum(u_data, axis=0) * dy
    d_psi_dx = np.gradient(psi_int_y, dx, axis=1)
    residual = d_psi_dx + np.nan_to_num(v_data, nan=0)
    psi_correction = np.nancumsum(residual, axis=1) * dx

psi = psi_int_y - psi_correction
psi = psi - np.nanmean(psi)  # Center around zero

# Create mask for valid data points
valid_mask = ~np.isnan(psi)
xx, yy = np.meshgrid(x_coords, y_coords)

# Scattered data interpolation setup
valid_points = np.column_stack([xx[valid_mask], yy[valid_mask]])
valid_psi = psi[valid_mask]

# Create high-res grid
x_fine = np.linspace(x_coords.min(), x_coords.max(), 800)
y_fine = np.linspace(y_coords.min(), y_coords.max(), 200)
xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)

# Natural neighbor interpolation
interp = CloughTocher2DInterpolator(valid_points, valid_psi)
psi_fine = interp(xx_fine, yy_fine)

print("ψ min:", np.nanmin(psi), "ψ max:", np.nanmax(psi))


# Plot setup
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 6), gridspec_kw={'height_ratios': [3, 1]})
norm = Normalize(vmin=-3, vmax=8)
cmap = 'jet'
im = ax.imshow(
            v_data,
            extent=foil_extent,  # Set the extent to match the foil dimensions
            origin='lower',      # Ensure the origin is at the bottom-left
            cmap=cmap,      # Use a colormap (you can change this to any other colormap)
            aspect='equal',
            norm=norm # Automatically adjust the aspect ratio
        )

im2 = ax2.imshow(
    psi,
    extent=foil_extent,
    origin='lower',
    cmap=cmap,
    aspect='equal'
)
ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')
ax2.set_title('v-velocity component')

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)  # empty mappable with the
sm.set_array([])
cbar = fig.colorbar(sm,
        ax=ax,
        orientation='vertical',
        fraction=0.046,  # how wide the bar is relative to the Axes
        pad=0.04, # gap between bar and Axes
        shrink=0.6
)
cbar.set_label('Velocity (m/s)')

# Contour parameters - small band around zero to catch all branches
levels = np.linspace(-1e-2, 1e-2, 10)  # Three levels centered on zero

# Plot contours
cs = ax.contour(x_coords, y_coords, psi, levels=levels, colors='black', linewidths=1.5)

# Formatting
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Streamfunction ψ=0 Contours')
ax.grid(True)
plt.tight_layout()
plt.show()
