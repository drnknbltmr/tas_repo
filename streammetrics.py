import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas

# Read data
case = 3

# 1) load data
u_data, v_data, foil_extent = tas.read_npz(case,'data_files/dewarped_data.npz')

u_data, v_data = tas.frame_process(u_data, v_data,-1)

# 2) Extract domain extents
x_min, x_max = foil_extent[0], foil_extent[1]
y_min, y_max = foil_extent[2], foil_extent[3]

# 3) Compute a simple streamfunction
Ny, Nx = u_data.shape
dx = (x_max - x_min) / (Nx - 1)
dy = (y_max - y_min) / (Ny - 1)

psi = np.zeros_like(u_data)

x_coords = np.linspace(x_min, x_max, Nx)
y_coords = np.linspace(y_min, y_max, Ny)

x_points = []
y_points = []

dv_dx = np.gradient(v_data, dx, axis=1)
du_dy = np.gradient(u_data, dy, axis=0)
stream_matrix = du_dy - dv_dx

print(stream_matrix)


for i in range(0, stream_matrix.shape[0]-1):
    for j in range(0, stream_matrix.shape[1]-1):
        if np.abs(stream_matrix[i,j]) < 0.1:
            x_points.append(x_coords[i])
            y_points.append(y_coords[j])

x_points = np.array(x_points)
y_points = np.array(y_points)

print(x_points)

# Plotting the regressed polynomial
plt.figure(figsize=(20, 12))
# Create the heatmap of u_data
# plt.imshow(
#     u_data,
#     extent=foil_extent,  # Set the extent to match the foil dimensions
#     origin='lower',      # Ensure the origin is at the bottom-left
#     cmap='jet',      # Use a colormap (you can change this to any other colormap)
#     aspect='equal'        # Automatically adjust the aspect ratio
# )

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_points, y_points, s=4, c='red', label='Data Points')
plt.title('Cubic Regression of Points Where Velocity First Drops Below Zero')
plt.xlim(foil_extent[0], foil_extent[1])
plt.ylim(foil_extent[2], foil_extent[3])
plt.gca().set_aspect('equal')  # Maintain aspect ratio
plt.legend()
plt.grid(True)


# # 5) plot heat map
# tas.heat_maps(case, u_data_lsb, v_data_lsb, foil_extent)
tas.send_plot('metrics')

