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

x_coords = np.linspace(foil_extent[0], foil_extent[1], Nx)
y_coords = np.linspace(foil_extent[2], foil_extent[3], Ny)

for j in range(1, Ny):
    psi[j, 0] = psi[j-1, 0] + u_data[j-1, 0]*dy

# (b) Across each row
for j in range(Ny):
    for i in range(1, Nx):
        psi[j, i] = psi[j, i-1] - v_data[j, i-1]*dx + u_data[j, i-1]*dy

        if -0.1 < psi[j, i] < 0.1:
            x_points.append(x_coords[j])
            y_points.append(y_coords[i])


# # 4) Identify LSB region by psi < 0 (or whichever condition suits your flow)
# lsb_mask = (abs(psi) < 0.1)
# u_data_lsb = np.where(lsb_mask, u_data, np.nan)
# v_data_lsb = np.where(lsb_mask, v_data, np.nan)
# # x_data_lsb = np.where(lsb_mask, x_data, np.nan)
# # y_data_lsb = np.where(lsb_mask, y_data, np.nan)

x_points = np.array(x_points)
y_points = np.array(y_points)

# Perform cubic regression
coefficients = np.polyfit(x_points, y_points, 3)  # Fit a cubic polynomial
polynomial = np.poly1d(coefficients)  # Create a polynomial function

# Generate x values for plotting the polynomial, limited to the range of x_points
x_fit = np.linspace(min(x_points), max(x_points), 500)  # Smooth curve with 500 points
y_fit = polynomial(x_fit)  # Evaluate the polynomial at x_fit

# Plotting the regressed polynomial
plt.figure(figsize=(20, 12))
plt.plot(x_fit, y_fit, c='blue', label='Cubic Regression')
# plt.scatter(x_points, y_points, s=10, c='red', label='Data Points')  # Optional: Show scatter points
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Regression of Points Where Velocity First Drops Below Zero')
plt.xlim(foil_extent[0], foil_extent[1])
plt.ylim(foil_extent[2], foil_extent[3])
plt.gca().set_aspect('equal')  # Maintain aspect ratio
plt.legend()
plt.grid(True)


# # 5) plot heat map
# tas.heat_maps(case, u_data_lsb, v_data_lsb, foil_extent)
tas.send_plot('metrics')


# # Show the plot
# # Extract max_x and max_y from dewarped_foil
# max_x, max_y = foil_extent[1], foil_extent[3]
#
# # Find indices where |dewarped_u| < 0.05
# indices = np.where((u_data) > 0)
# u_data[indices] = np.nan
#
#
# # Convert matrix indices to real-world coordinates
# x_coords = np.linspace(0, max_x, u_data.shape[1])  # X axis range
# y_coords = np.linspace(0, max_y, u_data.shape[0])  # Y axis range
#
# x_vals = x_coords[indices[1]]  # Convert column indices to real X values
# y_vals = y_coords[indices[0]]  # Convert row indices to real Y values
#
