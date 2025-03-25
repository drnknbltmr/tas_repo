import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas

# Read data
case = 5

# Load the dewarped data
u_data, v_data, foil_extent = tas.read_npz(case, 'data_files/dewarped_data.npz')

# Process the frames to get an average velocity field
u_data, v_data = tas.frame_process(u_data, v_data, -1)

# Get the shape of the velocity data array
ny, nx = u_data.shape

# Generate x and y coordinates based on the foil extent
x_coords = np.linspace(foil_extent[0], foil_extent[1], nx)
y_coords = np.linspace(foil_extent[2], foil_extent[3], ny)

# Lists to store the coordinates of points where velocity first goes below zero
x_points = []
y_points = []

# Iterate over each column (x direction)
for j in range(nx):
    # Iterate from top to bottom (starting at highest y)
    for i in range(ny-1, -1, -1):
        if u_data[i, j] < 0:
            x_points.append(x_coords[j])
            y_points.append(y_coords[i])
            break  # Stop after finding the first occurrence
# Convert lists to numpy arrays for easier manipulation
x_points = np.array(x_points)
y_points = np.array(y_points)

if case == 5:
    x_points = x_points[2:]
    y_points = y_points[2:]

# Perform cubic regression
coefficients = np.polyfit(x_points, y_points, 3)  # Fit a cubic polynomial
polynomial = np.poly1d(coefficients)  # Create a polynomial function

# Generate x values for plotting the polynomial, limited to the range of x_points
x_fit = np.linspace(min(x_points), max(x_points), 500)  # Smooth curve with 500 points
y_fit = polynomial(x_fit)  # Evaluate the polynomial at x_fit

# Plotting the heatmap and overlaying the regression line
plt.figure(figsize=(20, 12))

# Create the heatmap of u_data
plt.imshow(
    u_data,
    extent=foil_extent,  # Set the extent to match the foil dimensions
    origin='lower',      # Ensure the origin is at the bottom-left
    cmap='jet',      # Use a colormap (you can change this to any other colormap)
    aspect='equal'        # Automatically adjust the aspect ratio
)

# Overlay the cubic regression line and scatter points
plt.plot(x_fit, y_fit, c='yellow', linewidth=2, label='Cubic Regression')  # Black regression line
plt.scatter(x_points, y_points, s=4, c='black', label='Data Points')       # Black scatter points

# Add labels, title, and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Regression Overlaid on Heatmap of u_data')
plt.legend()
plt.grid(False)  # Turn off grid to avoid clutter


# Show the plot (optional)
plt.show()