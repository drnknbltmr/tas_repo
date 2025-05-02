import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas
from matplotlib.colors import Normalize

def plotoverview(storedcoefficients):
    # Create a range of x values (adjust as needed)
    x = np.linspace(0, 0.1, 400)  # 400 points from -10 to 10

    # Plot each polynomial
    plt.figure(figsize=(10, 6))
    for i, coef in enumerate(storedcoefficients, 1):
        y = np.polyval(coef, x)  # Evaluate the polynomial
        plt.plot(x, y, label=f'case {i}')

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of 3rd-Degree Polynomials')
    plt.legend()
    plt.grid(True)
    tas.send_plot('metrics')

storedcoefficients = []
min_u_data_arr =[]
max_u_data_arr =[]

def main(casenumber):
    print(casenumber)
    # Load the dewarped data
    if casenumber == 1:
        u_data, v_data, foil_extent = tas.read_npz(casenumber, 'data_files/dewarped_data.npz')

    else:
        u_data, v_data, foil_extent = tas.read_npz_loop(casenumber, 'data_files/dewarped_data.npz')
    u_data, v_data = tas.frame_process(u_data, v_data, -1)

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

    # (a) Integrate vertically first column
    for j in range(Ny - 1):
        psi[j, 0] = psi[j, 0] + u_data[j + 1, 0] * dy

    # integrate horizontally first row
    for i in range(Nx - 1):
        psi[0, i] = psi[0, i] - v_data[0, i + 1] * dx

    # (b) Integrate horizontally across rows
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            psi[j, i] = psi[j, i] - v_data[j, i + 1] * dx + u_data[j + 1, i] * dy
            psi[j, i] = np.abs(psi[j, i])

            # # Detect separation points (psi <= 0)
            # if psi[j, i] < 0.000005:
            #     x_points.append(x_coords[i])
            #     y_points.append(y_coords[j])
    if casenumber == 5:
        for i in range(150, Nx - 50):  # Iterate over each column (x-coordinate)
            # Extract the column, considering only processed rows (0 to Ny-2)
            column = psi[0:Ny - 1, i]

            # Skip if all values are zero (unprocessed or invalid)
            if np.all(column == 0):
                continue

            # Find the minimum non-zero psi in this column
            non_zero_mask = (column > 0)  # Exclude zeros
            if np.any(non_zero_mask):
                j_min = np.argmin(column[non_zero_mask])  # Find row with min psi (excluding zeros)
                y_idx = np.where(non_zero_mask)[0][j_min]  # Get the actual row index
                if not np.isnan(u_data[y_idx - 4, i]):
                    # j_min = np.argmin(column[non_zero_mask])  # Find row with min psi (excluding zeros)
                    # y_idx = np.where(non_zero_mask)[0][j_min]  # Get the actual row index
                    x_points.append(x_coords[i])
                    y_points.append(y_coords[y_idx])
    else:
        for i in range(50, Nx - 50):  # Iterate over each column (x-coordinate)
            # Extract the column, considering only processed rows (0 to Ny-2)
            column = psi[0:Ny - 1, i]

            # Skip if all values are zero (unprocessed or invalid)
            if np.all(column == 0):
                continue

            # Find the minimum non-zero psi in this column
            non_zero_mask = (column > 0)  # Exclude zeros
            if np.any(non_zero_mask):
                j_min = np.argmin(column[non_zero_mask])  # Find row with min psi (excluding zeros)
                y_idx = np.where(non_zero_mask)[0][j_min]  # Get the actual row index
                if not np.isnan(u_data[y_idx - 4, i]):
                    # j_min = np.argmin(column[non_zero_mask])  # Find row with min psi (excluding zeros)
                    # y_idx = np.where(non_zero_mask)[0][j_min]  # Get the actual row index
                    x_points.append(x_coords[i])
                    y_points.append(y_coords[y_idx])

    x_points = np.array(x_points)
    y_points = np.array(y_points)



    # # 4) Identify LSB region by psi < 0 (or whichever condition suits your flow)
    # lsb_mask = (abs(psi) < 0.1)
    # u_data_lsb = np.where(lsb_mask, u_data, np.nan)
    # v_data_lsb = np.where(lsb_mask, v_data, np.nan)
    # # x_data_lsb = np.where(lsb_mask, x_data, np.nan)
    # # y_data_lsb = np.where(lsb_mask, y_data, np.nan)

    # Perform cubic regression
    coefficients = np.polyfit(x_points, y_points, 5)  # Fit a cubic polynomial
    polynomial = np.poly1d(coefficients)  # Create a polynomial function

    # Generate x values for plotting the polynomial, limited to the range of x_points
    x_fit = np.linspace(min(x_points), max(x_points), 500)  # Smooth curve with 500 points
    y_fit = polynomial(x_fit)  # Evaluate the polynomial at x_fit


    # allx_points.append(x_points)
    # ally_points.append(y_points)
    # allpolynomials.append(polynomial)

    return u_data, x_fit, y_fit, foil_extent, x_points, y_points

def getpolyline(casenumber):
    print(casenumber)
    # Load the dewarped data
    if casenumber == 1:
        u_data, v_data, foil_extent = tas.read_npz(casenumber, 'data_files/dewarped_data.npz')

    else:
        u_data, v_data, foil_extent = tas.read_npz_loop(casenumber, 'data_files/dewarped_data.npz')


    # Process the frames to get an average velocity field
    u_data, v_data = tas.frame_process(u_data, v_data, -1)

    # Get the shape of the velocity data array
    ny, nx = u_data.shape

    print(foil_extent)

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
            if u_data[i, j] <= 0:

                x_points.append(x_coords[j])
                y_points.append(y_coords[i])
                break  # Stop after finding the first occurrence
    # Convert lists to numpy arrays for easier manipulation
    x_points = np.array(x_points)
    y_points = np.array(y_points)

    if casenumber    == 5:
        x_points = x_points[3:]
        y_points = y_points[3:]

    # Perform cubic regression
    coefficients = np.polyfit(x_points, y_points, 3)  # Fit a cubic polynomial
    polynomial = np.poly1d(coefficients)  # Create a polynomial function

    #stores the coefficients
    storedcoefficients.append(coefficients)

    print('Coefficients for poly', coefficients)

    # Generate x values for plotting the polynomial, limited to the range of x_points
    x_fit = np.linspace(min(x_points), max(x_points), 500)  # Smooth curve with 500 points
    y_fit = polynomial(x_fit)  # Evaluate the polynomial at x_fit


    # For a cubic polynomial, we need to find critical points by taking derivative
    derivative = np.polyder(polynomial)  # Get the derivative of the polynomial
    critical_points = np.roots(derivative)  # Find where derivative = 0

    # # Filter critical points that are within our x-range and real numbers
    # valid_critical_points = []
    # for root in critical_points:
    #     if np.isreal(root) and min(x_points) <= root <= max(x_points):
    #         valid_critical_points.append(np.real(root))

    # Evaluate the polynomial at critical points to find maximum
    # if valid_critical_points:
    #     y_values = polynomial(valid_critical_points)
    #     max_index = np.argmax(y_values)
    #     highest_x = valid_critical_points[max_index]
    #     highest_y = y_values[max_index]
    #     print(f"Highest point (vertex): x = {highest_x:.4f}, y = {highest_y:.4f}")
    # else:
    #     print("No valid critical points found within the x-range")

    # Calculate the area under the curve between min and max x-points
    # Using numerical integration (trapezoidal rule)
    # x_area = np.linspace(min(x_points), max(x_points), 1000)
    # y_area = polynomial(x_area)
    # area = np.trapz(y_area, x_area)
    # print(f"Area under the curve: {area:.10f}")

    # Find intersection points with x-axis (roots of the polynomial)
    roots = np.roots(polynomial)
    real_roots = [np.real(root) for root in roots if np.isreal(root) and min(x_points) <= np.real(root) <= max(x_points)]
    print("Intersection points with x-axis:")
    for i, root in enumerate(real_roots):
        print(f"  Root {i+1}: x = {root:.4f}")

    return u_data, x_fit, y_fit, foil_extent

def combinedfigure():


# Plotting the heatmap and overlaying the regression line
norm = Normalize(vmin=-3, vmax=8)
casenumbers = [1,2,3,4,5]
cmap = 'jet'

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 14),sharey=True,
constrained_layout=True)
titles = [f'LSB Boundary: Case {i}' for i in range(1, 6)]

# Create the heatmap of u_data
for ax, title, casenumber in zip(axes, titles, casenumbers):
    u_data, x_fit, y_fit, foil_extent, x_points, y_points = main(casenumber)
    im = ax.imshow(
        u_data,
        extent=foil_extent,  # Set the extent to match the foil dimensions
        origin='lower',      # Ensure the origin is at the bottom-left
        cmap='jet',      # Use a colormap (you can change this to any other colormap)
        aspect='equal',
        norm=norm # Automatically adjust the aspect ratio
    )

    ax.plot(x_fit, y_fit, c='black', linewidth=2,
            label='Quintic Regression Data Points')
    ax.scatter(x_points, y_points, s=10, c='red', label='Data Points')

    ax.set_title(title)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    ax.grid(False)
    ax.legend(fontsize=9, loc='upper right')

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)  # empty mappable with the
sm.set_array([])

cbar = fig.colorbar(sm,
    ax=axes,
    orientation='vertical',
    fraction=0.046,  # how wide the bar is relative to the Axes
    pad=0.04, # gap between bar and Axes
    shrink=0.6
)

cbar.set_label('Velocity (m/s)')  # change text to whatever quantity you’re plotting


# Show the plot (optional)
tas.send_plot('metrics')
# plt.show()


#plotoverview(storedcoefficients)

