import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas

from matplotlib.colors import Normalize


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

    # Perform cubic regression
    if casenumber == 1:
        degree = 8
    elif casenumber == 2:
        degree = 4
    elif casenumber == 3:
        degree = 6
    elif casenumber == 4:
        degree = 6
    else:
        degree = 2

    coefficients = np.polyfit(x_points, y_points, degree)  # Fit a polynomial
    polynomial = np.poly1d(coefficients)  # Create a polynomial function

    # Find all real roots (where y=0)
    roots = np.roots(coefficients)
    real_roots = roots[np.isreal(roots)].real  # Keep only real roots

    # Find the root before min(x_points) and after max(x_points)
    root_before = real_roots[real_roots < min(x_points)]
    root_after = real_roots[real_roots > max(x_points)]
    if len(root_after) > 0:
        first_root_after = root_after[0]
    # Set plot range to include both roots (with some padding)
    x_min_plot = root_before[0] if len(root_before) > 0 and root_before > 0 else min(x_points) - 0.1 * (max(x_points) - min(x_points))
    x_max_plot = first_root_after if len(root_after) > 0 and first_root_after < 0.1 else max(x_points) + 0.1 * (max(x_points) - min(x_points))

    # Generate x values extending to both roots
    if extrapolate:
        x_fit = np.linspace(x_min_plot, x_max_plot, 500)
    else:
        x_fit = np.linspace(min(x_points), max(x_points), 500)
    y_fit = polynomial(x_fit)

    #bubble length according to data points
    dataLSBlength = max(x_points)-min(x_points)

    min_x = min(x_points)
    max_x = max(x_points)
    max_y = max(y_points)

    for i in range(len(x_points)):
        if y_points[i] == max_y:
            maxloc_x = x_points[i]

    #calculate area
    P_int = polynomial.integ()
    a, b = x_fit[0], x_fit[-1]
    area = P_int(b) - P_int(a)

    #finding bubble length using extrapolated polynomial
    polyLSBlength = first_root_after - root_before[0]

    return u_data, x_fit, y_fit, foil_extent, x_points, y_points, dataLSBlength, polyLSBlength, degree, min_x, max_x, max_y, area, maxloc_x


def combinedfigure():
    plt.figure(figsize=(16, 9))
    # Create the heatmap of u_data
    # plt.plot(x_fit, y_fit, c='blue', label='Cubic Regression')
    for case in range(1,6):
        u_data, x_fit, y_fit, foil_extent, x_points, y_points, dataLSBlength, polyLSBlength, degree = main(case)
        #plt.scatter(x_points, y_points, s=10, c='red', label='Data Points')  # Optional: Show scatter points
        plt.plot(x_fit, y_fit, c='black', linewidth=2, label='Quartic Regression Data Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Title')
    plt.xlim(foil_extent[0], foil_extent[1])
    plt.ylim(foil_extent[2], foil_extent[3])
    plt.gca().set_aspect('equal')  # Maintain aspect ratio
    plt.legend()
    plt.grid(True)
    plt.show()


def marloesplot():
    # Plotting the heatmap and overlaying the regression line
    norm = Normalize(vmin=-3, vmax=8)
    casenumbers = [1,2,3,4,5]
    cmap = 'jet'
    frequencies = [0, 4, 8, 16, 300]

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 14),sharey=True, constrained_layout=True)

    # Create the heatmap of u_data
    for ax, casenumber, frequency in zip(axes, casenumbers, frequencies):
        u_data, x_fit, y_fit, foil_extent, x_points, y_points, dataLSBlength, polyLSBlength, degree, min_x, max_x, max_y, area, maxloc_x = main(casenumber)
        x_fit = x_fit * 10**3
        y_fit = y_fit * 10**3
        foil_extent = foil_extent * 10**3
        x_points = x_points * 10**3
        y_points = y_points * 10**3
        im = ax.imshow(
            u_data,
            extent=foil_extent,  # Set the extent to match the foil dimensions
            origin='lower',      # Ensure the origin is at the bottom-left
            cmap='jet',      # Use a colormap (you can change this to any other colormap)
            aspect='equal',
            norm=norm # Automatically adjust the aspect ratio
        )

        ax.plot(x_fit, y_fit, c='black', linewidth=2, label=f'Polynomial Regression (Degree {degree})')
        ax.scatter(x_points, y_points, s=10, c='red', label='Stream Function equals Zero')
        ax.set_title(f'Mean LSB Boundary for Case {casenumber} (Frequency = {frequency} Hz)')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        ax.grid(False)
        ax.legend(fontsize=9, loc='upper right')

        #printing all values
        print(f'for case {casenumber}: length = {dataLSBlength}, Starting point = {min_x}, Ending point = {max_x}, max height = {max_y}, area = {area}, maxloc_x = {maxloc_x}')

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
    #plt.show()

extrapolate = False

marloesplot()

#combinedfigure()


