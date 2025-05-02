import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas
import time

def main(u_data, v_data, foil_extent):
    u_data, v_data = tas.frame_process(u_data, v_data, -1)
    time1 = time.time()

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

    time2 = time.time()
    print('Time difference: ', time2 - time1)

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

    # Plotting the regressed polynomial
    plt.figure(figsize=(18, 5))
    # Create the heatmap of u_data
    plt.imshow(
        u_data,
        extent=foil_extent,  # Set the extent to match the foil dimensions
        origin='lower',  # Ensure the origin is at the bottom-left
        cmap='jet',  # Use a colormap (you can change this to any other colormap)
        aspect='equal'  # Automatically adjust the aspect ratio
    )

    plt.plot(x_fit, y_fit, c='black', label='Cubic Regression')
    plt.scatter(x_points, y_points, s=10, c='red', label='Data Points')  # Optional: Show scatter points
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Streamfunction = 0 case: {casenumber}')
    plt.xlim(foil_extent[0], foil_extent[1])
    plt.ylim(foil_extent[2], foil_extent[3])
    plt.gca().set_aspect('equal')  # Maintain aspect ratio
    plt.legend()
    plt.grid(True)

    # # 5) plot heat map
    # tas.heat_maps(case, u_data_lsb, v_data_lsb, foil_extent)
    tas.send_plot('metrics')

    allx_points.append(x_points)
    ally_points.append(y_points)
    allpolynomials.append(polynomial)

    return foil_extent

def plotall(x_points, y_points, foil_extent, polynomial):
    x_fit = np.linspace(min(x_points), max(x_points), 500)  # Smooth curve with 500 points
    y_fit = polynomial(x_fit)  # Evaluate the polynomial at x_fit

    # Plotting the regressed polynomial
    plt.figure(figsize=(18, 5))


    plt.plot(x_fit, y_fit, c='black', label='Cubic Regression')
    plt.scatter(x_points, y_points, s=10, c='red', label='Data Points')  # Optional: Show scatter points
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('All Streamfunctions')
    plt.xlim(foil_extent[0], foil_extent[1])
    plt.ylim(foil_extent[2], foil_extent[3])
    plt.gca().set_aspect('equal')  # Maintain aspect ratio
    plt.legend()
    plt.grid(True)

    # # 5) plot heat map
    # tas.heat_maps(case, u_data_lsb, v_data_lsb, foil_extent)
    tas.send_plot('metrics')

#-----------------------------------------------------------------------------------------------------


singlecase = 3
allcases = True

allx_points = []
ally_points = []
allpolynomials = []

# Read data for all cases
if allcases:
    for casenumber in range(1,6):
        # Read data
        print(casenumber)
        # Load the dewarped data
        if casenumber == 1:
            u_data, v_data, foil_extent = tas.read_npz(casenumber, 'data_files/dewarped_data.npz')
            foilmain(u_data, v_data, foil_extent)
            plotall(allx_points[casenumber-1], ally_points[casenumber-1], foil_extent, allpolynomials[casenumber-1])
        else:
            u_data, v_data, foil_extent = tas.read_npz_loop(casenumber, 'data_files/dewarped_data.npz')
            main(u_data, v_data, foil_extent)
            plotall(allx_points[casenumber - 1], ally_points[casenumber - 1], foil_extent, allpolynomials[casenumber - 1])

else:
    casenumber = singlecase
    u_data, v_data, foil_extent = tas.read_npz(casenumber, 'data_files/dewarped_data.npz')
    main(u_data, v_data, foil_extent)


