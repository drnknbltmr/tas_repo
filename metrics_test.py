import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas

for casenumber in range(1,6):
    # Read data
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

    print('Coefficients for poly', coefficients)

    # Generate x values for plotting the polynomial, limited to the range of x_points
    x_fit = np.linspace(min(x_points), max(x_points), 500)  # Smooth curve with 500 points
    y_fit = polynomial(x_fit)  # Evaluate the polynomial at x_fit


    # For a cubic polynomial, we need to find critical points by taking derivative
    derivative = np.polyder(polynomial)  # Get the derivative of the polynomial
    critical_points = np.roots(derivative)  # Find where derivative = 0

    # Filter critical points that are within our x-range and real numbers
    valid_critical_points = []
    for root in critical_points:
        if np.isreal(root) and min(x_points) <= root <= max(x_points):
            valid_critical_points.append(np.real(root))

    # Evaluate the polynomial at critical points to find maximum
    if valid_critical_points:
        y_values = polynomial(valid_critical_points)
        max_index = np.argmax(y_values)
        highest_x = valid_critical_points[max_index]
        highest_y = y_values[max_index]
        print(f"Highest point (vertex): x = {highest_x:.4f}, y = {highest_y:.4f}")
    else:
        print("No valid critical points found within the x-range")

    # Calculate the area under the curve between min and max x-points
    # Using numerical integration (trapezoidal rule)
    x_area = np.linspace(min(x_points), max(x_points), 1000)
    y_area = polynomial(x_area)
    area = np.trapz(y_area, x_area)
    print(f"Area under the curve: {area:.8f}")

    # Find intersection points with x-axis (roots of the polynomial)
    roots = np.roots(polynomial)
    real_roots = [np.real(root) for root in roots if np.isreal(root) and min(x_points) <= np.real(root) <= max(x_points)]
    print("Intersection points with x-axis:")
    for i, root in enumerate(real_roots):
        print(f"  Root {i+1}: x = {root:.4f}")


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
    plt.plot(x_fit, y_fit, c='black', linewidth=2, label='Cubic Regression')  # Black regression line
    plt.scatter(x_points, y_points, s=4, c='red', label='Data Points')       # Black scatter points

    # plt.fill_between(x_fit, y_fit, y2=min(y_coords), where=(x_fit >= min(x_points)) & (x_fit <= max(x_points)),
    #                  color='purple', alpha=0, label='Area under curve')

    # Add labels, title, and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Regression Overlaid on Heatmap of u_data')
    plt.legend()
    plt.grid(False)  # Turn off grid to avoid clutter


    # Show the plot (optional)
    # tas.send_plot('metrics')
    # plt.show()
