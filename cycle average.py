"""In this file the cycle averages of the velocity field will be calculated,
to do this we will average the different cases out based on frequencies 1, 3, 30 HZ"""
import numpy as np
import math
import matplotlib.widgets as mpl
import matplotlib.pyplot as plt
import tas_functions as tas
#this function takes in the input velocity tensors, the frequency,
#the output will be a list of tensors per cycle

tperframe = 2.52025/3000
framepersec = 3000/2.52025
f1, f2, f3 = 1, 3, 30
ncyc1,ncyc2,ncyc3 = 2, 7, 75

def velocity_cycle_average(frequency, U, V, framepersec):
    frames_per_cycle = int(framepersec / frequency)  # Frames per cycle
    ncyc = len(U) // frames_per_cycle  # Number of complete cycles

    #shorten U and V such that there are only complete cycles
    U_short = U[:ncyc * frames_per_cycle]
    V_short = V[:ncyc * frames_per_cycle]

    # Reshape U and V into (ncyc, frames_per_cycle, height, width) efficiently
    tensor_list_u = U_short.reshape(ncyc, frames_per_cycle, U.shape[1], U.shape[2])
    tensor_list_v = V_short.reshape(ncyc, frames_per_cycle, V.shape[1], V.shape[2])

    velocity_cycle_average_u = np.mean(tensor_list_u, axis = 0)
    velocity_cycle_average_v = np.mean(tensor_list_v, axis = 0)

    return velocity_cycle_average_u, velocity_cycle_average_v



"""the cycle average of the vorticity will be calculated underneath"""

def cycle_average_vorticity(vorticity_field, framepersec, U, V, dx, dy):

    #import the vorticity field from the function "vorticity_field"
    vorticity = tas.vorticity_field(U, V, dx, dy)

    frames_per_cycle = int(framepersec / frequency)  # Frames per cycle
    ncyc = len(vorticity) // frames_per_cycle  # Number of complete cycles

    # shorten vorticity such that there are only complete cycles
    vorticity_short = vorticity[:ncyc * frames_per_cycle]

    # Reshape vorticity into (ncyc, frames_per_cycle, height, width) efficiently
    tensor_list_vorticity = vorticity_short.reshape(ncyc, frames_per_cycle, vorticity.shape[1], vorticity.shape[2])

    #average it out over the cycles
    vorticity_cycle_average = np.nanmean(tensor_list_vorticity, axis=0)
    return vorticity_cycle_average



######################################################################################
# # testing the functions with dummy data
# # Example parameters
# framepersec = 100  # Frames per second
# frequency = 5      # Hz (cycles per second)
# total_frames = 3000  # Total frames
# height, width = 64, 64  # Example spatial dimensions
# dx = 0.001
# dy = 0.001
#
# # Generate random velocity field data (U, V) with shape (total_frames, height, width)
# U = np.random.rand(total_frames, height, width)
# V = np.random.rand(total_frames, height, width)
# print(U.shape)
# velocity_cycle_average_u, velocity_cycle_average_v = velocity_cycle_average(frequency, U, V, framepersec)
# print(velocity_cycle_average_u.shape)
# print(velocity_cycle_average_v.shape)
#
# vorticity_averaged = cycle_average_vorticity(vorticity_field, framepersec, U, V, dx, dy)
# print("vorticity averaged shape: ", vorticity_averaged.shape)


###########################################################################
#testing with real data

case = 3
u_data, v_data, foil_extent = tas.read_npz(case,'data_files/dewarped_data.npz')
tperframe = 2.52025/3000
framepersec = 3000/2.52025
f1, f2, f3 = 1, 3, 30
ncyc1,ncyc2,ncyc3 = 2, 7, 75
velocity_cycle_average_u, velocity_cycle_average_v = velocity_cycle_average(f3, u_data, v_data, framepersec)
print(velocity_cycle_average_u.shape)
print(velocity_cycle_average_v.shape)

print(np.shape(velocity_cycle_average_u))
frames = np.shape(velocity_cycle_average_u)[0]
initial_frame = 0
ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # Position: [left, bottom, width, height]
slider = mpl.Slider(ax_slider, 'Frame', 0, frames - 1, valinit=initial_frame, valstep=1)

# Update function
def update(val):
    frame_idx = int(slider.val)  # Get slider value
    image.set_array(data[frame_idx])  # Update image
    ax.set_title(f"Frame {frame_idx}")
    fig.canvas.draw_idle()  # Redraw

# Connect slider to update function
slider.on_changed(update)

#plotting
tas.heat_maps(case, velocity_cycle_average_u, velocity_cycle_average_v, foil_extent)#
tas.send_plot('meanvorticity')
# plt.show()






