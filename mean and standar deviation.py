import numpy as np


def mean_velocity_field(U_array,V_array):
    mean_velocity_field_u = np.nanmean(U_array, axis=0)
    mean_velocity_field_v = np.nanmean(V_array, axis=0)
    return mean_velocity_field_u, mean_velocity_field_v

print(mean_velocity_field(testU,testV))

def standard_deviation(U_array,V_array):
    std_u = np.nanstd(U_array, axis=0, ddof=1)
    std_v = np.nanstdstd(V_array, axis=0, ddof=1)
    return std_u, std_v

print("std_U = ", standard_deviation(testU,testV)[0])





###################################################################################################
# #testing the code
# # making a test array to be able to test the mean function
# n_frames = 10  # Number of time frames
# ny = 5         # Grid size in y-direction
# nx = 5         # Grid size in x-direction
#
# # Generate random velocity components
# testU = np.random.rand(n_frames, ny, nx)  # Random x-velocity field
# testV = np.random.rand(n_frames, ny, nx)


