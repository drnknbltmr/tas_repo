"""in this file functions are created to find the mean vorticity of a vector field,
it is assumed that U and V are known matrices"""
import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas


case = 5



u_data, v_data, foil_extent = tas.read_npz(case,'data_files/dewarped_data.npz')

dx = 0.0004
dy = 0.0004


#choose which timeframe you want
u_avg, v_avg = tas.frame_process(u_data, v_data,-1)

# tas.heat_maps(2, U, vorticity_field(U, V, dx, dy), foil_extent)
tas.heat_maps(case, u_avg, tas.mean_vorticity(u_data, v_data,dx,dy), foil_extent)
tas.send_plot('meanvorticity')

print(tas.mean_vorticity(u_data, v_data, dx, dy))



