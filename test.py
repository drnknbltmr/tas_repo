import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas
import time

start_time = time.time()

case = tas.prompt_case()

u_data, v_data, foil_extent = tas.read_npz(case,'data_files/dewarped_data.npz')

start_time = tas.log_time(start_time, "Reading Data")

u_data, v_data = tas.frame_process(u_data, v_data, 0)

tas.heat_maps_png(case,u_data,v_data,foil_extent, 1000)
plt.show()


