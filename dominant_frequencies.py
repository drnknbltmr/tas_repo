import numpy as np

# case = tas.prompt_case() ; read the file
case = 2
u_data, v_data, foil_extent = tas.read_npz(case,'data_files/dewarped_data.npz')
u_data, v_data = tas.frame_process(u_data, v_data,-1)
print(foil_extent)
u = u_data