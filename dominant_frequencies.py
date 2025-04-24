
import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas
from numpy.ma.core import indices

# case = tas.prompt_case() ; read the file
case = 2
u_data, v_data, foil_extent = tas.read_npz(case,'data_files/dewarped_data.npz')
#raw velocity fields
u_raw = u_data
v_raw = v_data

#mean velocity fields
u_data, v_data = tas.frame_process(u_data, v_data,-1)
u_mean = u_data
v_mean = v_data

#fluctuations required for FFT
u_prime=u_raw-u_mean
v_prime=v_raw-v_mean

N=u_prime.shape[1]

dx = (foil_extent[1] - foil_extent[0])/N
dk = 2 * np.pi / (foil_extent[1] - foil_extent[0])
k = np.linspace(dk, (N // 2 - 1) * dk, 305)
print('dk', foil_extent[1] - foil_extent[0])


def get_psd(u_prime, v_prime):
    '''
    Obtains Power Spectral Density matrix.

    Function outputs a 2d matrix of PSD values at each x,y coordinate.

    1) For each snapshot:
        1) FFT along the last axis for both u_prime and v_prime
        2) PSD = |u_prime * conjugate(v_prime)| * 2 -> outputs the PSD at each snapshot, x and y coordinate
        3) Limit to positive frequencies
    2) Averages over all snapshots
    3) Ignores high PSD values

    Args:
        u_prime is the u_prime velocity data
        v_prime is the v_prime velocity data
    '''
    # Compute FFT along the last axis (assumed to be the time or signal dimension)
    u_fft = np.fft.fft(u_prime, axis=-1)
    v_fft = np.fft.fft(v_prime, axis=-1)

    # Compute PSD matrix
    psd = np.abs(u_fft * np.conjugate(v_fft)) * 2 / N ** 2 / dk

    # Compute averaged PSD matrix
    psd = np.mean(psd, axis=0)

    # Clip high PSD values
    return psd

psd = get_psd(u_prime, v_prime)
np.nan_to_num(psd, nan=0.0)

# Pick a frequency index (e.g., 30)
freq_index = 1

print(psd.shape)
print("First few values of PSD:", psd[:5, :5])
"""
# Create a heatmap for the PSD at that frequency
plt.figure(figsize=(10, 6))
plt.pcolormesh(psd, shading='auto', cmap='viridis')  # Adjust cmap to your preference
plt.colorbar(label='PSD')
plt.title('2D Spatial Distribution of PSD')
plt.xlabel('x-index')
plt.ylabel('y-index')
plt.tight_layout()
tas.send_plot('integral_plots')
"""
