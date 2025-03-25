import matplotlib.pyplot as plt
import requests
import io
import time
import numpy as np
import os

def send_plot(chat_name):
    if chat_name == 'plots_n_shit':
        webhook_url = 'https://discord.com/api/webhooks/1348573409474252890/V54cxRpT6qjEAN4WqXoM5u1QCyR9lPhB_FS5QzETSGmFPFeYIOCAAke96PN_Tp9izscu'
    elif chat_name == 'integral_plots':
        webhook_url = 'https://discord.com/api/webhooks/1351119490770927677/J4hJ568PKrKxbuP_Wn15U7tGntG5qYpS05wBUId6KNc3oPZxnksKN8ca1LkF0CZflw2I'
    elif chat_name == 'metrics':
        webhook_url = 'https://discord.com/api/webhooks/1351122362598428757/LVzzLhUmbeObyoFPb35stI0vn-k_6Oi0ktWFjpkgyyZ8AcRbuVE4gxKpnYA9PKawCm9n'
    elif chat_name == 'meanvorticity':
        webhook_url = 'https://discord.com/api/webhooks/1353648026853314642/7KUL4D_DsNFSnVwfV4OK25WhHNY0d9j5_a8qKX8rkGb0sUusUktuRhFIq1WenQFQKdDi'
    image_bytes = io.BytesIO()
    plt.savefig(image_bytes, format='png')
    image_bytes.seek(0)
    data = {"content": "Here's your fking plot" }
    files = {'file': ('plot.png', image_bytes, 'image/png')}
    response = requests.post(webhook_url, data=data, files=files)
    plt.close()

def prompt_case():
    while True:
        case = input("1: 0 Hz\n" "2: 4 Hz\n" "3: 8 Hz\n" "4: 16 Hz\n" "5: 300 Hz\n" "Your choice: ")
        try:
            case = int(case)
            if 1 <= case <= 5:
                break
            print("Please enter a number between 1 and 5")
        except ValueError:
            print("Invalid input! Please enter a number")
    return case

def frame_process(u_data, v_data,frame):
    if 0 <= frame <= 3000:
        u_data = u_data[frame, :, :]
        v_data = v_data[frame, :, :]
    else:
        u_data = np.nanmean(u_data, axis=0)
        v_data = np.nanmean(v_data, axis=0)
    return u_data, v_data

def read_npz(case, file_directory):
    os.chdir("..")
    data = np.load(file_directory)
    foil_extent = data['dewarped_foil']
    u_data = data[f'case_{case}_dewarped_u']
    v_data = data[f'case_{case}_dewarped_v']
    return u_data, v_data, foil_extent

def log_time(start, label):
    elapsed = time.time() - start
    print(f"{label}: {elapsed:.4f} seconds")
    return time.time()

def heat_maps(case, u_data, v_data, foil_extent):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    cmap = plt.cm.jet.copy()
    cmap.set_bad('white')

    im = ax1.imshow(u_data, extent=foil_extent, aspect='equal', origin='lower', cmap=cmap)
    ax1.set_title(f'Case {case} - Tangential Velocity (u)')
    ax1.set_xlabel('Arc Length (m)')
    ax1.set_ylabel('Distance from Airfoil (m)')
    cbar_u = fig.colorbar(im, ax=ax1)
    cbar_u.set_label('u')

    im_v = ax2.imshow(v_data, extent=foil_extent, aspect='equal', origin='lower', cmap=cmap)
    ax2.set_title(f'Case {case} - Normal Velocity (v)')
    ax2.set_xlabel('Arc Length (m)')
    ax2.set_ylabel('Distance from Airfoil (m)')
    cbar_v = fig.colorbar(im_v, ax=ax2)
    cbar_v.set_label('v')
    plt.tight_layout()


def mean_vorticity(u_data, v_data, dx, dy):
    n_frames, ny, nx = u_data.shape
    vorticity_tensor = np.empty((n_frames, ny, nx))

    def vorticity_field(u_frame, v_frame, dx, dy):

        dv_dx = np.gradient(v_frame, dx, axis=1)
        du_dy = np.gradient(u_frame, dy, axis=0)
        vorticity = dv_dx - du_dy
        return vorticity

    for frame in range(n_frames):
        u = u_data[frame, :, :]
        v = u_data[frame, :, :]
        vorticity_tensor[frame] = vorticity_field(u, v, dx, dy)

    mean_vorticity_in_time = np.nanmean(vorticity_tensor, axis=0)
    return mean_vorticity_in_time