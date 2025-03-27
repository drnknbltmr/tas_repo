#2D: Calculate the mean integral quantities of the boundary layer
#(displacement and momentum thickness, shape factor).
#Comment on your results.
import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas

case = tas.prompt_case()
u_data, v_data, foil_extent = tas.read_npz(case,'data_files/dewarped_data.npz')
u_data, v_data = tas.frame_process(u_data, v_data,-1)
print(foil_extent)
u = u_data

#print("max x =", foil_extent[1])

print("u_data_shape =", u_data.shape)
print("u_1_1 =", u_data[1][1])
#print("col_1", u[:,1]) #100 values

pixel_size=(foil_extent[1])/400
x_list = np.linspace(0, foil_extent[1], 400)


#read data

#this function takes the location vector x as input and calculates the thickness of the boundary
#layer at each x location in the vector
def boundary_layer_thickness(x,U_inf=5.47):
    nu=1.477e-5
    delta=5*np.sqrt(nu*x/U_inf)
    return delta
#this functions take a matrix u, which indexes the parallel velocities for a given frame
#(so the u velocity at each pixel) and calculates the specified
#parameters over a frame
#the outputs are vectors that represent these values along the chord
def displacement_thickness(u):
    u=np.nan_to_num(u)
    delta_1=[]
    for i in range (u.shape[1]):
        U_inf = np.max(u[:, i]) #takes maximum value from the column i in matrix u
        f=1-u[:,i]/U_inf
        delta_1.append(np.sum(f) * pixel_size) #n_x = 400 pixels #h_image = image height
    return delta_1

#delta_1 = displacement_thickness(u_data)
#print("displacement thickness =", delta_1) #works, gives 400 values, order of magnitude 0.00x - 0.0x
#print(len(delta_1)) #400 values

def momentum_thickness(u):
    u = np.nan_to_num(u)
    delta_2=[]
    for i in range (u.shape[1]):
        U_inf=np.max(u[:,i])
        g=(u[:,i]/U_inf)*(1-u[:,i]/U_inf)
        delta_2.append(np.sum(g)*pixel_size)  #n_x=400 pixels #h_image = image height
    return delta_2

#delta_2 = momentum_thickness(u_data)
#print("momentum_thickness =", delta_2) #prints 400 values, order of magnitude 0.00x
#print("length momentum thickness =", len(momentum_thickness))

def shape_factor(u):
    H = displacement_thickness(u)/momentum_thickness(u)
    return H

def plot_displacement_thickness(x,delta_1):
    x_values = x[0, :]  # Extract x-coordinates from the first row (or any row)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, delta_1, marker='o', linestyle='-', color='b', label=r'$\delta_1$')
    plt.xlabel("x-location")
    plt.ylabel("Displacement Thickness")
    plt.title("Displacement Thickness vs. x-location")
    plt.grid(True)
    plt.legend()
    tas.send_plot('integral_plots')

def plot_momentum_thickness(x,delta_2):
    x_values = x[0, :]  # Extract x-coordinates from the first row (or any row)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, delta_2, marker='o', linestyle='-', color='red', label=r'$\delta_2$')
    plt.xlabel("x-location")
    plt.ylabel("Momentum Thickness")
    plt.title("Momentum Thickness vs. x-location")
    plt.grid(True)
    plt.legend()
    tas.send_plot('integral_plots')

def plot_shape_factor(x,u):
    x_values = x[0, :]  # Extract x-coordinates from the first row (or any row)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, shape_factor(u), marker='o', linestyle='-', color='green', label=r'$\H$')
    plt.xlabel("x-location")
    plt.ylabel("Shape factor")
    plt.title("Shape factor vs. x-location")
    plt.grid(True)
    plt.legend()
    tas.send_plot('integral_plots')





"""
print(displacement_thickness(U).shape)
print(displacement_thickness(U))
print(x)
"""

plot_displacement_thickness(x_list,displacement_thickness(u_data))
plot_momentum_thickness(x_list,momentum_thickness(u_data))
plot_shape_factor(x_list,u_data)
