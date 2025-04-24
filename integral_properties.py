#2D: Calculate the mean integral quantities of the boundary layer
#(displacement and momentum thickness, shape factor).
#Comment on your results.
import numpy as np
import matplotlib.pyplot as plt
import tas_functions as tas
from numpy.ma.core import indices

# case = tas.prompt_case()
case = 2
u_data, v_data, foil_extent = tas.read_npz(case,'data_files/dewarped_data.npz')
u_data, v_data = tas.frame_process(u_data, v_data,-1)
print(foil_extent)
u = u_data

#print("max x =", foil_extent[1])
y_max = foil_extent[3]

print("u_data_shape =", u_data.shape)
print("u_1_1 =", u_data[1][1])
#print("col_1", u[:,1]) #100 values

pixel_size=(foil_extent[1])/400
x_list = np.linspace(0, foil_extent[1], 400)


#read data

def boundary_layer_indeces(x, U_e, u, y_max):
    u = np.nan_to_num(u)
    indeces = []
    bl_thickness = []
    for i in range (u.shape[1]):
        u_col = u[:, i]
        U_inf = np.full(100, U_e)

        # Find indices where list1 is greater than or equal to 0.99 * list2
        matches = np.where(u_col >= 0.99 * U_inf)[0]
        indeces.append(matches[0]) if matches.size >0 else None
        #indeces = np.array(indeces)

    for i in range (len(indeces)):
        #bl_thickness = indeces/100*y_max
        bl_thickness.append((indeces[i]/100)*y_max)
    return(indeces, bl_thickness)

print("indeces =", boundary_layer_indeces(x_list, 5.47, u_data, y_max)[0])
print("len_indeces=", len(boundary_layer_indeces(x_list, 5.47, u_data, y_max)[0]))
print("bl_thickness =", boundary_layer_indeces(x_list, 5.47, u_data, y_max)[1])
print("len_bl_thickness =", len(boundary_layer_indeces(x_list, 5.47, u_data, y_max)[1]))

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
#U_e is the free stream velocity in x direction
def displacement_thickness(u, U_e):
    u=np.nan_to_num(u)
    delta_1=[]
    #U_inf = U_e
    for i in range (u.shape[1]):
        U_inf = np.max(u[:, i]) #takes maximum value from the column i in matrix u
        f=1-u[:,i]/U_inf
        delta_1.append(np.sum(f) * pixel_size) #n_x = 400 pixels #h_image = image height
    return delta_1

#delta_1 = displacement_thickness(u_data)
#print("displacement thickness =", delta_1) #works, gives 400 values, order of magnitude 0.00x - 0.0x
#print(len(delta_1)) #400 values

def momentum_thickness(u, U_e):
    u = np.nan_to_num(u)
    #U_inf = U_e
    delta_2=[]
    for i in range (u.shape[1]):
        U_inf=np.max(u[:,i])
        g=(u[:,i]/U_inf)*(1-u[:,i]/U_inf)
        delta_2.append(np.sum(g)*pixel_size)  #n_x=400 pixels #h_image = image height
    return delta_2

#delta_2 = momentum_thickness(u_data)
#print("momentum_thickness =", delta_2) #prints 400 values, order of magnitude 0.00x
#print("length momentum thickness =", len(momentum_thickness))

def shape_factor(u, U_e):
    #H = displacement_thickness(u)/momentum_thickness(u) #this doesn't work, so we try with the line below
    H = [displacement_thickness(u, U_e)[i] / momentum_thickness(u, U_e)[i] for i in range(len(displacement_thickness(u, U_e)))]
    return H

def plot_displacement_thickness(x,delta_1):
    x_values = x  # Extract x-coordinates from the first row (or any row)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, delta_1, marker='o', linestyle='-', color='b', label=r'$\delta_1$')
    plt.xlabel("x-Location [m]")
    plt.ylabel("Displacement Thickness [m]")
    plt.title("Displacement Thickness vs. x-Location")
    plt.grid(True)
    plt.legend()
    tas.send_plot('integral_plots')

def plot_momentum_thickness(x,delta_2):
    x_values = x  # Extract x-coordinates from the first row (or any row)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, delta_2, marker='o', linestyle='-', color='red', label=r'$\delta_2$')
    plt.xlabel("x-Location [m]")
    plt.ylabel("Momentum Thickness [m]")
    plt.title("Momentum Thickness vs. x-Location")
    plt.grid(True)
    plt.legend()
    tas.send_plot('integral_plots')

def plot_shape_factor(x,H):
    x_values = x  # Extract x-coordinates from the first row (or any row)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, H, marker='o', linestyle='-', color='green', label=r'$\H$')
    plt.xlabel("x-Location [m]")
    plt.ylabel("Shape Factor")
    plt.title("Shape Factor vs. x-Location [m]")
    plt.grid(True)
    plt.legend()
    tas.send_plot('integral_plots')

def plot_bl_thickness(x,boundary_layer_indeces):
    x_values = x  # Extract x-coordinates from the first row (or any row)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, boundary_layer_indeces[1], marker='o', linestyle='-', color='b', label=r'$\H$')
    plt.xlabel("x-Location [m]")
    plt.ylabel("Boundary Layer Thickness [m]")
    plt.title("Boundary Layer Thickness vs. x-Location")
    plt.grid(True)
    plt.legend()
    tas.send_plot('integral_plots')




"""
print(displacement_thickness(U).shape)
print(displacement_thickness(U))
print(x)
"""

plot_displacement_thickness(x_list,displacement_thickness(u_data, 5.47))
plot_momentum_thickness(x_list,momentum_thickness(u_data, 5.47))
plot_shape_factor(x_list,shape_factor(u_data, 5.47))
plot_bl_thickness(x_list,boundary_layer_indeces(x_list, 5.47, u_data, y_max))
