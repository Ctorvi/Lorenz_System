import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functions.lorenz import lorenz
from functions.lorenz import evolve_lorenz
from functions.lorenz import plot_lorenz_trajectories
from functions.lorenz import z_maximum_event
from functions.bifurcation import maxima_Z
from functions.bifurcation import maxima_Z_fast

from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
 


t_init = 0
t_final = 200


rho_values = [0,50,100,150,200,250,300,350,400]


###### FIRST BRANCH #######

###initial_states = [[1.,1,1]]

### COMPUTING ###

# for rho in rho_values:
 
 
#  results, rho_plot, z_plot = maxima_Z_fast(
#         initial_state=initial_states,
#         rho_values=np.linspace(rho, rho + 50, 400),
#         t_span=(t_init, t_final),
#         t_transient=100,
#         sigma=10.0,
#         beta=8 / 3,
#     )

#  data = np.column_stack((rho_plot, z_plot))

#  np.savetxt(
#     f"lorenz_bifurcation_{rho}_{rho + 50}.csv",
#     data,
#     delimiter=",",
#     header="rho,z_max",
#     comments="", 
#  )

#  plt.scatter(rho_plot, z_plot, s=1, color="black")
#  plt.xlabel(r"$\rho$")
#  plt.ylabel(r"$z_{\max}$")




### LOADING AND PLOTTING ###

plt.figure(figsize=(14, 7))
plt.rcParams.update({"font.size": 16})
plt.tick_params(axis="both", labelsize=16)
plt.title("Lorenz System pseudo-bifurcation diagram", fontsize=24, pad=16)

for rho in rho_values[:-2]:
 
 data = np.loadtxt(
    f"data_bifurcation/upper_branch/lorenz_bifurcation_{rho}_{rho + 50}.csv",
    delimiter=",",
    skiprows=1,   
 )

 rho_plot = data[:, 0]
 z_plot = data[:, 1]

 plt.scatter(rho_plot, z_plot, s=0.0001, color="black")

 plt.xlabel(r"$\rho$", fontsize=20)
 plt.ylabel(r"$Z_{\max}$", fontsize=20)

plt.xlim(0, 350)
plt.ylim(0, 400)

plt.savefig("bifurcation_diagram.png", dpi=300, bbox_inches='tight')

plt.show()


###### SECOND BRANCH #######

# initial_states = [-1.,-1,1]
 
# ## COMPUTING ###

# for rho in rho_values:
 
 
#  results, rho_plot, z_plot = maxima_Z_fast(
#         initial_state=initial_states,
#         rho_values=np.linspace(rho, rho + 50, 400),
#         t_span=(t_init, t_final),
#         t_transient=100,
#         sigma=10.0,
#         beta=8 / 3,
#     )

#  data = np.column_stack((rho_plot, z_plot))

#  np.savetxt(
#     f"data_bifurcation/lower_branch/lorenz_bifurcation_{rho}_{rho + 50}.csv",
#     data,
#     delimiter=",",
#     header="rho,z_max",
#     comments="", 
#  )

#  plt.scatter(rho_plot, z_plot, s=1, color="black")
#  plt.xlabel(r"$\rho$")
#  plt.ylabel(r"$z_{\max}$")




## LOADING AND PLOTTING ###

# plt.figure(figsize=(14, 7))

# for rho in rho_values[:-6]:
 
#  data = np.loadtxt(
#     f"data_bifurcation/lower_branch/lorenz_bifurcation_{rho}_{rho + 50}.csv",
#     delimiter=",",
#     skiprows=1,
#  )

#  rho_plot = data[:, 0]
#  z_plot = data[:, 1]

#  plt.scatter(rho_plot, z_plot, s=0.0001, color="red")

#  plt.xlabel(r"$\rho$", fontsize=20)
#  plt.ylabel(r"$z_{\max}$", fontsize=20)

# plt.savefig("bifurcation_diagram.png", dpi=300, bbox_inches='tight')

# plt.show()




