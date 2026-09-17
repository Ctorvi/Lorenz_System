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
 

initial_states = [[1.01,1,1]]

rho_values = np.linspace(0, 50, 100)

results, rho_plot, z_plot = maxima_Z_fast(
    initial_state=[1, 1, 1],
    rho_values=rho_values,
    t_span=(0, 200),
    t_transient=100,
)

data = np.column_stack((rho_plot, z_plot))

np.savetxt(
    "lorenz_bifurcation_0_100.csv",
    data,
    delimiter=",",
    header="rho,z_max",
    comments="",
)

### reload datas ###

# data = np.loadtxt(
#     "lorenz_bifurcation.csv",
#     delimiter=",",
#     skiprows=1,
# )

# rho_plot = data[:, 0]
# z_plot = data[:, 1]

# plt.scatter(rho_plot, z_plot, s=1, color="black")
# plt.xlabel(r"$\rho$")
# plt.ylabel(r"$z_{\max}$")
# plt.show()
