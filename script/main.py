import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functions.lorenz import lorenz
from functions.lorenz import evolve_lorenz
from functions.lorenz import plot_lorenz_trajectories
# from functions.lorenz import trajectory_plane_intersection_x1_zero
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
 


sigma = 10
rho = 28
beta = 8 / 3
t_init=0
t_final=100
sample_points=100000

initial_state=[1.01, 1, 1]  
initial_states = [[1.01,1,1],[1, 1, 1]]
t_eval = np.linspace(t_init, t_final, sample_points)




evolution, time  = evolve_lorenz(
    initial_state=initial_state,
    t_span=(t_init, t_final),
    t_eval=t_eval,
    args=(sigma, rho, beta)
)

# solution=solve_ivp(
#     lorenz,
#     t_span=(t_init, t_final),
#     y0=initial_state,
#     t_eval=t_eval,
#     args=(sigma, rho, beta)
# )



# fig= plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# plt.title("Lorenz Attractor")

# plot_lorenz_trajectories(
#     fig=fig,
#     ax=ax,
#     initial_states=initial_states,
#     t_span=(t_init, t_final),
#     t_eval=t_eval,
#     interval=None,
#     args=(10.0, 28.0, 8/3)
# )




