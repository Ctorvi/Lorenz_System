import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functions.lorenz import lorenz
from functions.lorenz import evolve_lorenz
from functions.lorenz import plot_lorenz_trajectories
# from functions.lorenz import trajectory_plane_intersection_x1_zero
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
from functions.lorenz import animate_lorenz_rho_comparison
 


# sigma = 10
# rho = 10
# beta = 8 / 3
# t_init=0
# t_final=100
# sample_points=5000 
# t_eval = np.linspace(t_init, t_final, sample_points)


# initial_state=[1, 1, 1]  
initial_states = [[1.,1,1],[-1, -1, 1],[5,6,8],[-4,-5,7]]





# evolution, time  = evolve_lorenz(
#     initial_state=initial_state,
#     t_span=(t_init, t_final),
#     t_eval=t_eval,
#     args=(sigma, rho, beta)
# )

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
#     interval=0.0000001,
#     args=(10.0, 5, 8/3)
# )




t_span = (0, 140)
t_eval = np.linspace(*t_span, 1000000)

rho_values = [0.5, 20, 50, 100.35, 120, 133 ,140, 160, 175 ,200, 220, 360]

fig, animation = animate_lorenz_rho_comparison(
    initial_states=initial_states,
    rho_values=rho_values,
    t_span=t_span,
    t_eval=t_eval,
    interval=None,
    frame_step=5000,
    t_transient=100
)


