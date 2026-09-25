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
import seaborn as sns


initial_states = [[1.,1,1],[-1, -1, 1],[5,6,8],[-4,-5,7]]



t_init = 0
t_final = 80
t_step = 1/1000
t_transient = 40


t_span = (0, 80)
t_eval = np.linspace(*t_span, int((t_span[1] - t_span[0]) / t_step) + 1)

rho_values = [0.5, 20, 50, 100.35, 120, 133 ,140, 160, 175 ,200, 220, 360]




fig, animation = animate_lorenz_rho_comparison(
    initial_states=initial_states,
    rho_values=rho_values,
    t_span=t_span,
    t_eval=t_eval,
    # interval=None,
    interval=10**(-6),
    frame_step=100,
    t_transient=t_transient,
)



##### TO DO :  
   # 1. Add a function to save the animation as a video file (e.g., MP4 or GIF).        
   # 2. Add an instance for plotting without animation
   # 3. Add function for finding n-periodic orbit
   # 4. Add a function to compute Lyapunov exponents