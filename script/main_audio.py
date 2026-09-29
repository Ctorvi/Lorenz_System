import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functions.lorenz import lorenz
from functions.lorenz import evolve_lorenz
from functions.lorenz import plot_lorenz_trajectories
from functions.lorenz import z_maximum_event
from functions.sounds import plot_lorenz_trajectories_with_audio
from functions.sounds import export_lorenz_rho_comparison
from functions.sounds import export_lorenz_rho_bifurcation_comparison
from functions.bifurcation import maxima_Z
from functions.bifurcation import maxima_Z_fast
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
import sounddevice as sd
from scipy.io.wavfile import write

t_init = 0
t_final = 20
t_step = 1/1000

t_span = (t_init, t_final)
t_eval = np.linspace(*t_span, int((t_span[1] - t_span[0]) / t_step) + 1)

export_lorenz_rho_comparison(
    # rho_values=[20,50,100.35, 150, 160, 220,250, 360],
    rho_values=[93,100.35,133, 150, 160, 220,250, 360],
    initial_states=[[1, 1, 1]],
    t_span=t_span,
    t_eval=t_eval,
    interval=2,
    filename="8_periodics.mp4",
    base_frequency=100.0,
    frequency_gain=20.0,
)




# rho_values = [50, 93, 100.35, 133 ,140, 150, 160,181.5, 200, 220, 270,330]
# initial_states = [[1.,1,1],[5,6,8],[-4,-5,7]]

# export_lorenz_rho_bifurcation_comparison(
#     rho_values=rho_values,
#     initial_states=initial_states,
#     t_span=t_span,
#     t_eval=t_eval,
#     bifurcation_data_dir="data_bifurcation/upper_branch",
#     filename="lorenz_rho_bifurcation_comparison.mp4",
#     interval=1,
#     fps=30,
# )