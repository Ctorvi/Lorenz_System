import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functions.lorenz import lorenz
from functions.lorenz import evolve_lorenz
from functions.lorenz import plot_lorenz_trajectories
from functions.lorenz import z_maximum_event
from functions.sounds import plot_lorenz_trajectories_with_audio
from functions.bifurcation import maxima_Z
from functions.bifurcation import maxima_Z_fast
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
import sounddevice as sd
from scipy.io.wavfile import write


sigma = 10
rho = 10
beta = 8 / 3
t_init=0
t_final=30
sample_points=1000 
t_eval = np.linspace(t_init, t_final, sample_points)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

animation = plot_lorenz_trajectories_with_audio(
    fig, ax,
    initial_states=[[1, 1, 1]],  #,[-1,-1,8],[4,6,7],[-4,-5,7]],
    t_span=(0, 30),
    t_eval=t_eval,
    interval=1000,
    args=(10.0, 5.0, 8/3),
    audio=True,     
    base_frequency=150.0,
    frequency_gain=1.0,
)