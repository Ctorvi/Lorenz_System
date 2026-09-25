import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from functions.lorenz import lorenz
from functions.lorenz import evolve_lorenz
from functions.lorenz import z_maximum_event
from functions.bifurcation import maxima_Z
from functions.bifurcation import maxima_Z_fast
import numpy as np
from scipy.io.wavfile import write
import time
import sounddevice as sd
from scipy.ndimage import gaussian_filter1d
from scipy.io.wavfile import write


def lorenz_trajectory_to_wav(
    solution,
    filename="lorenz.wav",
    audio_duration=15.0,
    base_frequency=180.0,
    frequency_gain=30.0,
    sample_rate=44100,
):
    t = solution.t
    x, y = solution.y[0], solution.y[1]

    if len(t) < 2 or t[-1] <= t[0]:
        raise ValueError("La solution doit contenir au moins deux instants distincts.")

    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)

    radius_squared = x**2 + y**2
    omega = (x * dy_dt - y * dx_dt) / np.maximum(radius_squared, 1e-8)

    # Chaque instant audio correspond à un instant de la simulation.
    audio_t = np.linspace(t[0], t[-1], int(audio_duration * sample_rate))
    omega_audio = np.interp(audio_t, t, np.abs(omega))

    omega_audio = gaussian_filter1d(
    omega_audio, sigma=0.2 * sample_rate,)  # lissage sur environ 15 ms)

    # Conversion choisie pour obtenir une hauteur audible, en hertz.
    # frequency = base_frequency + frequency_gain * omega_audio
    # frequency = np.clip(frequency, 20.0, 0.45 * sample_rate)

    frequency = base_frequency + frequency_gain * omega_audio
    frequency = np.clip(frequency, 40.0, 2000.0)

    # Intégrer la fréquence maintient la phase continue lorsque la note change.
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    sound = 0.25 * np.sin(phase)

    # Courtes rampes pour éviter les clics au début et à la fin.
    fade_length = min(int(0.02 * sample_rate), len(sound) // 2)
    if fade_length:
        fade = np.linspace(0.0, 1.0, fade_length)
        sound[:fade_length] *= fade
        sound[-fade_length:] *= fade[::-1]

    write(filename, sample_rate, sound.astype(np.float32))
    return omega, frequency



def plot_lorenz_trajectories_with_audio(
    fig,
    ax,
    initial_states,
    t_span,
    t_eval,
    interval=10,
    args=(10.0, 28.0, 8/3),
    audio=True,
    base_frequency=180.0,
    frequency_gain=30.0,
    sample_rate=44100,
):
    initial_states = np.asarray(initial_states, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    if initial_states.ndim == 1:
        initial_states = initial_states[np.newaxis, :]

    if initial_states.ndim != 2 or initial_states.shape[1] != 3:
        raise ValueError("initial_states doit être de forme (3,) ou (n, 3).")

    if len(t_eval) < 2 or np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval doit contenir au moins deux temps croissants.")

    solutions = [
        evolve_lorenz(initial_state, t_span, t_eval, args=args)[0]
        for initial_state in initial_states
    ]

    if any(len(x) != len(t_eval) for x, y, z in solutions):
        raise ValueError("Les solutions doivent être évaluées aux temps de t_eval.")

    all_x = np.concatenate([x for x, y, z in solutions])
    all_y = np.concatenate([y for x, y, z in solutions])
    all_z = np.concatenate([z for x, y, z in solutions])

    # Une petite marge évite les limites identiques pour une coordonnée constante.
    def limits(values):
        low, high = values.min(), values.max()
        margin = max(0.05 * (high - low), 1e-3)
        return low - margin, high + margin

    ax.set_xlim(*limits(all_x))
    ax.set_ylim(*limits(all_y))
    ax.set_zlim(*limits(all_z))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Trajectoires du système de Lorenz")

    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_states)))

    if interval is None:
        for index, ((x, y, z), color) in enumerate(zip(solutions, colors)):
            ax.plot(
                x, y, z,
                color=color,
                lw=0.8,
                label=f"Trajectoire {index + 1}",
            )
        ax.legend()
        plt.show()
        return None

    if interval <= 0:
        raise ValueError("interval doit être positif, en millisecondes.")

    trajectories = []
    points = []

    for index, color in enumerate(colors):
        trajectory, = ax.plot(
            [], [], [],
            color=color,
            lw=0.8,
            label=f"Trajectoire {index + 1}",
        )
        point, = ax.plot(
            [], [], [],
            marker="o",
            color=color,
            markersize=4,
        )
        trajectories.append(trajectory)
        points.append(point)

    ax.legend()

    # Durée nominale de l'animation à la vitesse demandée.
    duration = (len(t_eval) - 1) * interval / 1000.0

    if audio:
        n_samples = max(1, round(duration * sample_rate))
        audio_time = np.arange(n_samples) / sample_rate

        # Temps simulé correspondant à chaque échantillon audio.
        simulation_time = (
            t_eval[0]
            + audio_time / duration * (t_eval[-1] - t_eval[0])
        )

        mixed_sound = np.zeros(n_samples, dtype=float)

        for x, y, z in solutions:
            dx_dt = np.gradient(x, t_eval)
            dy_dt = np.gradient(y, t_eval)

            # Vitesse angulaire autour de l'origine dans le plan (x, y).
            radius_squared = x**2 + y**2
            omega = (
                (x * dy_dt - y * dx_dt)
                / np.maximum(radius_squared, 1e-8)
            )

            omega_audio = np.interp(
                simulation_time, t_eval, np.abs(omega)
            )
            frequency = base_frequency + frequency_gain * omega_audio
            frequency = np.clip(frequency, 20, 0.45 * sample_rate)

            phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
            mixed_sound += np.sin(phase)

        mixed_sound *= 0.03 / len(solutions)

        fade_length = min(round(0.02 * sample_rate), n_samples // 2)
        if fade_length:
            fade = np.linspace(0, 1, fade_length)
            mixed_sound[:fade_length] *= fade
            mixed_sound[-fade_length:] *= fade[::-1]

        mixed_sound = mixed_sound.astype(np.float32)
        # write("test_lorenz.wav", sample_rate, mixed_sound)

    start_time = None
    animation = None

    def update(_):
        nonlocal start_time

        if start_time is None:
            if audio:
                sd.play(mixed_sound, samplerate=sample_rate, blocking=False,latency="high",)
            start_time = time.perf_counter()

        elapsed = min(time.perf_counter() - start_time, duration)
        simulation_t = (
            t_eval[0]
            + elapsed / duration * (t_eval[-1] - t_eval[0])
        )
        frame = min(
            np.searchsorted(t_eval, simulation_t, side="right") - 1,
            len(t_eval) - 1,
        )

        artists = []

        for trajectory, point, (x, y, z) in zip(trajectories, points, solutions):

            # trajectory.set_data(x[:frame + 1], y[:frame + 1])
            # trajectory.set_3d_properties(z[:frame + 1])

            # point.set_data([x[frame]], [y[frame]])
            # point.set_3d_properties([z[frame]])

            step = 5

            trajectory.set_data(x[:frame + 1:step],y[:frame + 1:step],)
            trajectory.set_3d_properties(z[:frame + 1:step])

            point.set_data([x[frame]], [y[frame]])
            point.set_3d_properties([z[frame]])


            artists.extend([trajectory, point])

        ax.set_title(f"Trajectoires de Lorenz — t = {t_eval[frame]:.2f}")

        if elapsed >= duration and animation is not None:
            animation.event_source.stop()

        return tuple(artists)

    def on_close(_):
        if audio:
            sd.stop()

    fig.canvas.mpl_connect("close_event", on_close)

    animation = FuncAnimation(
        fig,
        update,
        frames=None,
        interval=interval,
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )

    plt.show()
    return animation