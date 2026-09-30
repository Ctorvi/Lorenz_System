import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import subprocess
import tempfile
from pathlib import Path
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

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

    omega_audio = gaussian_filter1d(omega_audio, sigma=0.2 * sample_rate,)  # lissage sur environ 15 ms)

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
    video_filename=None,
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
            label=(rf"$\vec{{r}}_0 = ({initial_states[index, 0]:g}, "
                 f"{initial_states[index, 1]:g}, {initial_states[index, 2]:g})$")
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

    def update(export_frame):
        nonlocal start_time

        if video_filename is not None:
        # Lors de l'export, Matplotlib fournit directement le numéro d'image.
          frame = export_frame
          elapsed = frame * interval / 1000
        else:
          if start_time is None:
            if audio:
                sd.play(
                    mixed_sound,
                    samplerate=sample_rate,
                    blocking=False,
                    latency="high",
                )
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

            step = 5

            trajectory.set_data(x[:frame + 1:step],y[:frame + 1:step],)
            trajectory.set_3d_properties(z[:frame + 1:step])

            point.set_data([x[frame]], [y[frame]])
            point.set_3d_properties([z[frame]])


            artists.extend([trajectory, point])

        ax.set_title(f"Trajectoires de Lorenz — t = {t_eval[frame]:.2f}")

        if video_filename is None and elapsed >= duration and animation is not None:
          animation.event_source.stop()

        return tuple(artists)

    def on_close(_):
        if audio:
            sd.stop()

    fig.canvas.mpl_connect("close_event", on_close)

    animation = FuncAnimation(
        fig,
        update,
        frames=range(len(t_eval)) if video_filename is not None else None,
        interval=interval,
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )

    if video_filename is not None:
        if not audio:
            raise ValueError("Utilise audio=True pour exporter la vidéo avec son.")

        output = Path(video_filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            silent_video = temp_dir / "video.mp4"
            wav_file = temp_dir / "audio.wav"

            write(wav_file, sample_rate, mixed_sound)
            animation.save(
                silent_video,
                writer="ffmpeg",
                fps=1000 / interval,
            )

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(silent_video),
                    "-i", str(wav_file),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    str(output),
                ],
                check=True,
            )

        plt.close(fig)
        print(f"Vidéo enregistrée : {output}")
        return None

    plt.show()
    return animation


# def export_lorenz_rho_comparison(
#     rho_values,
#     initial_states,
#     t_span,
#     t_eval,
#     filename="lorenz_rho_comparison.mp4",
#     interval=10,
#     sigma=10.0,
#     beta=8 / 3,
#     base_frequency=180.0,
#     frequency_gain=30.0,
#     sample_rate=44100,
#     fps=30,
# ):

#     rho_values = list(rho_values)
#     initial_states = np.asarray(initial_states, dtype=float)
#     t_eval = np.asarray(t_eval, dtype=float)

#     if not rho_values:
#         raise ValueError("rho_values ne peut pas être vide.")

#     if initial_states.ndim == 1:
#         initial_states = initial_states[np.newaxis, :]

#     if initial_states.ndim != 2 or initial_states.shape[1] != 3:
#         raise ValueError("initial_states doit être de forme (3,) ou (n, 3).")

#     if len(t_eval) < 2 or np.any(np.diff(t_eval) <= 0):
#         raise ValueError("t_eval doit contenir au moins deux temps croissants.")

#     if interval <= 0 or fps <= 0:
#         raise ValueError("interval et fps doivent être positifs.")

#     # Calculer les mêmes conditions initiales pour chaque rho.
#     panels = []
#     for rho in rho_values:
#         solutions = [
#             evolve_lorenz(
#                 initial_state,
#                 t_span,
#                 t_eval,
#                 args=(sigma, rho, beta),
#             )[0]
#             for initial_state in initial_states
#         ]

#         if any(len(x) != len(t_eval) for x, y, z in solutions):
#             raise ValueError(f"Solution incomplète pour rho={rho}.")

#         panels.append(solutions)

#     n_columns = min(4, len(rho_values))
#     n_rows = int(np.ceil(len(rho_values) / n_columns))

#     fig = plt.figure(figsize=(5 * n_columns, 4 * n_rows))
#     panel_artists = []
#     panel_colors = plt.cm.viridis(np.linspace(0, 1, len(rho_values)))

#     for index, (rho, solutions) in enumerate(zip(rho_values, panels)):
#         ax = fig.add_subplot(n_rows, n_columns, index + 1, projection="3d")
#         panel_color = panel_colors[index]

#         all_x = np.concatenate([x for x, y, z in solutions])
#         all_y = np.concatenate([y for x, y, z in solutions])
#         all_z = np.concatenate([z for x, y, z in solutions])

#         def limits(values):
#             low, high = values.min(), values.max()
#             margin = max(0.05 * (high - low), 1e-3)
#             return low - margin, high + margin

#         ax.set_xlim(*limits(all_x))
#         ax.set_ylim(*limits(all_y))
#         ax.set_zlim(*limits(all_z))
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.set_zlabel("z")

#         lines = []
#         points = []

#         for _ in solutions:
#             line, = ax.plot([], [], [], color=panel_color, lw=0.8)
#             point, = ax.plot(
#                 [], [], [],
#                 marker="o",
#                 color=panel_color,
#                 markersize=4,
#             )
#             lines.append(line)
#             points.append(point)

#         panel_artists.append((ax, rho, solutions, lines, points))

#     fig.tight_layout(rect=(0, 0, 1, 0.96))

#     # interval fixe la vitesse de la simulation dans la vidéo.
#     duration = (len(t_eval) - 1) * interval / 1000
#     n_frames = int(np.ceil(duration * fps))
#     video_duration = n_frames / fps

#     # Le son couvre exactement la durée de la vidéo.
#     n_samples = int(round(video_duration * sample_rate))
#     audio_time = np.arange(n_samples) / sample_rate
#     simulation_time = np.clip(
#         t_eval[0]
#         + audio_time / duration * (t_eval[-1] - t_eval[0]),
#         t_eval[0],
#         t_eval[-1],
#     )

#     mixed_sound = np.zeros(n_samples, dtype=float)
#     n_voices = len(rho_values) * len(initial_states)

#     for solutions in panels:
#         for x, y, z in solutions:
#             dx_dt = np.gradient(x, t_eval)
#             dy_dt = np.gradient(y, t_eval)

#             radius_squared = x**2 + y**2
#             omega = (
#                 (x * dy_dt - y * dx_dt)
#                 / np.maximum(radius_squared, 1e-8)
#             )

#             omega_audio = np.interp(
#                 simulation_time,
#                 t_eval,
#                 np.abs(omega),
#             )
#             frequency = base_frequency + frequency_gain * omega_audio
#             frequency = np.clip(frequency, 20, 0.45 * sample_rate)

#             phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
#             mixed_sound += np.sin(phase)

#     mixed_sound *= 0.03 / n_voices

#     fade_length = min(round(0.02 * sample_rate), n_samples // 2)
#     if fade_length:
#         fade = np.linspace(0, 1, fade_length)
#         mixed_sound[:fade_length] *= fade
#         mixed_sound[-fade_length:] *= fade[::-1]

#     mixed_sound = mixed_sound.astype(np.float32)

#     def update(image_index):
#         video_time = image_index / fps
#         simulation_t = min(
#             t_eval[0]
#             + video_time / duration * (t_eval[-1] - t_eval[0]),
#             t_eval[-1],
#         )
#         frame = min(
#             np.searchsorted(t_eval, simulation_t, side="right") - 1,
#             len(t_eval) - 1,
#         )

#         artists = []
#         step = 5

#         for ax, rho, solutions, lines, points in panel_artists:
#             for line, point, (x, y, z) in zip(
#                 lines, points, solutions
#             ):
#                 line.set_data(
#                     x[:frame + 1:step],
#                     y[:frame + 1:step],
#                 )
#                 line.set_3d_properties(z[:frame + 1:step])

#                 point.set_data([x[frame]], [y[frame]])
#                 point.set_3d_properties([z[frame]])

#                 artists.extend([line, point])

#             ax.set_title(rf"$\rho = {rho:g}$", pad=10)

#         fig.suptitle(
#             f"Trajectoires de Lorenz — t = {t_eval[frame]:.2f}",
#             y=0.99,
#         )

#         return artists

#     animation = FuncAnimation(
#         fig,
#         update,
#         frames=range(n_frames),
#         interval=1000 / fps,
#         blit=False,
#         repeat=False,
#     )

#     output = Path(filename)

#     with tempfile.TemporaryDirectory() as temp_dir:
#         temp_dir = Path(temp_dir)
#         silent_video = temp_dir / "video.mp4"
#         wav_file = temp_dir / "audio.wav"

#         write(wav_file, sample_rate, mixed_sound)
#         animation.save(
#             silent_video,
#             writer=FFMpegWriter(fps=fps),
#         )

#         subprocess.run(
#             [
#                 "ffmpeg", "-y",
#                 "-i", str(silent_video),
#                 "-i", str(wav_file),
#                 "-c:v", "copy",
#                 "-c:a", "aac",
#                 "-shortest",
#                 str(output),
#             ],
#             check=True,
#         )

#     plt.close(fig)
#     print(f"Vidéo enregistrée : {output}")


def export_lorenz_rho_comparison(
    rho_values,
    initial_states,
    t_span,
    t_eval,
    filename="lorenz_rho_comparison.mp4",
    interval=10,
    sigma=10.0,
    beta=8 / 3,
    base_frequency=180.0,
    frequency_gain=30.0,
    sample_rate=44100,
    fps=30,
):
    import subprocess
    import tempfile
    from pathlib import Path
    from matplotlib.animation import FFMpegWriter

    rho_values = list(rho_values)
    initial_states = np.asarray(initial_states, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    if not rho_values:
        raise ValueError("rho_values ne peut pas être vide.")

    if initial_states.ndim == 1:
        initial_states = initial_states[np.newaxis, :]

    if initial_states.ndim != 2 or initial_states.shape[1] != 3:
        raise ValueError("initial_states doit être de forme (3,) ou (n, 3).")

    if len(t_eval) < 2 or np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval doit contenir au moins deux temps croissants.")

    if interval <= 0 or fps <= 0:
        raise ValueError("interval et fps doivent être positifs.")

    base_frequencies = np.asarray(base_frequency, dtype=float)
    if base_frequencies.ndim == 0:
        base_frequencies = np.full(
            len(rho_values),
            base_frequencies.item(),
        )
    elif base_frequencies.ndim != 1 or len(base_frequencies) != len(rho_values):
        raise ValueError(
            "base_frequency doit être un scalaire ou un vecteur de "
            "même longueur que rho_values."
        )

    if np.any(base_frequencies < 0):
        raise ValueError("base_frequency doit contenir des valeurs positives.")

    # Calculer les mêmes conditions initiales pour chaque rho.
    panels = []
    for rho in rho_values:
        solutions = [
            evolve_lorenz(
                initial_state,
                t_span,
                t_eval,
                args=(sigma, rho, beta),
            )[0]
            for initial_state in initial_states
        ]

        if any(len(x) != len(t_eval) for x, y, z in solutions):
            raise ValueError(f"Solution incomplète pour rho={rho}.")

        panels.append(solutions)

    n_columns = min(4, len(rho_values))
    n_rows = int(np.ceil(len(rho_values) / n_columns))

    fig = plt.figure(figsize=(5 * n_columns, 4.8 * n_rows))
    panel_artists = []
    panel_colors = plt.cm.viridis(np.linspace(0, 0.7, len(rho_values)))

    for index, (rho, solutions) in enumerate(zip(rho_values, panels)):
        ax = fig.add_subplot(n_rows, n_columns, index + 1, projection="3d")
        panel_color = panel_colors[index]

        all_x = np.concatenate([x for x, y, z in solutions])
        all_y = np.concatenate([y for x, y, z in solutions])
        all_z = np.concatenate([z for x, y, z in solutions])

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

        lines = []
        points = []

        for _ in solutions:
            line, = ax.plot([], [], [], color=panel_color, lw=0.8)
            point, = ax.plot(
                [], [], [],
                marker="o",
                color=panel_color,
                markersize=4,
            )
            lines.append(line)
            points.append(point)

        panel_artists.append((ax, rho, solutions, lines, points))

    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=4.0)

    # interval fixe la vitesse de la simulation dans la vidéo.
    duration = (len(t_eval) - 1) * interval / 1000
    n_frames = int(np.ceil(duration * fps))
    video_duration = n_frames / fps

    # Le son couvre exactement la durée de la vidéo.
    n_samples = int(round(video_duration * sample_rate))
    audio_time = np.arange(n_samples) / sample_rate
    simulation_time = np.clip(
        t_eval[0]
        + audio_time / duration * (t_eval[-1] - t_eval[0]),
        t_eval[0],
        t_eval[-1],
    )

    mixed_sound = np.zeros(n_samples, dtype=float)
    n_voices = len(rho_values) * len(initial_states)

    for base_frequency_value, solutions in zip(base_frequencies, panels):
        for x, y, z in solutions:
            dx_dt = np.gradient(x, t_eval)
            dy_dt = np.gradient(y, t_eval)

            radius_squared = x**2 + y**2
            omega = (
                (x * dy_dt - y * dx_dt)
                / np.maximum(radius_squared, 1e-8)
            )

            omega_audio = np.interp(
                simulation_time,
                t_eval,
                np.abs(omega),
            )
            frequency = base_frequency_value + frequency_gain * omega_audio
            frequency = np.clip(frequency, 20, 0.45 * sample_rate)

            phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
            mixed_sound += np.sin(phase)

    mixed_sound *= 0.03 / n_voices

    fade_length = min(round(0.02 * sample_rate), n_samples // 2)
    if fade_length:
        fade = np.linspace(0, 1, fade_length)
        mixed_sound[:fade_length] *= fade
        mixed_sound[-fade_length:] *= fade[::-1]

    mixed_sound = mixed_sound.astype(np.float32)

    def update(image_index):
        video_time = image_index / fps
        simulation_t = min(
            t_eval[0]
            + video_time / duration * (t_eval[-1] - t_eval[0]),
            t_eval[-1],
        )
        frame = min(
            np.searchsorted(t_eval, simulation_t, side="right") - 1,
            len(t_eval) - 1,
        )

        artists = []
        step = 5

        for ax, rho, solutions, lines, points in panel_artists:
            for line, point, (x, y, z) in zip(
                lines, points, solutions
            ):
                line.set_data(
                    x[:frame + 1:step],
                    y[:frame + 1:step],
                )
                line.set_3d_properties(z[:frame + 1:step])

                point.set_data([x[frame]], [y[frame]])
                point.set_3d_properties([z[frame]])

                artists.extend([line, point])

            ax.set_title(rf"$\rho = {rho:g}$", pad=10)

        fig.suptitle(
             f"Trajectoires de Lorenz — t = {t_eval[frame]:.2f}",
             y=0.99,
        )
        
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=range(n_frames),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    output = Path(filename)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        silent_video = temp_dir / "video.mp4"
        wav_file = temp_dir / "audio.wav"

        write(wav_file, sample_rate, mixed_sound)
        animation.save(
            silent_video,
            writer=FFMpegWriter(fps=fps),
        )

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(silent_video),
                "-i", str(wav_file),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output),
            ],
            check=True,
        )

    plt.close(fig)
    print(f"Vidéo enregistrée : {output}")


def export_lorenz_rho_bifurcation_comparison(
    rho_values,
    initial_states,
    t_span,
    t_eval,
    bifurcation_data_dir="data_bifurcation/upper_branch",
    filename="lorenz_rho_bifurcation_comparison.mp4",
    interval=10,
    sigma=10.0,
    beta=8 / 3,
    fps=30,
):
    """Export the bifurcation comparison layout as a silent MP4 video."""

    rho_values = np.asarray(rho_values, dtype=float)
    initial_states = np.asarray(initial_states, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    if rho_values.ndim != 1 or len(rho_values) == 0:
        raise ValueError("rho_values doit contenir au moins une valeur.")
    if initial_states.ndim == 1:
        initial_states = initial_states[np.newaxis, :]
    if initial_states.ndim != 2 or initial_states.shape[1] != 3:
        raise ValueError("initial_states doit être de forme (3,) ou (n, 3).")
    if len(t_eval) < 2 or np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval doit contenir au moins deux temps croissants.")
    if interval <= 0 or fps <= 0:
        raise ValueError("interval et fps doivent être positifs.")

    bifurcation_path = Path(bifurcation_data_dir)
    data_files = sorted(bifurcation_path.glob("lorenz_bifurcation_*.csv"))
    if not data_files:
        raise FileNotFoundError(
            f"Aucun fichier de bifurcation trouvé dans {bifurcation_path}."
        )

    bifurcation_data = np.vstack([
        np.loadtxt(file, delimiter=",", skiprows=1)
        for file in data_files
    ])
    bifurcation_rho = bifurcation_data[:, 0]
    bifurcation_z = bifurcation_data[:, 1]

    solutions = []
    for rho in rho_values:
        solutions.append([
            evolve_lorenz(
                initial_state,
                t_span,
                t_eval,
                args=(sigma, rho, beta),
            )[0]
            for initial_state in initial_states
        ])

    n_columns = max(1, int(np.ceil(len(rho_values) / 2)))
    fig = plt.figure(
        figsize=(4.8 * n_columns, 10),
        facecolor="black",
    )
    grid = GridSpec(
        2,
        n_columns,
        figure=fig,
        height_ratios=[1.6, 2.8],
        hspace=0.4,
        wspace=0.2,
    )
    trajectory_grid = grid[1, :].subgridspec(
        2,
        n_columns,
        hspace=0.12,
        wspace=0.2,
    )

    panel_colors = plt.cm.viridis(
        np.linspace(0.05, 0.95, len(rho_values))
    )
    trajectory_colors = sns.color_palette(
        "husl",
        n_colors=len(initial_states),
    )

    bifurcation_ax = fig.add_subplot(grid[0, :])
    bifurcation_ax.set_facecolor("black")
    bifurcation_ax.scatter(
        bifurcation_rho,
        bifurcation_z,
        s=0.0001,
        marker=".",
        color="white",
    )
    for rho, color in zip(rho_values, panel_colors):
        bifurcation_ax.axvline(
            rho,
            color=color,
            linewidth=1.2,
            alpha=0.95,
        )
    bifurcation_ax.set_xlim(0, 350)
    bifurcation_ax.set_ylim(0, 400)
    bifurcation_ax.set_xlabel(r"$\rho$", color="white")
    bifurcation_ax.set_ylabel(r"$Z_{\max}$", color="white")
    bifurcation_ax.set_title(
        "Diagramme de bifurcation",
        color="white",
        pad=10,
    )
    bifurcation_ax.tick_params(colors="white")
    for spine in bifurcation_ax.spines.values():
        spine.set_color("white")

    trajectories = []
    points = []
    trajectory_axes = []

    for rho_index, (rho, rho_solutions) in enumerate(
        zip(rho_values, solutions)
    ):
        row = rho_index // n_columns
        column = rho_index % n_columns
        ax = fig.add_subplot(
            trajectory_grid[row, column],
            projection="3d",
        )
        trajectory_axes.append(ax)
        ax.set_facecolor("black")

        all_x = np.concatenate([x for x, y, z in rho_solutions])
        all_y = np.concatenate([y for x, y, z in rho_solutions])
        all_z = np.concatenate([z for x, y, z in rho_solutions])

        def axis_limits(values):
            margin = 0.05 * max(np.ptp(values), 1.0)
            return values.min() - margin, values.max() + margin

        ax.set_xlim(*axis_limits(all_x))
        ax.set_ylim(*axis_limits(all_y))
        ax.set_zlim(*axis_limits(all_z))
        ax.text2D(
            0.5,
            -0.08,
            rf"$\rho = {rho:g}$",
            transform=ax.transAxes,
            color=panel_colors[rho_index],
            ha="center",
            va="top",
        )
        ax.tick_params(colors="white", labelsize=0, length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.grid(True)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"].update(
                color=(1.0, 1.0, 1.0, 0.18),
                linewidth=0.4,
            )
            axis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
            axis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))

        rho_trajectories = []
        rho_points = []
        for index, color in enumerate(trajectory_colors):
            trajectory, = ax.plot(
                [],
                [],
                [],
                color=color,
                linewidth=0.1,
                label=(
                    rf"$\vec{{r}}_0 = ({initial_states[index, 0]:g}, "
                    f"{initial_states[index, 1]:g}, "
                    f"{initial_states[index, 2]:g})$"
                ),
            )
            point, = ax.plot(
                [], [], [], marker="o", color=color, markersize=4
            )
            rho_trajectories.append(trajectory)
            rho_points.append(point)

        trajectories.append(rho_trajectories)
        points.append(rho_points)

    for empty_index in range(len(rho_values), 2 * n_columns):
        row = empty_index // n_columns
        column = empty_index % n_columns
        fig.add_subplot(
            trajectory_grid[row, column]
        ).set_visible(False)

    fig.subplots_adjust(
        left=0.0625,
        right=0.95,
        top=0.93,
    )
    title_y = trajectory_axes[0].get_position().y1 + 0.02
    title = fig.text(
        0.32,
        title_y,
        f"Système de Lorenz — t = {t_eval[0]:.2f}",
        color="white",
        ha="center",
        va="bottom",
    )
    handles, labels = trajectory_axes[0].get_legend_handles_labels()
    legend_handles = [
        Line2D(
            [],
            [],
            color=handle.get_color(),
            linestyle=handle.get_linestyle(),
            linewidth=1.5,
        )
        for handle in handles
    ]
    fig.legend(
        legend_handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.62, title_y),
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
        fontsize=8,
    )

    duration = (len(t_eval) - 1) * interval / 1000.0
    frame_count = max(2, int(np.ceil(duration * fps)))

    def update(image_index):
        video_time = image_index / fps
        simulation_t = min(
            t_eval[0]
            + video_time / duration * (t_eval[-1] - t_eval[0]),
            t_eval[-1],
        )
        frame = min(
            np.searchsorted(t_eval, simulation_t, side="right") - 1,
            len(t_eval) - 1,
        )

        artists = []
        step = 5
        for rho_index, rho_solutions in enumerate(solutions):
            for trajectory, point, (x, y, z) in zip(
                trajectories[rho_index],
                points[rho_index],
                rho_solutions,
            ):
                trajectory.set_data(
                    x[:frame + 1:step],
                    y[:frame + 1:step],
                )
                trajectory.set_3d_properties(z[:frame + 1:step])
                point.set_data([x[frame]], [y[frame]])
                point.set_3d_properties([z[frame]])
                artists.extend([trajectory, point])

        title.set_text(f"Système de Lorenz — t = {t_eval[frame]:.2f}")
        return tuple(artists)

    animation = FuncAnimation(
        fig,
        update,
        frames=range(frame_count),
        interval=1000 / fps,
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )

    output = Path(filename)
    animation.save(
        output,
        writer=FFMpegWriter(fps=fps),
    )
    plt.close(fig)
    print(f"Vidéo enregistrée : {output}")

