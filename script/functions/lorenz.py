import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def lorenz(t, state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state

    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ]


def z_maximum_event(t, state, sigma, rho, beta):
    x, y, z = state
    return x * y - beta * z   # dz/dt

def evolve_lorenz(initial_state, t_span, t_eval, args=(10.0, 28.0, 8/3)):
    solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, args=args)
    return solution.y, solution.t


def plot_lorenz_trajectories(
    fig,
    ax,
    initial_states,
    t_span,
    t_eval,
    interval=10,
    args=(10.0, 28.0, 8/3)
):
    initial_states = np.asarray(initial_states, dtype=float)

    # Autoriser une condition initiale unique : [x0, y0, z0]
    if initial_states.ndim == 1:
        initial_states = initial_states[np.newaxis, :]

    if initial_states.ndim != 2 or initial_states.shape[1] != 3:
        raise ValueError(
            "initial_states doit être de forme (3,) ou (n, 3)."
        )

    

    solutions = [
        evolve_lorenz(initial_state, t_span, t_eval, args=args)[0]
        for initial_state in initial_states
    ]

    # Limites communes à toutes les trajectoires
    all_x = np.concatenate([x for x, y, z in solutions])
    all_y = np.concatenate([y for x, y, z in solutions])
    all_z = np.concatenate([z for x, y, z in solutions])

    ax.set_xlim(all_x.min(), all_x.max())
    ax.set_ylim(all_y.min(), all_y.max())
    ax.set_zlim(all_z.min(), all_z.max())

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Trajectoires du système de Lorenz")

    colors = plt.cm.viridis(
        np.linspace(0, 1, len(initial_states))
    )

    # Cas sans animation
    if interval is None:
        for index, ((x, y, z), color) in enumerate(
            zip(solutions, colors)
        ):
            ax.plot(
                x,
                y,
                z,
                color=color,
                lw=0.8,
                label=f"Trajectoire {index + 1}",
            )

        ax.legend()
        plt.show()
        return None

    # Cas avec animation
    trajectories = []
    points = []

    for index, color in enumerate(colors):
        trajectory, = ax.plot(
            [],
            [],
            [],
            color=color,
            lw=0.8,
            label=f"Trajectoire {index + 1}",
        )

        point, = ax.plot(
            [],
            [],
            [],
            marker="o",
            color=color,
            markersize=4,
        )

        trajectories.append(trajectory)
        points.append(point)

    ax.legend()

    def update(frame):
        artists = []

        for trajectory, point, (x, y, z) in zip(
            trajectories,
            points,
            solutions,
        ):
            trajectory.set_data(
                x[:frame + 1],
                y[:frame + 1],
            )
            trajectory.set_3d_properties(
                z[:frame + 1]
            )

            point.set_data(
                [x[frame]],
                [y[frame]],
            )
            point.set_3d_properties(
                [z[frame]]
            )

            artists.extend([trajectory, point])

        ax.set_title(
            f"Trajectoires de Lorenz — t = {t_eval[frame]:.2f}"
        )

        return tuple(artists)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(t_eval),
        interval=interval,
        blit=False,
    )
    plt.show()
    return animation



def animate_lorenz_rho_comparison(
    initial_states,
    rho_values,
    t_span,
    t_eval,
    sigma=10.0,
    beta=8 / 3,
    interval=10,
    frame_step=1,
    t_transient=0
):
    """
    Anime les trajectoires de Lorenz pour plusieurs valeurs de rho.

    Chaque valeur de rho est affichée dans un sous-graphe différent.
    Les mêmes conditions initiales sont utilisées pour chaque rho.

    Parameters
    ----------
    initial_states : array-like, shape (3,) ou (n, 3)
        Une ou plusieurs conditions initiales.

    rho_values : array-like
        Valeurs du paramètre rho à comparer.

    t_span : tuple
        Intervalle temporel (t_initial, t_final).

    t_eval : array-like
        Instants auxquels la solution est évaluée.

    sigma, beta : float
        Autres paramètres du système de Lorenz.

    interval : int
        Temps entre deux images, en millisecondes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure contenant les sous-graphes.

    animation : matplotlib.animation.FuncAnimation
        Objet animation.
    """

    initial_states = np.asarray(initial_states, dtype=float)
    rho_values = np.asarray(rho_values, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    # Autoriser une condition initiale unique
    if initial_states.ndim == 1:
        initial_states = initial_states[np.newaxis, :]

    if initial_states.ndim != 2 or initial_states.shape[1] != 3:
        raise ValueError(
            "initial_states doit être de forme (3,) ou (n, 3)."
        )

    if rho_values.ndim != 1 or len(rho_values) == 0:
        raise ValueError(
            "rho_values doit contenir au moins une valeur."
        )

    n_plots = len(rho_values)

    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / 4))

    fig, axes_grid = plt.subplots(
     n_rows,
     n_cols,
     figsize=(5 * n_cols, 4.5 * n_rows),
     subplot_kw={"projection": "3d"},
     squeeze=False,
    )

# Conversion de la grille d'axes en une liste
    axes = axes_grid.ravel()

# Supprimer les sous-graphes inutilisés
    for index in range(n_plots, len(axes)):
     fig.delaxes(axes[index])

# Ne conserver que les axes réellement utilisés
    axes = axes[:n_plots]

    # solutions[rho_index][initial_state_index] = (x, y, z)
    solutions = []

    for rho in rho_values:
        rho_solutions = [
            evolve_lorenz(
                initial_state,
                t_span,
                t_eval,
                args=(sigma, rho, beta),
            )[0]
            for initial_state in initial_states
        ]

        solutions.append(rho_solutions)

    transient_mask = t_eval >= t_transient

    if not np.any(transient_mask):
     raise ValueError(
        "t_transient doit être inférieur ou égal au dernier instant de t_eval."
    )

    t_display = t_eval[transient_mask]

    solutions = [
        [
            (
                x[transient_mask],
                y[transient_mask],
                z[transient_mask],
            )
            for x, y, z in rho_solutions
        ]   
        for rho_solutions in solutions
    ]

    colors = plt.cm.viridis(
        np.linspace(0, 1, len(initial_states))
    )

    # Contiendra les courbes et les points de chaque axe
    trajectories = []
    points = []

    for rho_index, (rho, ax) in enumerate(zip(rho_values, axes)):

    # Solutions correspondant uniquement à cette valeur de rho
     rho_solutions = solutions[rho_index]

     all_x = np.concatenate([
        x for x, y, z in rho_solutions
     ])
     all_y = np.concatenate([
        y for x, y, z in rho_solutions
     ])
     all_z = np.concatenate([
        z for x, y, z in rho_solutions
     ])

    # Petite marge autour des trajectoires
     x_margin = 0.05 * max(np.ptp(all_x), 1.0)
     y_margin = 0.05 * max(np.ptp(all_y), 1.0)
     z_margin = 0.05 * max(np.ptp(all_z), 1.0)

     ax.set_xlim(
        all_x.min() - x_margin,
        all_x.max() + x_margin,
     )
     ax.set_ylim(
        all_y.min() - y_margin,
        all_y.max() + y_margin,
     )
     ax.set_zlim(
        all_z.min() - z_margin,
        all_z.max() + z_margin,
     )

     ax.set_xlabel("x")
     ax.set_ylabel("y")
     ax.set_zlabel("z")
     ax.set_title(rf"$\rho = {rho:g}$")

     rho_trajectories = []
     rho_points = []

     for index, color in enumerate(colors):
        trajectory, = ax.plot(
            [],
            [],
            [],
            color=color,
            linewidth=0.8,
            label=f"CI {index + 1}",
          )

        point, = ax.plot(
            [],
            [],
            [],
            marker="o",
            color=color,
            markersize=4,
        )

        rho_trajectories.append(trajectory)
        rho_points.append(point)

     trajectories.append(rho_trajectories)
     points.append(rho_points)

    #  trajectories.append(rho_trajectories)
    #  points.append(rho_points)
    # Une seule légende suffit puisque les conditions initiales sont identiques
    axes[0].legend()

    time_text = fig.suptitle(
    f"Système de Lorenz — t = {t_display[0]:.2f}"
    )

    def update(frame):
        artists = []

        for rho_index, rho_solutions in enumerate(solutions):
            for trajectory, point, (x, y, z) in zip(
                trajectories[rho_index],
                points[rho_index],
                rho_solutions,
            ):
                trajectory.set_data(
                    x[:frame + 1],
                    y[:frame + 1],
                )
                trajectory.set_3d_properties(
                    z[:frame + 1]
                )

                point.set_data(
                    [x[frame]],
                    [y[frame]],
                )
                point.set_3d_properties(
                    [z[frame]]
                )

                artists.extend([trajectory, point])

        time_text.set_text(
          f"Système de Lorenz — t = {t_display[frame]:.2f}"
        )

        artists.append(time_text)

        return tuple(artists)

    animation = FuncAnimation(
        fig,
        update,
        frames=range(0, len(t_display), frame_step),
        interval=interval,
        blit=False,
    )

    fig.tight_layout()
    plt.show()

    return fig, animation





