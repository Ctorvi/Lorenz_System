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
        evolve_lorenz(initial_state, t_span, t_eval, args=(10.0, 28.0, 8/3))[0]
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





