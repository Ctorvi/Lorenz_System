import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from functions.lorenz import lorenz
from functions.lorenz import evolve_lorenz
from functions.lorenz import z_maximum_event



def poincare_section_vs_rho( initial_state,rho_values,t_span, t_transient,sigma=10.0, beta=8 / 3):
    """
    Calcule les intersections avec x = 0 et dx/dt > 0
    pour différentes valeurs de rho.

    Returns
    -------
    results : dict
        results[rho] contient les intersections de forme (n, 3).
    """

    initial_state = np.asarray(initial_state, dtype=float)

    if initial_state.shape != (3,):
        raise ValueError("initial_state doit être de forme (3,).")

    def section_x_zero(t, state, sigma, rho, beta):
        return state[0]

    section_x_zero.direction = 1
    section_x_zero.terminal = False

    results = {}

    rho_plot = []
    z_plot = []

    for rho in rho_values:
        args = (sigma, rho, beta)

        solution = solve_ivp(
            lorenz,
            t_span,
            initial_state,
            args=args,
            events=section_x_zero,
            rtol=1e-9,
            atol=1e-11,
            max_step=0.05,
        )

        if not solution.success:
            raise RuntimeError(
                f"Échec de l'intégration pour rho={rho}: "
                f"{solution.message}"
            )

        event_times = solution.t_events[0]
        event_states = solution.y_events[0]

        # Suppression du régime transitoire
        mask = event_times >= t_transient
        intersections = event_states[mask]

        results[rho] = intersections

        if len(intersections) > 0:
            z_intersections = intersections[:, 2]

            rho_plot.extend(
                np.full(len(z_intersections), rho)
            )
            z_plot.extend(z_intersections)

    plt.plot(rho_plot, z_plot, marker="o", linestyle="", markersize=1, color="black")

    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$z$ sur la section $x=0$")
    plt.title(
        r"Intersections de Poincaré : "
        r"$x=0$, $\dot{x}>0$"
    )
    plt.show()
    return results



def maxima_Z(
    initial_state,
    rho_values,
    t_span,
    t_eval,
    t_transient,
    sigma=10.0,
    beta=8 / 3,
):
    """
    Calcule les maxima locaux de z(t), après le régime transitoire,
    pour différentes valeurs de rho.

    Returns
    -------
    results : dict
        results[rho] contient les maxima locaux de z pour cette
        valeur de rho.

    rho_plot : np.ndarray
        Valeurs de rho répétées pour chaque maximum.

    z_plot : np.ndarray
        Ensemble des maxima de z.
    """
    initial_state = np.asarray(initial_state, dtype=float)

    if initial_state.shape != (3,):
        raise ValueError("initial_state doit être de forme (3,).")

    results = {}

    rho_plot = []
    z_plot = []

    for rho in rho_values:
        args = (sigma, rho, beta)

        traj, time = evolve_lorenz(
            initial_state,
            t_span,
            t_eval,
            args=args,
        )

        # Suppression du régime transitoire
        transient_mask = time >= t_transient
        traj_after_transient = traj[:, transient_mask]

        # Coordonnée z après le régime transitoire
        z = traj_after_transient[2]

        # Détection des maxima locaux stricts
        maxima_mask = (
            (z[1:-1] > z[:-2])
            & (z[1:-1] > z[2:])
        )

        maxima_indices = np.where(maxima_mask)[0] + 1
        z_maxima = z[maxima_indices]

        # Stockage des résultats pour cette valeur de rho
        results[float(rho)] = z_maxima

        # Données destinées au diagramme
        rho_plot.extend(
            np.full(len(z_maxima), rho)
        )
        z_plot.extend(z_maxima)

    rho_plot = np.asarray(rho_plot)
    z_plot = np.asarray(z_plot)

    plt.figure(figsize=(9, 6))
    plt.plot(
        rho_plot,
        z_plot,
        marker="o",
        linestyle="",
        markersize=1,
        color="black",
    )
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$z_{\mathrm{max}}$")
    plt.title(r"Diagramme de bifurcation du système de Lorenz")
    plt.grid(alpha=0.2)
    plt.show()

    return results, rho_plot, z_plot



def maxima_Z_fast(
    initial_state,
    rho_values,
    t_span,
    t_transient,
    sigma=10.0,
    beta=8 / 3,
    rtol=1e-7,
    atol=1e-9,
):
    """
    Calcule les maxima de z pour différentes valeurs de rho,
    sans imposer un tableau t_eval dense.
    """
    initial_state = np.asarray(initial_state, dtype=float)

    if initial_state.shape != (3,):
        raise ValueError("initial_state doit être de forme (3,).")

    results = {}
    rho_plot = []
    z_plot = []

    for rho in rho_values:
        solution = solve_ivp(
            lorenz,
            t_span,
            initial_state,
            args=(sigma, rho, beta),
            events=z_maximum_event,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )

        if not solution.success:
            raise RuntimeError(
                f"Échec de l'intégration pour rho={rho}: "
                f"{solution.message}"
            )
        event_times = solution.t_events[0]
        event_states = solution.y_events[0]

        if event_times.size == 0:
          z_maxima = np.empty(0)
        else:
          mask = event_times >= t_transient
          z_maxima = event_states[mask, 2]

# Si aucun maximum : valeur asymptotique de z
        if z_maxima.size == 0:
          z_maxima = np.array([solution.y[2, -1]])

        results[float(rho)] = z_maxima

        rho_plot.extend(np.full(z_maxima.size, rho))
        z_plot.extend(z_maxima)

    rho_plot = np.asarray(rho_plot)
    z_plot = np.asarray(z_plot)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        rho_plot,
        z_plot,
        s=1,
        color="black",
        linewidths=0,
        rasterized=True,
    )

    ax.set(
        xlabel=r"$\rho$",
        ylabel=r"$z_{\max}$",
        title="Diagramme de bifurcation du système de Lorenz",
    )

    plt.show()

    return results, rho_plot, z_plot