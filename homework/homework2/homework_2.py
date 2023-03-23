"""
Code for homework 2.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def get_f(vp_vs_ratio):
    """Get f"""

    return 1 - 1 / (vp_vs_ratio ** 2)


def exact_p_velocity(
        theta,
        epsilon=0.6,
        delta=0.1,
        vp_vs_ratio=2
):
    """
    Calculate the exact P velocity in terms of Vp_0.

    Parameters
    ----------
    theta
        A vector of theta values to compute.
    epsilon
        The epsilon thomsen parameter
    delta
        The delta thomsen parameter
    vp_vs_ratio
        The ration of Vp_0 to Vs_0.
    """
    f = get_f(vp_vs_ratio)
    sin_theta = np.sin(theta)

    term1 = 1 + epsilon * sin_theta ** 2 - f / 2
    term2 = (1 + (2 * (epsilon * sin_theta**2)/f)) ** 2
    term3 = (2 * (epsilon - delta) * np.sin(2 * theta) ** 2) / f

    out = term1 + f / 2 * np.sqrt(term2 - term3)
    return np.sqrt(out)


def approx_p_velocity(
        theta,
        epsilon=0.6,
        delta=0.1,
        vp_vs_ratio=2
):
    """
    Calculate the approximate P velocity using quadratic term of weak anisotropy.

    Parameters
    ----------
    theta
        A vector of theta values to compute.
    epsilon
        The epsilon thomsen parameter
    delta
        The delta thomsen parameter
    vp_vs_ratio
        The ration of Vp_0 to Vs_0.
    """
    f = get_f(vp_vs_ratio)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    term1 = 1 + 2 * delta * sin_theta ** 2 * cos_theta ** 2
    term2 = 2 * epsilon * sin_theta ** 4
    term3 = (4 / f) * (epsilon - delta) * (epsilon * sin_theta ** 2 + delta * cos_theta ** 2)
    term4 = sin_theta ** 4 * cos_theta ** 2

    out = term1 + term2 + term3 * term4
    return np.sqrt(out)


def compare(theta, epsilon, deltas, vp_vs_ratio):
    """Create a plot comparing the approximate to exact for different values."""
    fig, axes = plt.subplots(2, len(deltas), sharex=True, sharey='row', figsize=(10, 1.5 * len(deltas)))

    theta_degrees = np.rad2deg(theta)

    for ind, delta in enumerate(deltas):
        ax1, ax2 = axes[:, ind]

        # calc approx and exact solutions
        approx = approx_p_velocity(theta, epsilon, delta, vp_vs_ratio)
        exact = exact_p_velocity(theta, epsilon, delta, vp_vs_ratio)
        resid = approx - exact

        # plot both
        ax1.plot(theta_degrees, approx, ls='--', label='approx')
        ax1.plot(theta_degrees, exact, ls='-', label='exact')
        ax1.set_title(f'$\delta=${delta:.02f}')
        ax1.set_xlim(0, 90)

        # plot residuals
        ax2.plot(theta_degrees, resid, label='Residual')
        ax2.set_xlabel(r"angle from vertical ($\theta$)")
        ax2.set_xlim(0, 90)

        # apply y axis labels
        if ind == 0:
            ax1.set_ylabel("V / $V_{0}$")
            ax1.legend()
            ax2.set_ylabel("$(V_{approx} - V_{exact})/V_{0}$")
    return fig, axes


if __name__ == "__main__":
    # define parameters
    theta_radians = np.linspace(0, np.pi / 2, 200)
    vp_vs_ratio = 2
    epsilon = 0.6
    deltas = [0.1, 0.3, 0.5, epsilon]
    # create paths
    output_path = Path(__file__).parent / 'output'
    output_path.mkdir(exist_ok=True, parents=True)

    fig, _ = compare(theta_radians, epsilon, deltas, vp_vs_ratio)

    fig.savefig(output_path / f'comparison.png')
