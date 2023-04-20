"""
Script to plot case 1.
"""
import matplotlib.pyplot as plt
import numpy as np


def get_slowness_a(theta, c11, c55, rho=1):
    """Get slowness as a function of theta for v_A"""
    term1 = c11 * (np.sin(theta) ** 2)
    term2 = c55 * np.cos(theta) ** 2
    velocity = np.sqrt((term1 + term2) / rho)
    return 1 / velocity


def get_slowness_b(theta, c33, c55, rho=1):
    """Get slowness as a function of theta for v_A"""
    term1 = c33 * (np.cos(theta) ** 2)
    term2 = c55 * np.sin(theta) ** 2
    velocity = np.sqrt((term1 + term2) / rho)
    return 1 / velocity


def make_slowness_plots(theta, c11, c33, c55, rho=1, title=""):
    """Make plot of slowness"""
    theta_deg = np.rad2deg(theta)
    slow_a = get_slowness_a(theta, c11, c55, rho=rho)
    slow_b = get_slowness_b(theta, c33, c55, rho=rho)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle(title)

    # plot slowness v theta
    ax1.plot(theta_deg, slow_a, label="$V_a$")
    ax1.plot(theta_deg, slow_b, label="$V_b$")
    ax1.set_xlabel(r"$\theta$ (degrees)")
    ax1.set_ylabel(r"relative slowness")
    ax1.legend()

    # plot radial slowness
    ax2.plot(np.sin(theta) * slow_a, np.cos(theta) * slow_a, label="$V_a$")
    ax2.plot(np.sin(theta) * slow_b, np.cos(theta) * slow_b, label="$V_b$")
    ax2.set_ylabel("relative vertical slowness")
    ax2.set_xlabel("relative horizontal slowness")
    ax2.set_ylim(0, 1.25)
    ax2.set_xlim(0, 1.25)
    ax2.invert_yaxis()
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')

    ax2.set_aspect('equal')
    plt.tight_layout()

    return fig, ax1


if __name__ == "__main__":
    theta = np.linspace(0, np.pi / 2, 200)
    # case 1
    fig1, _ = make_slowness_plots(
        theta=theta,
        c11=1.5,
        c33=1.5,
        c55=1,
        # title='case_1: $c_{11} > c_{55}$',
        title='',

    )
    fig1.savefig('outputs/case1.png')

    fig2, _ = make_slowness_plots(
        theta=theta,
        c11=1.0,
        c33=1.5,
        c55=1,
        title='',
        # title='case_2a: $c_{11} = c_{55}$',
    )
    fig2.savefig("outputs/case_2a.png")

    fig3, _ = make_slowness_plots(
        theta=theta,
        c11=0.75,
        c33=1.5,
        c55=1,
        title='',
        # title='case_2b: $c_{11} < c_{55}$',
    )
    fig3.savefig("outputs/case_2b.png")
