"""
Tests for exact phase velocity equations.
"""
import numpy as np
import anray
import matplotlib.pyplot as plt


if __name__ == "__main__":
    eps = np.deg2rad(1)
    theta = np.linspace(eps, np.pi / 2 - eps, 201)

    # out1 = anray.core._get_exact_phase_velocity(
    #     theta, epsilon=.2, delta=0.3, vp_0=3.0, vs_0=1.5, modifier=-1
    # )
    out1 = anray.core._get_exact_phase_velocity(
        theta, epsilon=0.195, delta=-0.22, vp_0=1.88, vs_0=1, modifier=-1
    )
    gvect = anray.core._get_group_velocity_array(
        theta,
        epsilon=0.195,
        delta=-0.22,
        vp_0=1.88,
        vs_0=1,
        phase_velocity=out1,
        modifier=-1,
    )
    gvel, gang = gvect[:, 0], gvect[:, 1]

    # behold the triplication!!!
    plt.plot(np.sin(gang) * gvel, np.cos(gang) * gvel)
    breakpoint()

    # plt.plot(np.sin(theta) * out1, np.cos(theta) * out1)
    #
    # out2 = anray.core._get_exact_phase_velocity(theta, .195, -.22, 1, -1)
    # breakpoint()
