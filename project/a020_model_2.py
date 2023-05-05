"""Calculate the arrival times for shot gathers for model 1."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anray
import local


if __name__ == "__main__":
    # get starting phase angles
    eps = np.deg2rad(1)
    theta = np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, 200)
    # load model
    model = pd.read_csv(local.model_2_path)
    # shoot rays
    out = anray.propagate(theta, model)
    # plot
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    anray.plot_phase(out[-1], "p-p-p+p+", ax=axes[0], color='blue')
    anray.plot_phase(out[-1], "s-s-s+s+", ax=axes[1], color='red')
    plt.tight_layout()
    fig.savefig(local.travel_curve_model_2)

    # get slowness surfaces
    slow = anray.core.SlownessFinder(model)
    fig, *_ = anray.plot_slowness_surfaces(slow)
    plt.tight_layout()
    fig.savefig(local.slowness_surface_2)

    # plot rays
    fig, *_ = anray.plot_rays(
        out, model, phase_dict={"p-p-p+p+": "blue", "s-s-s+s+": "red"}
    )
    fig.savefig(local.ray_path_2)