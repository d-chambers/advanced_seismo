"""Calculate the arrival times for shot gathers for model 1."""
import numpy as np

import anray

import pandas as pd

import local


if __name__ == "__main__":

    eps = np.deg2rad(1)
    theta = np.linspace(eps, np.pi/2-eps, 200)
    model = pd.read_csv(local.model_1_path)

    # note all rays are at 0.5 on final iteration for some reason

    breakpoint()
    out = anray.propagate(theta, model)




    pass
