"""Calculate the arrival times for shot gathers for model 1."""
import numpy as np

import anray

import pandas as pd

import local


if __name__ == "__main__":
    theta = np.linspace(0, np.pi/4, 1000)
    model = pd.read_csv(local.model_1_path)
    interface_df = anray.get_interface_df(model)

    # initialize and propagate array.
    df = (
        anray.init_phase_table(theta, interface_df)
        .pipe(anray.propagate, interface_df)
    )
    breakpoint()



    pass
