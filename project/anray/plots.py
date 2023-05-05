"""
Plotting utils
"""
import matplotlib.pyplot as plt
import pandas as pd

from .core import get_interface_df


def plot_phase(df, phase, ax=None, color=None, max_x=2.0):
    """
    Plot a specific phase's x vs arrival time.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    sub = df[(df["phase_history"] == phase) & (abs(df["x_1"]) < max_x)]
    ax.plot(sub["x_1"], sub["travel_time"], "-", color=color, label=phase)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title(phase)
    return ax


def plot_slowness_surfaces(slowfinder):
    """Plot the slowness surfaces."""
    color_dict = {"p": "blue", "s": "red"}
    keys = list(slowfinder._x_slowness_dict)
    layers = sorted({x.split("_")[0] for x in keys})
    fig, axes = plt.subplots(
        1, len(layers), figsize=(3 * len(layers), 3), sharey=True, sharex=True
    )
    ax_dict = {layer: ax for layer, ax in zip(layers, axes)}
    for key in keys:
        layer, phase = key.split("_")
        ax = ax_dict[layer]
        color = color_dict[phase]
        x_slow = slowfinder._x_slowness_dict[key]
        z_slow = slowfinder._z_slowness_dict[key]
        ax.plot(x_slow, z_slow, color=color, label=phase)
        ax.set_title(f"Layer: {layer}")
        ax.set_xlabel("Horizontal slowness (s/km)")
        ax.set_ylabel("Vertical slowness (s/km)")
    for ax in axes:
        ax.legend()

    return fig, axes


def plot_rays(df_list, model, phase_dict, max_x=2):
    """Plot the rays associated with a specific phase."""
    interfaces = get_interface_df(model)
    master_df = pd.concat(df_list, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for (phase, color), ax in zip(phase_dict.items(), axes):
        df = master_df[master_df["phase_history"] == phase]
        df = df[abs(df["x_1"]) < max_x]
        while len(df):
            paths = df[["x_0", "x_1", "z_0", "z_1"]].values
            for row in paths:
                ax.plot(row[:2].T, row[2:].T, "-", color=color, alpha=0.2)
            df = master_df[master_df.index.isin(df["parent_id"])]
        # plot the interface boundaries
        for ind, row in interfaces.iterrows():
            ax.axhline(row["z"], color='k')
        ax.plot(0, 0, "*", color="gray")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Z (km)")
        ax.set_title(phase[0].upper())
        ax.invert_yaxis()
    return fig, axes
