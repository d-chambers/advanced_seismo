"""
A module for calculating phase, group, and polarization vectors.
"""
from typing import Literal

import numpy as np
import jax


import pandas as pd

PHASE_TABLE_COLUMNS = {
    "id": float,
    "phase": str,
    "x_0": float,
    "z_0": float,
    "phase_angle": float,
    "phase_velocity": float,
    "group_angle": float,
    "group_velocity": float,
    "x_1": float,
    "z_1": float,
    "travel_time": float,
    "parent_id": float,
    "epsilon": float,
    "delta": float,
    "vp_0": float,
    "vs_0": float,
    "propagated": bool,
}

INTERFACE_COLUMNS = (
    "vp_0_top",
    "vp_0_bottom",
    "vs_0_top",
    "vs_0_bottom",
    "epsilon_top",
    "epsilon_bottom",
    "delta_top",
    "delta_bottom",
    "z",
    "is_top",
    "is_bottom",
)

# columns needed for calculating phase velocity.
PHASE_VELOCITY_COLUMNS = (
    'phase_angle',
    "vp_0",
    "vs_0",
    "delta",
    "epsilon",
)


def get_interface_df(model):
    """Get an interface dataframe from model."""
    interface = range(len(model) + 1)
    df = pd.DataFrame(index=interface, columns=list(INTERFACE_COLUMNS))
    df[['is_top', 'is_bottom']] = False
    df['z'] = [0] + list(np.cumsum(model['thickness']))
    df.loc[0, 'is_top'] = True
    df.loc[interface[-1], 'is_bottom'] = True
    # columns for assigning data
    cols = ['vp_0', "vs_0", "epsilon", "delta"]
    bottom_cols = [f"{x}_bottom" for x in cols]
    top_cols = [f"{x}_top" for x in cols]
    for layer, (_, row) in enumerate(model.iterrows()):
        data = row[cols].values
        df.loc[layer, bottom_cols] = data
        if layer < (len(interface)):
            df.loc[layer + 1, top_cols] = data
    return df


# --- functions for getting exact phase velocity

def _get_f(vp_0, vs_0):
    """Get f from eq. 1.59 of Tsvankin, 2012."""
    vp_vs_ratio = vp_0 / vs_0
    return 1 - 1 / (vp_vs_ratio ** 2)


def _get_exact_phase_velocity(
        phase_angle,
        epsilon,
        delta,
        vp_0,
        vs_0,
        modifier: Literal[1, -1] = 1,
):
    """
    Calculate the exact phase velocity in terms of Vp_0.

    Parameters
    ----------
    phase_angle
        A vector of theta values to compute.
    epsilon
        The epsilon thomsen parameter.
    delta
        The delta thomsen parameter.
    vp_0
        The vertical velocity of P waves.
    vs_0
        The vertical velocity of S waves.
    """
    f = _get_f(vp_0, vs_0)

    sin_theta = np.sin(phase_angle)
    term1 = 1 + epsilon * sin_theta ** 2 - f / 2
    term2 = (1 + (2 * (epsilon * sin_theta ** 2) / f)) ** 2
    term3 = (2 * (epsilon - delta) * np.sin(2 * phase_angle) ** 2) / f

    out = term1 + modifier * f / 2 * np.sqrt(term2 - term3)
    return np.sqrt(out) * vp_0


# --- functions for estimating derivatives of phase velocity


def _get_phase_derivative(
        phase_velocity,
        epsilon,
        delta,
        vp0,
        vs0,
        modifer=1,
        small=np.pi / 10_000,
):
    """Estimate the derivative for values of theta using finite differences 5 p"""
    args = (epsilon, delta, vp0, vs0, modifer)
    vb2 = _get_exact_phase_velocity(phase_velocity - 2 * small, *args)
    vb1 = _get_exact_phase_velocity(phase_velocity - 2 * small, *args)
    vf1 = _get_exact_phase_velocity(phase_velocity - 2 * small, *args)
    vf2 = _get_exact_phase_velocity(phase_velocity - 2 * small, *args)
    return ((1/12) * vb2 + (-2/3) * vb1 + (2/3) * vf1 + (-1/12) * vf2) / small


# --- functions for getting group velocity.

def _get_group(phase_angle, epsilon, delta, vp_0, vs_0, modifier=1, phase_velocity=None):
    """Get the group velocity vector for a phase velocity (uses 1.14 and 1.15) """
    args = (phase_angle, epsilon, delta, vp_0, vs_0, modifier)
    dv = _get_phase_derivative(*args)
    if phase_velocity is None:
        phase_velocity = _get_exact_phase_velocity(*args)
    vg_x = phase_velocity * np.sin(phase_velocity) + dv * np.cos(phase_velocity)
    vg_z = phase_velocity * np.cos(phase_velocity) - dv * np.sin(phase_velocity)
    out = np.vstack([vg_x, vg_z]).T
    group_vel = np.linalg.norm(out, axis=1)
    group_angle = np.arctan(out[:, 0]/out[:, 1])
    breakpoint()
    return


def get_group_phases(phase_angle, epsilon, delta, vp_0, vs_0, modifier=1):
    """
    Return the phase velocity, group angle, and group velocity.

    Modifier=1 for P and -1 for SV.
    """
    args = (phase_angle, epsilon, delta, vp_0, vs_0, modifier)
    phase_vel = _get_exact_phase_velocity(*args)
    group_vel = _get_group(*args, phase_velocity=phase_vel)
    breakpoint()



# ------ propagation functions

def init_phase_table(theta, interface_df):
    """Create the phase table to begin with."""
    df = pd.DataFrame(columns=list(PHASE_TABLE_COLUMNS), index=range(len(theta)))
    df['phase_angle'] = theta
    df['phase'] = 'P'
    df[['x_0', 'z_0']] = 0.0
    df['propagated'] = False
    df['id'] = range(len(df))
    # init material properties for starting rays.
    cols = ['epsilon', 'delta', 'vp_0', 'vs_0']
    df[cols] = interface_df.iloc[0][[f"{x}_bottom" for x in cols]].values
    return df.astype(PHASE_TABLE_COLUMNS)


def propagate(table, model):
    """
    Propagate the un-propagated rays forward.

    This essentially fills in the phase velocity, group_velocity,
    x_1, z_1, and travel times.
    """
    df = table[~table['propagated']]  # rows that need to be propagated.
    is_p, is_s = df['phase'] == 'P', df['phase'] == 'S'
    group_cols = ["phase_velocity", "group_angle", "group_velocity"]
    # first get phase velocity
    phase_dict = {v: df[v].values.astype(float) for v in PHASE_VELOCITY_COLUMNS}
    df.loc[is_p, group_cols] = get_group_phases(**phase_dict)
    df.loc[is_s, group_cols] = get_group_phases(modifier=-1, **phase_dict)

    breakpoint()
    print(df)


def find_next_point(current_point, phase_angle, phase_velocity, model):
    """
    Given the current point and phase velocity, find the next point.
    """
    pass
