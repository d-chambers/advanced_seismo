"""
A module for calculating phase, group, and polarization vectors.
"""
from typing import Literal

import numpy as np

import pandas as pd

from .constants import (
    PHASE_TABLE_COLUMNS,
    INTERFACE_COLUMNS,
    MODEL_PARAMS,
    COLS_FOR_GETTING_PHASE_VELOCITY,
    COLS_FOR_GETTING_GROUP_VELOCITY,
    PHASE_OUTPUT_COLUMNS,
    GROUP_OUTPUT_COLUMNS,
    NEXT_INTERFACES_OUT,
    PROPOGATION_COL_MAP,
)


# --- Numpy low-level functions


def _get_f(vp_0, vs_0):
    """Get f from eq. 1.59 of Tsvankin, 2012."""
    vp_vs_ratio = vp_0 / vs_0
    return 1 - 1 / (vp_vs_ratio**2)


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
    term1 = 1 + epsilon * sin_theta**2 - f / 2
    term2 = (1 + (2 * (epsilon * sin_theta**2) / f)) ** 2
    term3 = (2 * (epsilon - delta) * np.sin(2 * phase_angle) ** 2) / f

    out = term1 + modifier * f / 2 * np.sqrt(term2 - term3)
    return np.sqrt(out) * vp_0


def _get_group_velocity_array(
    phase_angle, epsilon, delta, vp_0, vs_0, phase_velocity, modifier=1
):
    """Get the group velocity vector for a phase velocity (uses 1.14 and 1.15)"""

    def _get_phase_derivative(
        phase_velocity,
        epsilon,
        delta,
        vp0,
        vs0,
        modifer=1,
        small=np.pi / 10_000,
    ):
        """
        Estimate the derivative for values of theta using finite difference
        5 point stencil.
        """
        args = (epsilon, delta, vp0, vs0, modifer)
        vb2 = _get_exact_phase_velocity(phase_velocity - 2 * small, *args)
        vb1 = _get_exact_phase_velocity(phase_velocity - 1 * small, *args)
        vf1 = _get_exact_phase_velocity(phase_velocity + 1 * small, *args)
        vf2 = _get_exact_phase_velocity(phase_velocity + 2 * small, *args)
        return (
            (1 / 12) * vb2 + (-2 / 3) * vb1 + (2 / 3) * vf1 + (-1 / 12) * vf2
        ) / small

    args = (phase_angle, epsilon, delta, vp_0, vs_0, modifier)
    dv = _get_phase_derivative(*args)
    vg_x = phase_velocity * np.sin(phase_angle) + dv * np.cos(phase_angle)
    # We apply a minus sign on vg_z to conform to VTI notation where theta
    # is measured from -Z axis
    vg_z = -(phase_velocity * np.cos(phase_angle) - dv * np.sin(phase_angle))
    # calc norm and angle and return
    group_vel = np.linalg.norm(np.vstack([vg_x, vg_z]).T, axis=1)
    group_angle = np.arctan2(vg_x, -vg_z)
    # sanity checks
    assert np.all(group_vel >= phase_velocity)
    return np.vstack([group_vel, group_angle]).T


# --- setup functions


def init_phase_table(theta, interface_df):
    """Create the phase table to begin with."""

    def _init_phase_table(theta, interface_df, phase="p"):
        df = pd.DataFrame(columns=list(PHASE_TABLE_COLUMNS), index=range(len(theta)))
        df["phase_angle"] = theta
        df["phase"] = phase
        df["phase_history"] = df["phase"] + "-"
        df[["x_0", "z_0"]] = 0.0
        df["layer"] = 1
        df["parent_id"] = -1
        df["travel_time"] = 0.0
        # init material properties for starting rays.
        cols = ["epsilon", "delta", "vp_0", "vs_0"]
        df[cols] = interface_df.iloc[0][[f"{x}_bottom" for x in cols]].values
        return df

    df1 = _init_phase_table(theta, interface_df, "p")
    df2 = _init_phase_table(theta, interface_df, "s")
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    # first propagation
    return df.astype(PHASE_TABLE_COLUMNS)


def get_interface_df(model):
    """Get an interface dataframe from model."""
    interface = range(len(model) + 1)
    df = pd.DataFrame(index=interface, columns=list(INTERFACE_COLUMNS))
    df[["is_top", "is_bottom"]] = False
    df["z"] = [0] + list(np.cumsum(model["thickness"]))
    df.loc[0, "is_top"] = True
    df.loc[interface[-1], "is_bottom"] = True
    # columns for assigning data
    cols = ["vp_0", "vs_0", "epsilon", "delta"]
    bottom_cols = [f"{x}_bottom" for x in cols]
    top_cols = [f"{x}_top" for x in cols]
    for layer, (_, row) in enumerate(model.iterrows()):
        data = row[cols].values
        df.loc[layer, bottom_cols] = data
        if layer < (len(interface)):
            df.loc[layer + 1, top_cols] = data
    return df


# --- phase velocity functions


def _get_phase_array_dict(df):
    """Convert the"""
    phase_dict = {
        v: df[v].values.astype(float) for v in COLS_FOR_GETTING_PHASE_VELOCITY
    }
    phase_dict["modifier"] = df["phase"].map({"p": 1, "s": -1}).values
    assert not pd.isnull(phase_dict["modifier"]).any()
    return phase_dict


def get_phase_info_from_angles(df):
    """
    Given the phase angle and material props are known, return an array of:
    phase_angle, phase_velocity, slowness_x, slowness_z
    """
    phase_angles = df["phase_angle"]
    assert not pd.isnull(phase_angles).any(), "Missing phase angles found"
    phase_dict = _get_phase_array_dict(df)
    phase_velocity = _get_exact_phase_velocity(**phase_dict)
    slow_x = np.sin(phase_angles) / phase_velocity
    # I use -z because the convention for VTI (phase angles) is measuring
    # theta from -z axis.
    slow_z = -np.cos(phase_angles) / phase_velocity
    return np.vstack([phase_velocity, slow_x, slow_z]).T


# --- functions for getting group velocity.


def get_group_velocities(df):
    """
    Return an array of group angle and group velocity.
    """
    phase_dict = {
        v: df[v].values.astype(float) for v in COLS_FOR_GETTING_GROUP_VELOCITY
    }
    phase_dict["modifier"] = np.where(df["phase"] == "p", 1, -1)
    return _get_group_velocity_array(**phase_dict)


# ------ propagation functions


def _get_next_interfaces(df, interface_df):
    """Find the next interfaces coords, return x_1, z_1, and travel_time."""
    down_going = abs(df["group_angle"]) < np.pi / 2
    # get int to increment layer number
    direction = np.where(down_going, 1, -1)
    next_ind = np.searchsorted(interface_df["z"], df["z_0"]) + direction
    z_new = interface_df["z"].values[next_ind]
    # zero inds passed interface limits. This allows NaNs to propagate
    # on rays leaving the model.
    z_new[(next_ind >= len(interface_df)) | (next_ind < 0)] = np.NAN
    # get travel time, new x coords.
    z_diff = z_new - df["z_0"]
    # get time it takes the ray to travel
    tdiff = np.abs(z_diff / (np.cos(df["group_angle"]) * df["group_velocity"]))
    x_diff = tdiff * df["group_velocity"] * np.sin(df["group_angle"])
    travel_time = df["travel_time"] + tdiff
    x_new = df["x_0"] + x_diff
    return np.vstack([x_new, z_new, travel_time]).T


def set_columns(df1, df2, col_map):
    """Set values of df1 to values of column df2."""
    for col1, col2 in col_map.items():
        if col2 == "index":
            df1[col1] = df2.index.values
        else:
            df1[col1] = df2[col2].values
    return df1


def _get_new_df(df, interface_df):
    """Create a new dataframe from the interface df represent next rays."""

    def _init_new(df, interface_df, up=True):
        mult = 1 if up else 2
        label = "top" if up else "bottom"
        phase_mod = df["phase"] + "+" if up else df["phase"] + "-"
        new_phase_hist = df["phase_history"] + phase_mod
        layer_mod = (df["down_going"].astype(int) + (-1 if up else 1)) / 2
        new_df = (
            pd.DataFrame(index=df.index + len(df) * mult, columns=df.columns)
            .pipe(set_columns, df, PROPOGATION_COL_MAP)
            .assign(
                phase_history=new_phase_hist.values,
                down_going=not up,
                layer=(df["layer"] + layer_mod).astype(int).values,
            )
        )
        cols = [f"{x}_{label}" for x in MODEL_PARAMS]
        new_df[list(MODEL_PARAMS)] = interface_df.loc[new_df["z_0"], cols].values
        new_df = new_df[~pd.isnull(new_df["delta"])]
        # purge out any rays with nan model parameters (out of model transmission)
        return new_df

    idf = interface_df.set_index("z")
    new_up = _init_new(df, idf)
    new_down = _init_new(df, idf, up=False)
    return pd.concat([new_up, new_down], axis=0)


def prune_bad_solutions(df):
    """
    get rid of rows whose stated direction don't match the group
    velocity vector.
    """
    group_angle_down_going = np.abs(df["group_angle"]) < (np.pi / 2)
    down_going = df["down_going"]
    return df[group_angle_down_going == down_going]


def propagate(theta, model_df, iterations=4, pure_modes=True):
    """
    Propagate the un-propagated rays forward.

    This essentially fills in the phase velocity, group_velocity,
    x_1, z_1, and travel times.
    """
    assert pure_modes, "Only pure modes for now."
    # setup propagation
    slow_finder = SlownessFinder(model_df)
    interface_df = get_interface_df(model_df)
    out = []
    stack = [init_phase_table(theta, interface_df)]
    while len(stack) < iterations + 1:
        df = stack[-1]
        # given a dataframe with know phase angles, find phase velocity
        # and group velocity
        df[list(PHASE_OUTPUT_COLUMNS)] = get_phase_info_from_angles(df)
        df[list(GROUP_OUTPUT_COLUMNS)] = get_group_velocities(df)
        df = prune_bad_solutions(df)
        df[list(NEXT_INTERFACES_OUT)] = _get_next_interfaces(df, interface_df)
        out.append(df)
        # generate next iteration of dataframe
        new_df = _get_new_df(df, interface_df)
        new_df["phase_angle"] = slow_finder(new_df)
        new_df = new_df[~pd.isnull(new_df["phase_angle"])]
        stack.append(new_df)
    return out


class SlownessFinder:
    """
    Create and store slowness curves for each layer/mode.

    Then use these curves to find horizontal slowness matches.
    """

    def __init__(self, model, theta_count=2500):
        self.model = model
        eps = np.pi / 100
        max_vals = np.pi
        self._theta = np.linspace(
            -max_vals, max_vals - max_vals / theta_count, theta_count
        )
        self._x_slowness_dict, self._z_slowness_dict = self._get_slowness_dict(model)

    def _get_slowness_dict(self, df):
        x_slow = {}
        z_slow = {}
        for _, row in df.iterrows():
            for phase, mod in zip(["p", "s"], [1, -1]):
                vars = dict(row)
                vars.pop("thickness"), vars.pop("layer")
                phase_vel = _get_exact_phase_velocity(self._theta, modifier=mod, **vars)
                key = f"{int(row['layer'])}_{phase}"
                x_slow[key] = np.sin(self._theta) / phase_vel
                z_slow[key] = np.cos(self._theta) / phase_vel
        return x_slow, z_slow

    def _get_args(self, diff, down_going):
        sign = np.sign(diff)
        roll = np.roll(sign, axis=1, shift=1)
        # roll[:, 0] = sign[:, 0]  # don't induce wrap around diff
        sign_change = sign - roll
        # there should be exactly two solutions; when it becomes
        # multi-valued my code fails.
        assert np.all(np.abs(sign_change).sum(axis=1) <= 4)
        # ensure there is one down-going and one up going
        sign_mins = np.min(sign_change, axis=1, keepdims=True)
        sign_maxs = np.max(sign_change, axis=1, keepdims=True)
        assert np.all(sign_mins == -2) and np.all(sign_maxs == 2)
        # assert np.all(np.any(sign_change == sign_mins, axis=1))
        # assert np.all(np.any(sign_change == sign_maxs, axis=1))
        argmin = np.argmin(sign_change, axis=1)
        argmax = np.argmax(sign_change, axis=1)
        out = np.where(down_going, argmin, argmax)
        # no args should be on the edge of the array.
        # assert np.all((out > 0) & (out < (len(self._theta) - 1)))
        return out

    def extrapolate_theta(self, args, diffs):
        """Extrapolate values of theta"""
        inds = np.arange(len(diffs))
        args_p = args - 1
        args_a = args + 1
        theta_p, theta_a = self._theta[args_p], self._theta[args_a]
        diff_p, diff_a = diffs[inds, args_p], diffs[inds, args_a]
        # use y = ax + b; x is theta, y is diff
        slope = (diff_a - diff_p) / (theta_a - theta_p)
        y_intercept = diff_p - slope * theta_p
        theta_out = -y_intercept / slope
        bad_inds = ~((theta_out > theta_p) & (theta_out < theta_a))
        # for bad indicies just use closest value
        theta_out[bad_inds] = np.NaN  # failed on these, just toss
        # assert not np.any(bad_inds)
        return theta_out

    def __call__(self, df):
        """
        Given a dataframe with horizontal slowness, calculate phase and
        group velocities.
        """
        keys = df["layer"].astype(str) + "_" + df["phase"]
        slowness_x = df["slowness_x"].values[:, None]
        slow_arrays = np.array([self._x_slowness_dict[x] for x in keys])
        # figure out where horizontal slowness values are equal.
        diffs = slowness_x - slow_arrays
        args = self._get_args(diffs, df["down_going"].values)
        thetas = self.extrapolate_theta(args, diffs)
        return thetas
