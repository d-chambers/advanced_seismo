PHASE_TABLE_COLUMNS = {
    "phase": str,
    "x_0": float,
    "z_0": float,
    "phase_angle": float,
    "phase_velocity": float,
    "slowness_x": float,
    "slowness_z": float,
    "group_angle": float,
    "group_velocity": float,
    "x_1": float,
    "z_1": float,
    "layer": int,
    "travel_time": float,
    "parent_id": int,
    "epsilon": float,
    "delta": float,
    "vp_0": float,
    "vs_0": float,
    "down_going": bool,
    "phase_history": str,
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
MODEL_PARAMS = (
    "vp_0",
    "vs_0",
    "delta",
    "epsilon",
)
COLS_FOR_GETTING_PHASE_VELOCITY = (
    'phase_angle',
    *MODEL_PARAMS
)
COLS_FOR_GETTING_GROUP_VELOCITY = (
    "phase_velocity",
    *COLS_FOR_GETTING_PHASE_VELOCITY
)
PHASE_OUTPUT_COLUMNS = (
    "phase_velocity",
    "slowness_x",
    "slowness_z",
)
GROUP_OUTPUT_COLUMNS = (
    "group_velocity",
    "group_angle",
)
NEXT_INTERFACES_OUT = (
    "x_1",
    'z_1',
    'travel_time',
)
PHASE_MODIFIER_DICT = {'p': 1, 'sv': -2}
PROPOGATION_COL_MAP = {
    "parent_id": "index",
    "x_0": "x_1",
    "z_0": "z_1",
    "phase": "phase",
    "travel_time": "travel_time",
    "slowness_x": "slowness_x",
}
