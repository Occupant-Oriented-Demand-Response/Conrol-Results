from typing import Any

import pandas as pd
from casadi import DM
from do_mpc.controller import MPC
from do_mpc.model import Model
from pandera.typing import DataFrame
from pydantic import PositiveInt

from paper_revision.core.dataprovider import DataSchema
from paper_revision.core.zones import Zones


def create_mpc_controller(model: Model, data: DataFrame[DataSchema]) -> MPC:
    """Create a do_mpc.controller.MPC instance based on the given model and data."""
    mpc_controller = MPC(model)

    mpc_setup = {
        "n_horizon": 64,
        "n_robust": 0,
        "open_loop": 0,
        "t_step": (data["timestamp"][1] - data["timestamp"][0]).seconds,
        "state_discretization": "collocation",
        "collocation_type": "radau",
        "collocation_deg": 2,
        "collocation_ni": 2,
        "store_full_solution": True,
        "nlpsol_opts": {"ipopt.print_level": 0, "print_time": 0}  # Deaktivate ipopt outputs
        # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'} # Use MA27 linear solver in ipopt for faster calculations
    }

    mpc_controller.set_param(**mpc_setup)

    # objective function
    mpc_controller.set_objective(mterm=DM(0), lterm=model.aux["costs"])

    # constraints
    mpc_controller.set_nl_cons("heat_pump_power_lower_bound", expr=-model.aux["heat_pump_power"], ub=0)
    mpc_controller.set_nl_cons(
        "heat_pump_power_upper_bound", expr=model.aux["heat_pump_power"] - model.tvp["heat_pump_power_max"], ub=0
    )
    mpc_controller.set_nl_cons("heat_pump_modulation_lower_bound", expr=-model.aux["heat_pump_modulation"], ub=0)
    mpc_controller.set_nl_cons("heat_pump_modulation_upper_bound", expr=model.aux["heat_pump_modulation"], ub=1)
    # mpc_controller.set_nl_cons("heat_pump_modulation_constraint",
    # expr=-model.aux["heat_pump_modulation_constraint"], ub=-1)

    for zone in Zones:
        mpc_controller.bounds["lower", "_u", f"heat_flow_{zone}"] = 0
        mpc_controller.set_nl_cons(
            f"temperature_{zone}_lower_bound",
            expr=model.aux[f"temperature_{zone}_lower_bound"],
            ub=0,
            soft_constraint=True,
            penalty_term_cons=1e6,
        )
        mpc_controller.set_nl_cons(
            f"temperature_{zone}_upper_bound",
            expr=model.aux[f"temperature_{zone}_upper_bound"],
            ub=0,
            soft_constraint=True,
            penalty_term_cons=1e6,
        )

    # time-varying parameters
    tvp_template = mpc_controller.get_tvp_template()

    def tvp_fun(time_now: PositiveInt) -> Any:
        index = data.index[data["timestamp"] == data["timestamp"][0] + pd.Timedelta(seconds=int(time_now))]
        for k in range(mpc_controller.n_horizon + 1):  # pylint: disable=E1101 (no-member)
            for parameter in model.tvp.keys():
                if not parameter == "default":
                    tvp_template["_tvp", k, parameter] = data[parameter][index + k]
        return tvp_template

    mpc_controller.set_tvp_fun(tvp_fun)

    mpc_controller.setup()

    return mpc_controller
