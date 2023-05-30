from typing import Any

import pandas as pd
from do_mpc.model import Model
from do_mpc.simulator import Simulator
from pandera.typing import DataFrame
from pydantic import PositiveInt

from paper_revision.core.dataprovider import DataSchema


def create_simulator(model: Model, data: DataFrame[DataSchema]) -> Simulator:
    """Create a do_mpc.simulator.Simulator instance based on the given model and data."""
    simulator = Simulator(model)

    simulator_setup = {
        "integration_tool": "cvodes",
        "abstol": 1e-10,
        "reltol": 1e-10,
        "t_step": (data["timestamp"][1] - data["timestamp"][0]).seconds,
    }

    simulator.set_param(**simulator_setup)

    # time-varying parameters
    tvp_template = simulator.get_tvp_template()

    def tvp_fun(time_now: PositiveInt) -> Any:
        index = data.index[data["timestamp"] == data["timestamp"][0] + pd.Timedelta(seconds=int(time_now))]

        for parameter in model.tvp.keys():
            if not parameter == "default":
                tvp_template[parameter] = data[parameter][index]
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator
