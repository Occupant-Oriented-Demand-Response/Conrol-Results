import copy
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from do_mpc.data import save_results

from paper_revision.core.dataprovider import DataProvider
from paper_revision.core.externaldata import retrieve_data
from paper_revision.core.heatpump import HeatPumpModel
from paper_revision.core.model import create_model
from paper_revision.core.modelparameters import ModelParametersType
from paper_revision.core.mpc_controller import create_mpc_controller
from paper_revision.core.simulator import create_simulator
from paper_revision.core.temperaturebounds import TemperatureBoundsType
from paper_revision.core.zones import ZoneDataProvider, Zones
from paper_revision.utils.flat_list import flat_list
from paper_revision.utils.json import JSONConfig

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def main() -> None:
    """
    Runs the MPC simulation for each week of the provided external data.

    The function loads and generates the necessary data and configuration files, creates the model and simulator,
    and sets up the MPC controller. Then, for each week of the external data, it runs the main loop of the simulation,
    saves the results, and moves on to the next week.
    """

    ###############################################################################################
    ###############               define scenario here                              ###############
    ###############################################################################################

    result_dir = DATA_DIR / "results" / "mpc" / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    temperature_bounds_type = TemperatureBoundsType.BASE
    model_parameters_type = ModelParametersType.LOW
    initial_states = np.array([23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0])

    ###############################################################################################
    ###############################################################################################

    # load / generate data
    external_data = retrieve_data(DATA_DIR / "external" / "timeseries_data.csv")

    heat_pump_config = JSONConfig(
        DATA_DIR / "config" / "heat_pump_config.json", DATA_DIR / "config" / "heat_pump_schema.json"
    )
    heat_pump_model = HeatPumpModel(config=heat_pump_config)

    model_parameters_config = JSONConfig(
        DATA_DIR / "config" / "model_parameters_config.json", DATA_DIR / "config" / "model_parameters_schema.json"
    )
    temperature_bounds_config = JSONConfig(
        DATA_DIR / "config" / "temperature_bounds_config.json", DATA_DIR / "config" / "temperature_bounds_schema.json"
    )

    zone_data_provider = ZoneDataProvider(
        model_parameters_data=model_parameters_config.load(), temperature_bounds_data=temperature_bounds_config.load()
    )

    data_provider = DataProvider()
    data_provider.generate_dataset(
        data=external_data,
        heat_pump_model=heat_pump_model,
        temperature_bounds=partial(zone_data_provider.temperature_bounds, bounds_type=temperature_bounds_type),
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    data_provider.save_dataset(result_dir / "dataset.csv")

    # generate model
    model = create_model(
        model_parameters=partial(zone_data_provider.model_parameters, parameters_type=model_parameters_type)
    )

    # run each week
    for week in range(len(data_provider.weekly_datasets) - 2):
        # get data for that week
        data = data_provider.get_data(week)
        mpc_data = pd.concat([data, data_provider.get_data(week + 1)]).reset_index(drop=True)

        # get initial states
        current_states = copy.deepcopy(initial_states)

        # generate simulator
        simulator = create_simulator(model, data)
        simulator.x0 = current_states

        # setup mpc controller
        mpc_controller = create_mpc_controller(model, mpc_data)
        mpc_controller.x0 = current_states
        mpc_controller.set_initial_guess()

        # run main loop
        for index in range(len(data.index)):
            inputs = mpc_controller.make_step(current_states)
            current_states = simulator.make_step(inputs)

        # save results
        save_results(
            save_list=[mpc_controller, simulator],
            result_name=f"{temperature_bounds_type}_{model_parameters_type}_week_{week}",
            result_path=str(result_dir) + "/",
        )

        results = data
        for zone in Zones:
            results[f"temperature_{zone}_air"] = flat_list(simulator.data["_x", f"temperature_{zone}_air"])
            results[f"temperature_{zone}_internal"] = flat_list(simulator.data["_x", f"temperature_{zone}_internal"])
            results[f"discomfort_{zone}"] = flat_list(
                [
                    max(
                        [0],
                        simulator.data["_aux", f"temperature_{zone}_lower_bound"][index],
                        simulator.data["_aux", f"temperature_{zone}_upper_bound"][index],
                    )
                    for index in range(len(data.index))
                ]
            )
            results[f"heat_flow_{zone}"] = flat_list(simulator.data["_u", f"heat_flow_{zone}"])
        results["heat_flow_total"] = flat_list(simulator.data["_aux", "heat_flow_total"])
        results["heat_pump_power"] = flat_list(simulator.data["_aux", "heat_pump_power"])
        results["discomfort_total"] = results[[f"discomfort_{zone}" for zone in Zones]].sum(axis=1)
        results["costs"] = flat_list(simulator.data["_aux", "costs"])
        results.to_csv(result_dir / f"{temperature_bounds_type}_{model_parameters_type}_week_{week}.csv", index=False)


if __name__ == "__main__":
    main()
