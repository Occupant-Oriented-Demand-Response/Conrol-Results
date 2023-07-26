from typing import Callable

from casadi import logic_and, logic_or
from do_mpc.model import Model

from src.core.modelparameters import ModelParameters
from src.core.zones import Zones


def create_model(model_parameters: Callable[[Zones], ModelParameters]) -> Model:
    """Create a do_mpc.model.Model instance based on the given model parameters."""
    model = Model(model_type="continuous", symvar_type="SX")

    # time-varying parameters / disturbances
    electricity_price = model.set_variable("_tvp", "electricity_price")
    ambient_temperature = model.set_variable("_tvp", "ambient_temperature")
    solar_radiation = model.set_variable("_tvp", "solar_radiation")
    heat_pump_cop = model.set_variable("_tvp", "heat_pump_cop")
    heat_pump_power_max = model.set_variable("_tvp", "heat_pump_power_max")

    # zones
    temperatures = {}
    heat_flows = {}
    bounds = {}

    for zone in Zones:
        # states
        temperatures[zone] = {
            "air": model.set_variable("_x", f"temperature_{zone}_air"),
            "internal": model.set_variable("_x", f"temperature_{zone}_internal"),
        }

        # inputs
        heat_flows[zone] = model.set_variable("_u", f"heat_flow_{zone}")

        # ordinary differential equations
        # fmt: off
        model.set_rhs(
            f"temperature_{zone}_air",
            ((temperatures[zone]["internal"] - temperatures[zone]["air"]) / model_parameters(zone).resistance_inside
                + (ambient_temperature - temperatures[zone]["air"]) / model_parameters(zone).resistance_window
                + model_parameters(zone).coefficient_solar * solar_radiation
                + heat_flows[zone]
            ) / model_parameters(zone).capacity_air
        )

        model.set_rhs(
            f"temperature_{zone}_internal",
            ((temperatures[zone]["air"] - temperatures[zone]["internal"]) / model_parameters(zone).resistance_inside
            ) / model_parameters(zone).capacity_internal
        )
        # fmt: on

        # time-varying parameters / constraints
        bounds[zone] = {
            "min": model.set_variable("_tvp", f"temperature_{zone}_min"),
            "max": model.set_variable("_tvp", f"temperature_{zone}_max"),
        }

        model.set_expression(f"temperature_{zone}_lower_bound", -temperatures[zone]["air"] + bounds[zone]["min"])
        model.set_expression(f"temperature_{zone}_upper_bound", temperatures[zone]["air"] - bounds[zone]["max"])

    # other equations
    heat_flow_total = model.set_expression("heat_flow_total", sum(heat_flows.values()))
    heat_pump_power = model.set_expression("heat_pump_power", heat_flow_total / heat_pump_cop)
    heat_pump_modulation = model.set_expression("heat_pump_modulation", heat_pump_power / heat_pump_power_max)
    model.set_expression("costs", electricity_price * heat_pump_power)
    model.set_expression(
        "heat_pump_modulation_constraint",
        logic_or(logic_and(heat_pump_modulation >= 0.2, heat_pump_modulation <= 1), heat_pump_modulation == 0),
    )

    model.setup()
    return model
