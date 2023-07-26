from dataclasses import dataclass, field
from typing import Callable, Self

from src.utils.interpolate import InterpolationType, interpolate1D
from src.utils.json import JSONConfig


@dataclass
class HeatPumpModel:
    """
    A data class representing a heat pump model.
    """

    name: str
    config: JSONConfig
    __maximum_electrical_power: Callable[[float], float] = field(repr=False)
    __coefficient_of_performance: Callable[[float], float] = field(repr=False)

    def __init__(self: Self, config: JSONConfig) -> None:
        """Initialize a HeatPumpModel instance."""
        self.config = config
        self.initialize()

    def initialize(self: Self) -> None:
        """Load the config data and initialize the HeatPumpModel instance with it."""
        data = self.config.load()

        self.name = data["name"]
        self.__maximum_electrical_power = interpolate1D(
            list(zip(data["outside_air_temperature"], data["electrical_power_consumption_max"])),
            InterpolationType.CUBIC,
        )
        self.__coefficient_of_performance = interpolate1D(
            list(zip(data["outside_air_temperature"], data["coefficient_of_performance_nenn"])),
            InterpolationType.CUBIC,
        )

    def maximum_electrical_power(self, ambient_temperature: float) -> float:
        """Get the maximum electrical power consumption of the heat pump model at a given ambient temperature."""
        return self.__maximum_electrical_power(ambient_temperature)

    def coefficient_of_performance(self, ambient_temperature: float) -> float:
        """Get the coefficient of performance of the heat pump model at a given ambient temperature."""
        return self.__coefficient_of_performance(ambient_temperature)
