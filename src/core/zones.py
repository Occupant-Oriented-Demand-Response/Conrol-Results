from enum import StrEnum, auto
from typing import Self

from paper_revision.core.modelparameters import ModelParameters, ModelParametersType
from paper_revision.core.temperaturebounds import TemperatureBounds, TemperatureBoundsType


class Zones(StrEnum):
    """A enum class representing the different zones."""

    ROOM_1 = auto()
    ROOM_2 = auto()
    ROOM_3 = auto()
    ROOM_4 = auto()
    ROOM_5 = auto()


class ZoneDataProvider:
    """A class for providing model parameters and temperature bounds for different zones."""

    __model_parameters_data: dict[ModelParametersType, dict[Zones, ModelParameters]]
    __temperature_bounds_data: dict[TemperatureBoundsType, dict[Zones, TemperatureBounds]]

    def __init__(
        self: Self,
        model_parameters_data: dict[ModelParametersType, dict[Zones, ModelParameters]],
        temperature_bounds_data: dict[TemperatureBoundsType, dict[Zones, TemperatureBounds]],
    ) -> None:
        """Initialize a ZoneDataProvider instance with the given model parameters and temperature bounds data."""
        self.__model_parameters_data = model_parameters_data
        self.__temperature_bounds_data = temperature_bounds_data

    def model_parameters(self: Self, zone: Zones, parameters_type: ModelParametersType) -> ModelParameters:
        """Returns a ModelParameters instance for the given zone and parameters type."""
        return ModelParameters(**self.__model_parameters_data[parameters_type][zone])

    def temperature_bounds(self: Self, zone: Zones, bounds_type: TemperatureBoundsType) -> TemperatureBounds:
        """Returns a TemperatureBounds instance for the given zone and bounds type."""
        return TemperatureBounds(**self.__temperature_bounds_data[bounds_type][zone])
