from dataclasses import dataclass
from enum import StrEnum, auto


class ModelParametersType(StrEnum):
    """A enum class representing the different types of ModelParameters instances."""

    LOW = auto()
    HIGH = auto()


@dataclass
class ModelParameters:
    """A data class representing the parameters of a (zone's) model."""

    capacity_air: float
    capacity_internal: float
    resistance_inside: float
    resistance_window: float
    coefficient_solar: float
