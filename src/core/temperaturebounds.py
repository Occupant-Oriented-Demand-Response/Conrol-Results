from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto
from typing import Self

from pydantic import conlist


class TemperatureBoundsType(StrEnum):
    """A enum class representing the different types of TemperatureBounds instances."""

    BASE = auto()
    ADAPTIVE = auto()


@dataclass
class TemperatureBounds:
    """A data class representing the temperature bounds of a TemperatureBoundsType for all zones."""

    hourly_minimum_temperatures: conlist(float, min_items=24, max_items=24)
    hourly_maximum_temperatures: conlist(float, min_items=24, max_items=24)

    def minimum_temperature(self: Self, timestamp: datetime) -> float:
        """Returns the minimum temperature bound for the given timestamp and zone."""
        return self.hourly_minimum_temperatures[timestamp.hour]

    def maximum_temperature(self: Self, timestamp: datetime) -> float:
        """Returns the maximum temperature bound for the given timestamp and zone."""
        return self.hourly_maximum_temperatures[timestamp.hour]
