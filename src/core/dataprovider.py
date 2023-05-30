from typing import Callable

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from pydantic import FilePath, PositiveInt

from paper_revision.core.externaldata import ExternalDataSchema
from paper_revision.core.heatpump import HeatPumpModel
from paper_revision.core.temperaturebounds import TemperatureBounds
from paper_revision.core.zones import Zones


class DataSchema(pa.SchemaModel):
    """The expected schema for the timeseries data."""

    timestamp: Series[pa.Timestamp]
    electricity_price: Series[float]
    ambient_temperature: Series[float]
    solar_radiation: Series[float] = pa.Field(ge=0)
    heat_pump_cop: Series[float]
    heat_pump_power_max: Series[float] = pa.Field(ge=0)
    temperature_room_1_min: Series[float]
    temperature_room_1_max: Series[float]
    temperature_room_2_min: Series[float]
    temperature_room_2_max: Series[float]
    temperature_room_3_min: Series[float]
    temperature_room_3_max: Series[float]
    temperature_room_4_min: Series[float]
    temperature_room_4_max: Series[float]
    temperature_room_5_min: Series[float]
    temperature_room_5_max: Series[float]


class DataProvider:
    """A data provider that loads and validates data from a CSV file."""

    dataset: DataFrame[DataSchema]
    weekly_datasets: list[DataFrame[DataSchema]]

    def generate_dataset(
        self,
        data: DataFrame[ExternalDataSchema],
        heat_pump_model: HeatPumpModel,
        temperature_bounds: Callable[[Zones], TemperatureBounds],
    ) -> None:
        """Generate a new dataset."""
        # generate heat pump data based on outside air temperature
        data["heat_pump_cop"] = [
            heat_pump_model.coefficient_of_performance(ambient_temperature)
            for ambient_temperature in data["ambient_temperature"]
        ]
        data["heat_pump_power_max"] = [
            heat_pump_model.maximum_electrical_power(ambient_temperature)
            for ambient_temperature in data["ambient_temperature"]
        ]

        # generate temperature bounds schedule for each zone/room based on timestamp
        for zone in Zones:
            data[f"temperature_{zone}_min"] = [
                temperature_bounds(zone).minimum_temperature(timestamp) for timestamp in data["timestamp"]
            ]
            data[f"temperature_{zone}_max"] = [
                temperature_bounds(zone).maximum_temperature(timestamp) for timestamp in data["timestamp"]
            ]

        try:
            DataSchema.validate(data, lazy=True)
        except pa.errors.SchemaErrors as err:
            print(err)

        self.dataset = data
        self.weekly_datasets = [
            dataframe.reset_index(drop=True)
            for _, dataframe in self.dataset.groupby(pd.to_datetime(self.dataset["timestamp"]).dt.to_period("W"))
        ]

    def load_dataset(self, data_path: FilePath) -> None:
        """Load an existing dataset from a .csv data file."""
        self.dataset = self._retrieve_data(data_path)

    def save_dataset(self, data_path: FilePath) -> None:
        """Save the dataset to a .csv data file."""
        self.dataset.to_csv(data_path, index=False)

    def get_data(self, week: PositiveInt = 0) -> DataFrame[DataSchema]:
        """Get a subset of the loaded dataset corresponding to the specified week."""
        return self.weekly_datasets[week]

    @pa.check_types(lazy=True)
    def _retrieve_data(self, path: FilePath) -> DataFrame[DataSchema]:
        """Load and validate data from a CSV file."""
        return pd.read_csv(path, parse_dates=["timestamp"])
