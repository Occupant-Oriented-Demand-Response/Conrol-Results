import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from pydantic import FilePath


class ExternalDataSchema(pa.SchemaModel):
    """The expected schema for the external data."""

    timestamp: Series[pa.Timestamp]
    electricity_price: Series[float]
    ambient_temperature: Series[float]
    solar_radiation: Series[float] = pa.Field(ge=0)


@pa.check_types(lazy=True)
def retrieve_data(path: FilePath) -> DataFrame[ExternalDataSchema]:
    """Load and validate external data from a CSV file."""
    return pd.read_csv(path, parse_dates=["timestamp"])
