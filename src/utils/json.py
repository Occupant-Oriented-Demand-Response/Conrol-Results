import json
from dataclasses import dataclass
from typing import Self

import jsonschema
from pydantic import FilePath


@dataclass
class JSONConfig:
    """A dataclass representing a JSON configuration and its corresponding JSON schema."""

    json_config_path: FilePath
    json_schema_path: FilePath

    def load(self: Self) -> dict:
        """Load data from the JSON file path and validate it against the JSON schema."""

        with open(file=self.json_config_path, mode="r", encoding="utf8") as file:
            data = json.load(file)

        with open(file=self.json_schema_path, mode="r", encoding="utf8") as file:
            jsonschema.validate(data, json.load(file))

        return data
