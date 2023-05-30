import pickle
from typing import Any

from pydantic import FilePath


def save(data: Any, filepath: FilePath) -> None:
    """Pickle the given data and write it to the specified file path."""
    if not filepath.exists():
        filepath.touch()

    with open(filepath, "wb") as file:
        pickle.dump(data, file)


def load(filepath: FilePath) -> Any:
    """Load pickled data from the specified file path and return it."""
    with open(filepath, "rb") as file:
        return pickle.load(file)
