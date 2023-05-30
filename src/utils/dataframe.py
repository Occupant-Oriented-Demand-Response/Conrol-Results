from pandas import DataFrame
from pydantic import FilePath


def save(dataframe: DataFrame, filepath: FilePath) -> None:
    """Save the given DataFrame to the specified file path, in either pickled or CSV format.

    Raises:
    NotImplementedError: If the file format specified by the file path is not supported.
    """
    if not filepath.exists():
        filepath.touch()

    match filepath.suffix:
        case ".pkl":
            dataframe.to_pickle(filepath)
        case ".csv":
            dataframe.to_csv(filepath)
        case _:
            raise NotImplementedError(f"Saving as {filepath.suffix}-files not implemented.")
