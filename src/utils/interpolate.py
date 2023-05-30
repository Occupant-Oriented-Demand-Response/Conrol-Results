from enum import StrEnum, auto
from typing import Callable

from numpy import interp
from scipy.interpolate import CubicSpline


class InterpolationType(StrEnum):
    """An enumeration of the supported interpolation types for the `interpolate1D` function."""

    LINEAR = auto()
    CUBIC = auto()


def interpolate1D(
    points: list[tuple[float, float]], interpolation_type: InterpolationType = InterpolationType.LINEAR
) -> Callable[[float], float]:
    """Interpolate a 1D function of the specified interpolation type from the given data points."""

    x_values = [i for i, j in points]
    y_values = [j for i, j in points]

    match interpolation_type:
        case InterpolationType.LINEAR:

            def func(x: float) -> float:
                return float(interp(x, x_values, y_values))

        case InterpolationType.CUBIC:

            def func(x: float) -> float:  # pylint: disable=E0102 (function-redefined)
                return float(CubicSpline(x_values, y_values)(x))

    return func
