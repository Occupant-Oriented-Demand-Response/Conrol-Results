from enum import StrEnum, auto
from typing import Callable

from scipy.optimize import curve_fit


class FitFunctionType(StrEnum):
    """An enumeration of the supported fit function types for the `fit1D` function."""

    LINEAR = auto()
    QUADRATIC = auto()
    CUBIC = auto()


FUNCTION_FACTORY = {
    FitFunctionType.LINEAR: lambda x, a, b: a * x + b,
    FitFunctionType.QUADRATIC: lambda x, a, b, c: a * x**2 + b * x + c,
    FitFunctionType.CUBIC: lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
}


def fit1D(
    points: list[tuple[float, float]], func_type: FitFunctionType = FitFunctionType.LINEAR
) -> Callable[[float], float]:
    """Fit a 1D function of the specified fit function type to the given points."""

    x_values = [i for i, j in points]
    y_values = [j for i, j in points]
    params, _ = curve_fit(  # pylint: disable=W0632 (unbalanced-tuple-unpacking)
        FUNCTION_FACTORY[func_type], x_values, y_values
    )
    return lambda x: FUNCTION_FACTORY[func_type](x, *params)
