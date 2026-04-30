"""
Utilities for handling time series frequency conversions.

Provides conversion between hours and time steps based on data frequency.
"""

from typing import Dict


# Frequency to steps per hour mapping
FREQ_TO_STEPS_PER_HOUR: Dict[str, int] = {
    "15min": 4,  # 4 steps per hour
    "30min": 2,  # 2 steps per hour
    "h": 1,      # 1 step per hour
}


def hours_to_steps(hours: int, frequency: str) -> int:
    """
    Convert hours to time steps based on data frequency.

    Args:
        hours: Number of hours
        frequency: Data frequency ('15min', '30min', 'h')

    Returns:
        Number of time steps

    Raises:
        ValueError: If frequency is not supported

    Examples:
        >>> hours_to_steps(24, '15min')
        96
        >>> hours_to_steps(6, '15min')
        24
        >>> hours_to_steps(24, 'h')
        24
    """
    if frequency not in FREQ_TO_STEPS_PER_HOUR:
        raise ValueError(
            f"Unsupported frequency: {frequency}. "
            f"Supported: {list(FREQ_TO_STEPS_PER_HOUR.keys())}"
        )

    steps_per_hour = FREQ_TO_STEPS_PER_HOUR[frequency]
    return hours * steps_per_hour


def steps_to_hours(steps: int, frequency: str) -> float:
    """
    Convert time steps to hours based on data frequency.

    Args:
        steps: Number of time steps
        frequency: Data frequency ('15min', '30min', 'h')

    Returns:
        Number of hours (may be fractional)

    Raises:
        ValueError: If frequency is not supported

    Examples:
        >>> steps_to_hours(96, '15min')
        24.0
        >>> steps_to_hours(24, '15min')
        6.0
        >>> steps_to_hours(24, 'h')
        24.0
    """
    if frequency not in FREQ_TO_STEPS_PER_HOUR:
        raise ValueError(
            f"Unsupported frequency: {frequency}. "
            f"Supported: {list(FREQ_TO_STEPS_PER_HOUR.keys())}"
        )

    steps_per_hour = FREQ_TO_STEPS_PER_HOUR[frequency]
    return steps / steps_per_hour


def get_seasonality_steps(frequency: str) -> int:
    """
    Get the number of steps for daily seasonality based on frequency.

    For NaiveSeasonal models, this determines the K parameter.

    Args:
        frequency: Data frequency ('15min', '30min', 'h')

    Returns:
        Number of steps in 24 hours (one day)

    Examples:
        >>> get_seasonality_steps('15min')
        96
        >>> get_seasonality_steps('h')
        24
    """
    return hours_to_steps(24, frequency)
