import re
from dataclasses import dataclass
from datetime import timedelta

from mlflow.exceptions import MlflowException

_DURATION_PATTERN = re.compile(r"^(\d+)([mhd])$")

_UNIT_LABELS = {
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


@dataclass(frozen=True)
class ArchivalDuration:
    """A validated archival retention duration (e.g. ``30d``, ``12h``, ``5m``)."""

    value: int
    unit: str

    def to_timedelta(self) -> timedelta:
        return timedelta(**{_UNIT_LABELS[self.unit]: self.value})

    def __str__(self) -> str:
        return f"{self.value}{self.unit}"


def parse_duration(raw: str) -> ArchivalDuration:
    """Parse a ``<number><unit>`` duration string.

    Accepted units: ``m`` (minutes), ``h`` (hours), ``d`` (days).

    Args:
        raw: The duration string to parse (e.g. ``"30d"``).

    Returns:
        An ``ArchivalDuration`` with the parsed value and unit.

    Raises:
        MlflowException: If *raw* is empty, has an unrecognised format,
            or specifies a non-positive value.
    """
    if not raw:
        raise MlflowException(
            "Archival duration must not be empty. "
            "Expected format: <number><unit> where unit is m, h, or d (e.g. '30d')."
        )

    match = _DURATION_PATTERN.match(raw.strip())
    if match is None:
        raise MlflowException(
            f"Invalid archival duration '{raw}'. "
            "Expected format: <number><unit> where unit is m, h, or d (e.g. '30d', '12h', '5m')."
        )

    value = int(match.group(1))
    unit = match.group(2)

    if value <= 0:
        raise MlflowException(
            f"Archival duration must be a positive integer, got '{raw}'."
        )

    return ArchivalDuration(value=value, unit=unit)
