from datetime import timedelta

import pytest

from mlflow.exceptions import MlflowException
from mlflow.tracing.archival_duration import ArchivalDuration, parse_duration
from mlflow.tracing.constant import SpansLocation


class TestParseDuration:
    @pytest.mark.parametrize(
        ("raw", "expected_value", "expected_unit"),
        [
            ("30d", 30, "d"),
            ("12h", 12, "h"),
            ("5m", 5, "m"),
            ("1d", 1, "d"),
            ("999d", 999, "d"),
        ],
    )
    def test_valid(self, raw, expected_value, expected_unit):
        result = parse_duration(raw)
        assert result == ArchivalDuration(value=expected_value, unit=expected_unit)

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "30x",
            "abc",
            "d30",
            "30",
            "d",
            "1.5d",
            "30 d",
            "-1d",
        ],
    )
    def test_invalid(self, raw):
        with pytest.raises(MlflowException):
            parse_duration(raw)

    def test_whitespace_stripped(self):
        result = parse_duration("  30d  ")
        assert result == ArchivalDuration(value=30, unit="d")


class TestArchivalDuration:
    @pytest.mark.parametrize(
        ("duration", "expected_td"),
        [
            (ArchivalDuration(30, "d"), timedelta(days=30)),
            (ArchivalDuration(12, "h"), timedelta(hours=12)),
            (ArchivalDuration(5, "m"), timedelta(minutes=5)),
        ],
    )
    def test_to_timedelta(self, duration, expected_td):
        assert duration.to_timedelta() == expected_td

    def test_str_roundtrip(self):
        for raw in ("30d", "12h", "5m"):
            assert str(parse_duration(raw)) == raw

    def test_frozen(self):
        d = ArchivalDuration(30, "d")
        with pytest.raises(AttributeError):
            d.value = 10


class TestSpansLocationArchiveRepo:
    def test_archive_repo_member(self):
        assert SpansLocation.ARCHIVE_REPO == "ARCHIVE_REPO"
        assert SpansLocation.ARCHIVE_REPO in SpansLocation

    def test_all_members(self):
        expected = {"TRACKING_STORE", "ARTIFACT_REPO", "ARCHIVE_REPO"}
        assert {m.value for m in SpansLocation} == expected
