import json
from unittest import mock

from click.testing import CliRunner

from mlflow.experiments import commands
from mlflow.tracing.constant import TraceTagKey


@mock.patch("mlflow.experiments._get_store")
class TestCreateWithArchivalRetention:
    def test_sets_retention_tag(self, mock_get_store):
        store = mock_get_store.return_value
        store.create_experiment.return_value = "1"

        result = CliRunner().invoke(
            commands,
            ["create", "-n", "test-exp", "--trace-archival-retention", "30d"],
        )

        assert result.exit_code == 0
        store.set_experiment_tag.assert_called_once()
        tag = store.set_experiment_tag.call_args[0][1]
        assert tag.key == TraceTagKey.ARCHIVAL_RETENTION
        value = json.loads(tag.value)
        assert value == {"type": "duration", "value": "30d"}

    def test_no_tag_when_retention_omitted(self, mock_get_store):
        store = mock_get_store.return_value
        store.create_experiment.return_value = "1"

        result = CliRunner().invoke(commands, ["create", "-n", "test-exp"])

        assert result.exit_code == 0
        store.set_experiment_tag.assert_not_called()

    def test_invalid_retention_rejected(self, mock_get_store):
        store = mock_get_store.return_value
        store.create_experiment.return_value = "1"

        result = CliRunner().invoke(
            commands,
            ["create", "-n", "test-exp", "--trace-archival-retention", "bad"],
        )

        assert result.exit_code != 0
        store.set_experiment_tag.assert_not_called()


@mock.patch("mlflow.experiments._get_store")
class TestUpdateExperiment:
    def test_sets_retention_tag(self, mock_get_store):
        store = mock_get_store.return_value

        result = CliRunner().invoke(
            commands,
            ["update", "-x", "42", "--trace-archival-retention", "12h"],
        )

        assert result.exit_code == 0
        store.set_experiment_tag.assert_called_once()
        tag = store.set_experiment_tag.call_args[0][1]
        assert tag.key == TraceTagKey.ARCHIVAL_RETENTION
        value = json.loads(tag.value)
        assert value == {"type": "duration", "value": "12h"}

    def test_archive_now_sets_empty_tag(self, mock_get_store):
        store = mock_get_store.return_value

        result = CliRunner().invoke(
            commands,
            ["update", "-x", "42", "--trace-archive-now"],
        )

        assert result.exit_code == 0
        store.set_experiment_tag.assert_called_once()
        tag = store.set_experiment_tag.call_args[0][1]
        assert tag.key == TraceTagKey.ARCHIVE_NOW
        assert json.loads(tag.value) == {}

    def test_archive_now_older_than_sets_duration_tag(self, mock_get_store):
        store = mock_get_store.return_value

        result = CliRunner().invoke(
            commands,
            ["update", "-x", "42", "--trace-archive-now-older-than", "1d"],
        )

        assert result.exit_code == 0
        store.set_experiment_tag.assert_called_once()
        tag = store.set_experiment_tag.call_args[0][1]
        assert tag.key == TraceTagKey.ARCHIVE_NOW
        value = json.loads(tag.value)
        assert value == {"older_than": "1d"}

    def test_archive_now_mutual_exclusivity(self, mock_get_store):
        result = CliRunner().invoke(
            commands,
            [
                "update", "-x", "42",
                "--trace-archive-now",
                "--trace-archive-now-older-than", "1d",
            ],
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_no_flags_rejected(self, mock_get_store):
        result = CliRunner().invoke(commands, ["update", "-x", "42"])

        assert result.exit_code != 0
        assert "At least one update flag" in result.output

    def test_invalid_retention_rejected(self, mock_get_store):
        result = CliRunner().invoke(
            commands,
            ["update", "-x", "42", "--trace-archival-retention", "garbage"],
        )

        assert result.exit_code != 0

    def test_invalid_archive_now_older_than_rejected(self, mock_get_store):
        result = CliRunner().invoke(
            commands,
            ["update", "-x", "42", "--trace-archive-now-older-than", "bad"],
        )

        assert result.exit_code != 0

    def test_retention_and_archive_now_together(self, mock_get_store):
        store = mock_get_store.return_value

        result = CliRunner().invoke(
            commands,
            [
                "update", "-x", "42",
                "--trace-archival-retention", "7d",
                "--trace-archive-now",
            ],
        )

        assert result.exit_code == 0
        assert store.set_experiment_tag.call_count == 2
