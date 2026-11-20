"""
Tests for utils/logger.py — multi-format metric logging.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from deepfake_detector.utils.logger import MetricLogger, setup_logging


class TestSetupLogging:
    def test_creates_log_file(self, tmp_path: Path) -> None:
        setup_logging(tmp_path, run_name="test_run")
        log_file = tmp_path / "test_run.log"
        assert log_file.exists()

    def test_creates_log_dir_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        setup_logging(nested, run_name="nested")
        assert nested.exists()


class TestMetricLogger:
    @pytest.fixture()
    def logger(self, tmp_path: Path) -> MetricLogger:
        return MetricLogger(tmp_path, run_name="unit_test", use_tensorboard=False)

    def test_csv_file_created(self, logger: MetricLogger, tmp_path: Path) -> None:
        logger.start_epoch()
        logger.log_epoch(1, {"train_loss": 0.5, "val_acc": 0.8})
        logger.close()
        csv_path = tmp_path / "unit_test_metrics.csv"
        assert csv_path.exists()

    def test_csv_has_correct_columns(self, logger: MetricLogger, tmp_path: Path) -> None:
        logger.start_epoch()
        logger.log_epoch(1, {"train_loss": 0.5, "val_acc": 0.8}, lr=1e-4)
        logger.close()
        csv_path = tmp_path / "unit_test_metrics.csv"
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert "epoch" in rows[0]
        assert "train_loss" in rows[0]
        assert "val_acc" in rows[0]
        assert "lr" in rows[0]

    def test_json_summary_written(self, logger: MetricLogger, tmp_path: Path) -> None:
        logger.start_epoch()
        logger.log_epoch(1, {"train_loss": 0.5})
        logger.start_epoch()
        logger.log_epoch(2, {"train_loss": 0.3})
        logger.close()
        json_path = tmp_path / "unit_test_summary.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["run_name"] == "unit_test"
        assert len(data["history"]) == 2
        assert data["history"][0]["epoch"] == 1
        assert data["history"][1]["epoch"] == 2

    def test_multiple_epochs_appended_to_csv(self, logger: MetricLogger, tmp_path: Path) -> None:
        for epoch in range(1, 6):
            logger.start_epoch()
            logger.log_epoch(epoch, {"loss": float(epoch)})
        logger.close()
        csv_path = tmp_path / "unit_test_metrics.csv"
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        assert rows[-1]["epoch"] == "5"
