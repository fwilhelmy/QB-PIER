"""Shared fixtures for model tests."""

from __future__ import annotations

import importlib

import pytest

from metrics.LoggingService import LoggingService


@pytest.fixture
def stub_lightning_components(monkeypatch):
    """Replace Lightning utilities that would otherwise hit the filesystem."""

    class _StubTensorBoardLogger:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _StubModelCheckpoint:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    base_module = importlib.import_module("models.BaseModel")
    monkeypatch.setattr(base_module, "TensorBoardLogger", _StubTensorBoardLogger)
    monkeypatch.setattr(base_module, "ModelCheckpoint", _StubModelCheckpoint)

    return _StubTensorBoardLogger, _StubModelCheckpoint


@pytest.fixture
def logging_service_root(tmp_path):
    """Ensure every test uses an isolated logging directory."""

    LoggingService._instance = None
    service = LoggingService(root_dir=str(tmp_path / "logs"))
    try:
        yield service
    finally:
        LoggingService._instance = None
