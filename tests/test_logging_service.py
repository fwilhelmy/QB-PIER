import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
sys.modules.setdefault("psutil", types.ModuleType("psutil"))

from src.metrics.LoggingService import LoggingService


def test_logging_service_writes_errors(tmp_path, capsys):
    # Reset singleton instance
    LoggingService._instance = None

    service = LoggingService(root_dir=str(tmp_path))
    service.log_error("first error")
    service.log_error(Exception("second error"))
    captured = capsys.readouterr()
    service.save_experiment()

    error_file = tmp_path / "errors.log"
    assert error_file.exists()
    content = error_file.read_text()
    assert "first error" in content
    assert "second error" in content
    assert "first error" in captured.out
    assert "second error" in captured.out

    # Clean up singleton to avoid cross-test contamination
    LoggingService._instance = None
