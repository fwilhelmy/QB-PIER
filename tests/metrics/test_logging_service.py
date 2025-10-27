import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import types
sys.modules.setdefault("psutil", types.ModuleType("psutil"))

from src.metrics.LoggingService import LoggingService

log = logging.getLogger(__name__)


def test_logging_service_writes_errors(tmp_path, caplog):
    # Reset singleton instance
    LoggingService._instance = None

    service = LoggingService(root_dir=str(tmp_path))
    with caplog.at_level(logging.ERROR, logger=log.name):
        log.error("first error")
        try:
            raise Exception("second error")
        except Exception:
            log.exception("second error")
    service.save_experiment()

    error_file = tmp_path / "errors.log"
    assert error_file.exists()
    content = error_file.read_text()
    assert "first error" in content
    assert "second error" in content
    assert "first error" in caplog.text
    assert "second error" in caplog.text

    # Clean up singleton to avoid cross-test contamination
    LoggingService._instance = None
