from src.LoggingService import LoggingService


def test_logging_service_saves_text_file(tmp_path):
    # Reset singleton instance
    LoggingService._instance = None

    service = LoggingService(root_dir=str(tmp_path))
    service.save_file("hello world", "info.txt")

    info_file = tmp_path / "info.txt"
    assert info_file.exists()
    assert info_file.read_text() == "hello world"

    # Clean up singleton to avoid cross-test contamination
    LoggingService._instance = None
