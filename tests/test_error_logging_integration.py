import sys
from pathlib import Path

# Ensure src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics.LoggingService import LoggingService
from metrics.MetricsContainer import MetricsContainer
from models.BaseModel import BaseModel

class DummyModel(BaseModel):
    def forward(self, x):
        return x

    @property
    def global_step(self):
        return 0


def test_base_model_fit_logs_error(tmp_path, capsys):
    LoggingService._instance = None
    LoggingService(root_dir=str(tmp_path))

    model = DummyModel(model_path="exp", input_size=(1, 2, 3), output_size=1)
    # Avoid pandas usage
    model.metrics.save_metrics = lambda: None

    class DummyTrainer:
        def fit(self, *args, **kwargs):
            raise RuntimeError("boom")
        def save_checkpoint(self, path):
            pass
    model.trainer = DummyTrainer()

    model.fit(None, None)
    captured = capsys.readouterr()
    LoggingService().save_experiment()
    error_file = tmp_path / "errors.log"
    assert error_file.exists()
    content = error_file.read_text()
    assert "Error during training: boom" in content
    assert "Error during training: boom" in captured.out
    LoggingService._instance = None


def test_metrics_container_load_metrics_logs_error(tmp_path, capsys):
    LoggingService._instance = None
    LoggingService(root_dir=str(tmp_path))

    mc = MetricsContainer(str(tmp_path))
    mc.load_metrics(str(tmp_path))
    captured = capsys.readouterr()
    LoggingService().save_experiment()

    error_file = tmp_path / "errors.log"
    assert error_file.exists()
    content = error_file.read_text()
    assert "Metrics file not found" in content
    assert "Metrics file not found" in captured.out
    LoggingService._instance = None
