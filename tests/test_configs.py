from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text()


def test_mnist_is_documented_primary_dataset_config():
    text = _text("conf/dataset/mnist.yaml")
    assert "MNIST" in text
    assert "input_shape" in text
    assert "classes" in text


def test_fgsm_is_documented_primary_attack_config():
    text = _text("conf/attack/fgsm.yaml")
    assert "FGSM" in text or "fgsm" in text.lower()
