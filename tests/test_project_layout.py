from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_project_layout_exists():
    expected_paths = [
        "conf/train.yaml",
        "conf/attack.yaml",
        "conf/dataset/mnist.yaml",
        "conf/attack/fgsm.yaml",
        "src/train.py",
        "src/attack.py",
        "src/qdevice.py",
        "docs/artifacts.md",
        "docs/training.md",
        "docs/attacking.md",
        "docs/implementation_state.md",
        "docs/project_structure.md",
    ]

    missing = [path for path in expected_paths if not (ROOT / path).exists()]
    assert not missing, f"Missing expected project files: {missing}"
