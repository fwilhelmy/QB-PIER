import os
from pathlib import Path

def _load_simple_yaml(path: Path) -> dict:
    cfg = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                cfg[key.strip()] = val.strip()
    return cfg

# Determine which qdevice config to load
_config_name = os.environ.get("QB_QDEVICE", "sim_backprop")
_conf_path = Path(__file__).resolve().parents[1] / "conf" / "qdevice" / f"{_config_name}.yaml"

try:
    _cfg = _load_simple_yaml(_conf_path)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Qdevice configuration '{_config_name}' not found at {_conf_path}"
    )

CIRCUIT_STYLE = _cfg.get("CIRCUIT_STYLE", "pennylane")
QDEVICE = _cfg.get("QDEVICE", "default.qubit")
INTERFACE = _cfg.get("INTERFACE", "torch")
QDIFF = _cfg.get("QDIFF", "backprop")
