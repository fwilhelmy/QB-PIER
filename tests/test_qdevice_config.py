from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def test_sim_qdevice_config_is_present_and_usable():
    cfg_path = ROOT / "conf" / "qdevice" / "sim.yaml"
    assert cfg_path.exists()

    cfg = _read_simple_yaml(cfg_path)
    # Supports both the old uppercase config names and the newer lowercase names.
    device = cfg.get("device") or cfg.get("QDEVICE")
    interface = cfg.get("interface") or cfg.get("INTERFACE")
    diff_method = cfg.get("diff_method") or cfg.get("QDIFF")

    assert device, "qdevice sim config must define device/QDEVICE"
    assert interface or diff_method, "qdevice sim config should define interface or diff method metadata"


def test_qdevice_module_imports_without_hydra():
    import qdevice  # noqa: F401
