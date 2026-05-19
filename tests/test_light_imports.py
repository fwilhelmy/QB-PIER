import importlib


def test_lightweight_core_modules_import():
    # These imports should not instantiate models, load data, or require a GPU.
    for module_name in ["HydraFactory", "qdevice", "utils"]:
        importlib.import_module(module_name)
