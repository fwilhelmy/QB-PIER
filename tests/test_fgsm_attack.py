import pytest
import importlib.util
import pathlib
import sys
import types

torch = pytest.importorskip("torch")

# Load FGSMAttack without importing the whole package to avoid side effects
_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "attacks" / "FGSMAttack.py"
spec = importlib.util.spec_from_file_location("src.attacks.FGSMAttack", _path)
module = importlib.util.module_from_spec(spec)

package = types.ModuleType("src.attacks")
package.__path__ = [str(_path.parent)]
sys.modules.setdefault("src.attacks", package)

data_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "data"
data_pkg = types.ModuleType("src.data")
data_pkg.__path__ = [str(data_path)]
sys.modules.setdefault("src.data", data_pkg)

config_pkg = types.ModuleType("Configuration")
config_pkg.DATA_DIR = ""
sys.modules.setdefault("Configuration", config_pkg)

spec.loader.exec_module(module)
FGSMAttack = module.FGSMAttack


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @property
    def loss_fn(self):
        return torch.nn.functional.cross_entropy

    @property
    def model_name(self):
        return "dummy"


def test_fgsm_epsilon_variation():
    model = DummyModel()
    images = torch.zeros(1, 1, 28, 28)
    labels = torch.tensor([1])
    dataset = torch.utils.data.TensorDataset(images, labels)

    fgsm = FGSMAttack(dataset)
    adv_small = fgsm.attack(model, epsilon=0.1)
    adv_large = fgsm.attack(model, epsilon=0.3)

    assert not torch.allclose(adv_small, adv_large)
