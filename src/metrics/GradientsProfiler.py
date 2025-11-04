from pytorch_lightning import Callback, LightningModule, Trainer
from typing import Dict, Optional
import logging

log = logging.getLogger(__name__)


class GradientProfiler(Callback):
    """Compute gradient norms and log them at the end of the epoch."""

    def __init__(self) -> None:
        self.metric_group = "grad_l2"
        self._latest: Dict[str, float] = {}

    def _compute_gradient_norms(self, pl_module: LightningModule) -> Dict[str, float]:
        norms: Dict[str, float] = {}
        for name, param in pl_module.named_parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            value = float(grad.detach().norm(2).item())
            norms[f"{self.metric_group}/{name}"] = value
        return norms

    # Cache gradient L2 norms right after backward, before optimizer.zero_grad()
    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._latest = self._compute_gradient_norms(pl_module)

    # Log last computed gradient norms captured during backprop. Gradients
    # are typically cleared by the time we reach epoch end, so we cache
    # them in on_after_backward.
    def on_train_epoch_end(  # type: ignore[override]
        self,
        trainer: Optional[object],
        pl_module: LightningModule,
    ) -> None:
        gradients = self._latest or self._compute_gradient_norms(pl_module)
        if gradients:
            pl_module.log_dict(gradients, on_step=False, on_epoch=True)
            log.info("Logged gradient norms at epoch %d", pl_module.current_epoch)
        else:
            log.warning("No gradients found to log at epoch %d", pl_module.current_epoch)