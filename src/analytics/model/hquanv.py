from __future__ import annotations

import math
import os
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor


def is_hquanv_model(model: Any) -> bool:
    """Return True when ``model`` looks like an HQNN using a quanv encoder."""
    encoder = getattr(model, "encoder", None)
    quanv = getattr(encoder, "quanv", None)
    return quanv is not None and hasattr(quanv, "vqc") and hasattr(quanv, "readout")


def _resolve_quanv2d(model: Any):
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "quanv"):
        raise ValueError("Model does not expose an encoder.quanv block.")
    return encoder.quanv


def _to_spatial_maps(flat_patch_features: Tensor, batch_size: int, height: int, width: int) -> Tensor:
    """Reshape patchwise features ``[B*L, C]`` into image-like maps ``[B, C, H, W]``."""
    channels = flat_patch_features.shape[-1]
    maps = flat_patch_features.view(batch_size, height * width, channels)
    maps = maps.permute(0, 2, 1).reshape(batch_size, channels, height, width)
    return maps


def extract_hquanv_maps(model: Any, inputs: Tensor) -> dict[str, Tensor]:
    """Explicitly traverse the quanv encoder and return raw/readout feature maps.

    Returned tensors are detached CPU tensors.
    """
    quanv2d = _resolve_quanv2d(model)

    if inputs.ndim != 4:
        raise ValueError(f"Expected inputs with shape [B, C, H, W], got {tuple(inputs.shape)}")

    device = next(model.parameters()).device
    x = inputs.to(device)

    patches = F.unfold(
        x,
        kernel_size=quanv2d.kernel_size,
        dilation=quanv2d.dilation,
        padding=quanv2d.padding,
        stride=quanv2d.stride,
    )
    batch_size = x.size(0)
    n_locations = patches.size(-1)
    flat_patches = patches.transpose(1, 2).reshape(batch_size * n_locations, quanv2d.n_qubits)

    raw_quantum = quanv2d.vqc(flat_patches)
    readout = quanv2d.readout(raw_quantum)

    h_out = int(
        (x.size(2) + 2 * quanv2d.padding - quanv2d.dilation * (quanv2d.kernel_size - 1) - 1)
        / quanv2d.stride
        + 1
    )
    w_out = int(
        (x.size(3) + 2 * quanv2d.padding - quanv2d.dilation * (quanv2d.kernel_size - 1) - 1)
        / quanv2d.stride
        + 1
    )

    raw_maps = _to_spatial_maps(raw_quantum, batch_size=batch_size, height=h_out, width=w_out)
    readout_maps = _to_spatial_maps(readout, batch_size=batch_size, height=h_out, width=w_out)

    return {
        "inputs": x.detach().cpu(),
        "patches": flat_patches.detach().cpu(),
        "raw_quantum": raw_quantum.detach().cpu(),
        "readout": readout.detach().cpu(),
        "raw_maps": raw_maps.detach().cpu(),
        "readout_maps": readout_maps.detach().cpu(),
    }


def _normalise_map(image: Tensor) -> Tensor:
    image = image.detach().float().cpu()
    image_min = image.min()
    image_max = image.max()
    denom = image_max - image_min
    if float(denom) <= 0.0:
        return torch.zeros_like(image)
    return (image - image_min) / denom


def _plot_channel_grid(maps: Tensor, title: str, save_path: str, cmap: str = "gray") -> None:
    channels = maps.shape[0]
    grid_size = int(math.ceil(math.sqrt(channels)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, ax in enumerate(axes):
        if idx < channels:
            image = _normalise_map(maps[idx])
            ax.imshow(image.numpy(), cmap=cmap)
            ax.set_title(f"ch {idx}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_input_image(image: Tensor, save_path: str) -> None:
    image = image.detach().float().cpu()
    fig, ax = plt.subplots(figsize=(4, 4))
    if image.shape[0] == 1:
        ax.imshow(image[0].numpy(), cmap="gray")
    else:
        display = image.permute(1, 2, 0)
        display = _normalise_map(display)
        ax.imshow(display.numpy())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("input")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(image: Tensor, raw_maps: Tensor, readout_maps: Tensor, save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    if image.shape[0] == 1:
        axes[0].imshow(image[0].numpy(), cmap="gray")
    else:
        display = _normalise_map(image.permute(1, 2, 0))
        axes[0].imshow(display.numpy())
    axes[0].set_title("input")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    raw_preview = _normalise_map(raw_maps.mean(dim=0))
    readout_preview = _normalise_map(readout_maps.mean(dim=0))
    axes[1].imshow(raw_preview.numpy(), cmap="gray")
    axes[1].set_title("raw quantum mean")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(readout_preview.numpy(), cmap="gray")
    axes[2].set_title("readout mean")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualize_hquanv_sample(
    model: Any,
    inputs: Tensor,
    out_dir: str,
    prefix: str = "sample",
) -> dict[str, Tensor]:
    """Save raw quantum and readout analyticss for one or more samples."""
    os.makedirs(out_dir, exist_ok=True)
    outputs = extract_hquanv_maps(model, inputs)

    batch_size = outputs["inputs"].shape[0]
    for batch_idx in range(batch_size):
        sample_prefix = os.path.join(out_dir, f"{prefix}_{batch_idx:02d}")
        image = outputs["inputs"][batch_idx]
        raw_maps = outputs["raw_maps"][batch_idx]
        readout_maps = outputs["readout_maps"][batch_idx]

        _plot_input_image(image, f"{sample_prefix}_input.png")
        _plot_channel_grid(raw_maps, "raw quantum output", f"{sample_prefix}_raw_quantum.png")
        _plot_channel_grid(readout_maps, "linear readout output", f"{sample_prefix}_readout.png")
        _plot_overview(image, raw_maps, readout_maps, f"{sample_prefix}_overview.png")

    torch.save(outputs, os.path.join(out_dir, f"{prefix}_feature_maps.pt"))
    return outputs
