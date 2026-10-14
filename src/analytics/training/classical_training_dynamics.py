"""Epoch-level, two-seed training dynamics for selected classical models.

The saved logger pickle is the source of truth.  Validation at step zero is a
pretraining sanity check, so it is excluded when pairing epoch-end metrics.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm.auto import tqdm
except ImportError:  # The bundled plotting runtime does not include tqdm.
    def tqdm(items, **_kwargs):
        return items


SEEDS = (0, 42)
PLOT_EPOCHS = 35
ARCHITECTURES = ("MLP", "ConvNet", "ResNet18", "MLP-Mixer")
MODEL_CLASSES = {"MLP": "mlp", "ConvNet": "convnet", "ResNet18": "resnet18", "MLP-Mixer": "mlpmixer"}
COLORS = {"MLP": "#0072B2", "ConvNet": "#D55E00", "ResNet18": "#009E73", "MLP-Mixer": "#7858A6"}
MARKERS = {"MLP": "circle", "ConvNet": "square", "ResNet18": "triangle", "MLP-Mixer": "diamond"}
METRICS = ("train/loss", "val/loss", "train/acc", "val/acc")


@dataclass(frozen=True)
class MeanCurve:
    model_dir: Path
    source_files: tuple[Path, Path]
    steps: tuple[int, ...]
    epochs: tuple[int, ...]
    available_epochs: int
    metrics: dict[str, tuple[float, ...]]
    skipped_pretraining_validation: bool


def discover_model_configs(model_root: Path) -> list[Path]:
    """Find configurations by the project-required ``.model`` anchor."""
    return sorted(anchor.parent for anchor in model_root.rglob(".model"))


def _run_metrics(model_dir: Path, seed: int) -> tuple[Path, tuple[int, ...], dict[str, tuple[float, ...]], bool]:
    source = model_dir / str(seed) / "metrics" / "logger.pkl"
    if not source.is_file():
        raise FileNotFoundError(f"Required seed {seed} is missing: {source}")
    with source.open("rb") as handle:
        raw = pickle.load(handle)["_raw"]

    train_steps = tuple(int(step) for step in raw["train"]["step"])
    val_steps = tuple(int(step) for step in raw["val"]["step"])
    skip_initial_val = len(val_steps) == len(train_steps) + 1 and val_steps[0] == 0 and val_steps[1:] == train_steps
    if skip_initial_val:
        val_steps = val_steps[1:]
    if not train_steps or train_steps != val_steps or any(b <= a for a, b in zip(train_steps, train_steps[1:])):
        raise ValueError(f"Training and validation epoch-end steps do not align: {source}")

    metrics = {}
    for group in ("train", "val"):
        for name in ("loss", "acc"):
            values = tuple(float(value) for value in raw[group][name])
            if group == "val" and skip_initial_val:
                values = values[1:]
            if len(values) != len(train_steps) or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Incomplete or nonfinite {group}/{name} values: {source}")
            if name == "loss" and not all(value > 0 for value in values):
                raise ValueError(f"Nonpositive loss prevents a logarithmic axis: {source}")
            if name == "acc" and not all(0 <= value <= 1 for value in values):
                raise ValueError(f"Accuracy is not stored as a fraction in [0, 1]: {source}")
            metrics[f"{group}/{name}"] = values
    return source, train_steps, metrics, skip_initial_val


def _comparable_hparams(model_dir: Path) -> None:
    snapshots = []
    for seed in SEEDS:
        path = model_dir / str(seed) / "tensorboard" / "hparams.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Seed comparison needs the saved hyperparameters: {path}")
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                 if not line.startswith(("model_path:", "log_dir:"))]
        snapshots.append(lines)
    if snapshots[0] != snapshots[1]:
        raise ValueError(f"Seed 0 and 42 have different model hyperparameters: {model_dir}")


def prepare_curves(selected: Mapping[str, Path]) -> dict[str, MeanCurve]:
    """Validate both seeds and average each epoch's four metrics arithmetically."""
    if set(selected) != set(ARCHITECTURES):
        raise ValueError(f"Select exactly these architectures: {ARCHITECTURES}")
    curves = {}
    for name in tqdm(ARCHITECTURES, desc="Loading classical runs"):
        model_dir = Path(selected[name])
        anchor = model_dir / ".model"
        if not anchor.is_file() or anchor.read_text(encoding="utf-8").strip().casefold() != MODEL_CLASSES[name]:
            raise ValueError(f"Model anchor does not identify {name}: {anchor}")
        _comparable_hparams(model_dir)
        runs = [_run_metrics(model_dir, seed) for seed in SEEDS]
        if runs[0][1] != runs[1][1]:
            raise ValueError(f"Seed epoch-end steps differ for {name}: {model_dir}")
        steps = runs[0][1]
        if len(steps) < PLOT_EPOCHS:
            raise ValueError(f"{name} has only {len(steps)} epochs; the shared {PLOT_EPOCHS}-epoch figure cannot be made")
        metrics = {
            metric: tuple((a + b) / 2 for a, b in zip(runs[0][2][metric], runs[1][2][metric]))[:PLOT_EPOCHS]
            for metric in METRICS
        }
        curves[name] = MeanCurve(
            model_dir=model_dir,
            source_files=(runs[0][0], runs[1][0]),
            steps=steps[:PLOT_EPOCHS],
            epochs=tuple(range(1, PLOT_EPOCHS + 1)),
            available_epochs=len(steps),
            metrics=metrics,
            skipped_pretraining_validation=runs[0][3] and runs[1][3],
        )
    return curves


def _font(size: int, bold: bool = False):
    names = ["C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
             "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _marker(draw: ImageDraw.ImageDraw, x: float, y: float, name: str, radius: int = 9) -> None:
    color = COLORS[name]
    shape = MARKERS[name]
    if shape == "circle":
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="white", outline=color, width=4)
    elif shape == "square":
        draw.rectangle((x-radius, y-radius, x+radius, y+radius), fill="white", outline=color, width=4)
    elif shape == "triangle":
        draw.polygon([(x, y-radius-2), (x-radius-2, y+radius), (x+radius+2, y+radius)], fill="white")
        draw.line([(x, y-radius-2), (x-radius-2, y+radius), (x+radius+2, y+radius), (x, y-radius-2)], fill=color, width=4, joint="curve")
    else:
        draw.polygon([(x, y-radius-2), (x+radius+2, y), (x, y+radius+2), (x-radius-2, y)], fill="white")
        draw.line([(x, y-radius-2), (x+radius+2, y), (x, y+radius+2), (x-radius-2, y), (x, y-radius-2)], fill=color, width=4, joint="curve")


def _rotated_label(image: Image.Image, text: str, x: int, y: int, font) -> None:
    box = ImageDraw.Draw(image).textbbox((0, 0), text, font=font)
    label = Image.new("RGBA", (box[2] - box[0] + 20, box[3] - box[1] + 20), "white")
    ImageDraw.Draw(label).text((10-box[0], 10-box[1]), text, fill="#26313D", font=font)
    label = label.rotate(90, expand=True)
    image.paste(label, (x-label.width//2, y-label.height//2), label)


def save_figure(curves: Mapping[str, MeanCurve], output_path: Path) -> Path:
    """Render the four-panel thesis PNG with shared row scales at 300 dpi."""
    width, height = 3500, 2300
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    heading_font = _font(66, True)
    axis_font, tick_font = _font(60), _font(56)

    legend_widths = (330, 420, 440, 560)
    legend_gap = 50
    legend_start = (width - sum(legend_widths) - 3*legend_gap)//2
    for index, name in enumerate(ARCHITECTURES):
        x = legend_start + sum(legend_widths[:index]) + index*legend_gap
        y = 70
        draw.line((x, y, x+74, y), fill=COLORS[name], width=7)
        _marker(draw, x+37, y, name, 11)
        draw.text((x+94, y), name, fill="#26313D", font=axis_font, anchor="lm")

    x_left, x_right, panel_width = 250, 1910, 1500
    top_y, bottom_y, panel_height = 240, 1220, 800
    max_epoch = max(len(curves[name].epochs) for name in ARCHITECTURES)
    all_accuracy = [100*value for curve in curves.values() for metric in ("train/acc", "val/acc") for value in curve.metrics[metric]]
    acc_min = 10 * math.floor(min(all_accuracy)/10)
    acc_ticks = list(range(acc_min, 101, 5))
    all_losses = [value for curve in curves.values() for metric in ("train/loss", "val/loss") for value in curve.metrics[metric]]
    loss_low = 10**math.floor(math.log10(min(all_losses)))
    loss_high = 10**math.ceil(math.log10(max(all_losses)))
    loss_ticks = [10**power for power in range(round(math.log10(loss_low)), round(math.log10(loss_high))+1)]

    for col, group in enumerate(("train", "val")):
        px = x_left if col == 0 else x_right
        draw.text((px + panel_width//2, 165), "Training" if col == 0 else "Validation",
                  fill="#18232F", font=heading_font, anchor="mm")

        for row, metric in enumerate(("loss", "acc")):
            py = top_y if row == 0 else bottom_y
            y_low, y_high = (loss_low, loss_high) if row == 0 else (acc_min, 100)
            ticks = loss_ticks if row == 0 else acc_ticks

            def x_map(epoch: int) -> float:
                return px + (epoch-1)/(max_epoch-1)*panel_width

            def y_map(value: float) -> float:
                fraction = (math.log10(value)-math.log10(y_low))/(math.log10(y_high)-math.log10(y_low)) if row == 0 else (value-y_low)/(y_high-y_low)
                return py + panel_height*(1-fraction)

            for tick in range(1, max_epoch+1):
                if tick == 1 or tick % 5 == 0:
                    xp = round(x_map(tick))
                    draw.line((xp, py, xp, py+panel_height), fill="#EFF2F5", width=2)
                    draw.line((xp, py+panel_height, xp, py+panel_height+11), fill="#687581", width=3)
                    if row == 1:
                        draw.text((xp, py+panel_height+48), str(tick), fill="#45515E", font=tick_font, anchor="mm")
            for tick in ticks:
                yp = round(y_map(tick))
                draw.line((px, yp, px+panel_width, yp), fill="#E6EBF0", width=2)
                if col == 0:
                    draw.line((px-11, yp, px, yp), fill="#687581", width=3)
                    tick_text = f"{tick:g}" if row == 0 else str(tick)
                    draw.text((px-22, yp), tick_text, fill="#45515E", font=tick_font, anchor="rm")
            draw.line((px, py, px, py+panel_height), fill="#67727D", width=4)
            draw.line((px, py+panel_height, px+panel_width, py+panel_height), fill="#67727D", width=4)
            if col == 0:
                _rotated_label(image, "Loss" if row == 0 else "Accuracy (%)", px-195, py+panel_height//2, axis_font)
            if row == 1:
                draw.text((px+panel_width//2, py+panel_height+145), "Epoch", fill="#26313D", font=axis_font, anchor="mm")

            for name in ARCHITECTURES:
                values = curves[name].metrics[f"{group}/{metric}"]
                if metric == "acc":
                    values = tuple(100*value for value in values)
                points = [(x_map(epoch), y_map(value)) for epoch, value in zip(curves[name].epochs, values)]
                draw.line(points, fill=COLORS[name], width=7, joint="curve")
                for index, (xp, yp) in enumerate(points):
                    if index == 0 or (index+1) % 5 == 0 or index == len(points)-1:
                        _marker(draw, xp, yp, name)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", dpi=(300, 300), optimize=True)
    return output_path
