"""Attack analytics helpers."""

from .attack_accuracy_df import plot_attack_accuracy_df
from .attack_effect_on_accuracy import plot_attack_effect_on_accuracy
from .confusion_matrix import plot_confusion_matrix
from .loss_landscape import plot_loss_landscape
from .perturbation_and_saliency import plot_perturbation_and_saliency
from .results import store_attack_results
from .transfer_attack_effects import plot_transfer_attack_effects
from .heatmap import plot_tp_heatmaps_by_class
from .attack_matrix import (
    ATTACK_TABLE_METRICS,
    MatrixAxisGroups,
    available_attack_metrics,
    plot_model_attack_effectiveness_matrix,
    cell_effectiveness_auc_drop,
    cell_effectiveness_eps_to_fraction,
    cell_effectiveness_max_rel_drop,
    cell_effectiveness_max_steepness,
    cell_effectiveness_metric_auc,
    cell_effectiveness_robust_auc,
    make_metric_cell_fn,
    validate_attack_metric,
    assign_axis_group_key,
    model_group_key,
    ordered_axis_group_labels,
    parse_canonical_model_name,
)

__all__ = [
    "plot_attack_accuracy_df",
    "plot_attack_effect_on_accuracy",
    "plot_confusion_matrix",
    "plot_loss_landscape",
    "plot_perturbation_and_saliency",
    "plot_transfer_attack_effects",
    "store_attack_results",
    "plot_tp_heatmaps_by_class",
    "plot_model_attack_effectiveness_matrix",
    "cell_effectiveness_auc_drop",
    "cell_effectiveness_eps_to_fraction",
    "cell_effectiveness_max_rel_drop",
    "cell_effectiveness_max_steepness",
    "cell_effectiveness_robust_auc",
    "cell_effectiveness_metric_auc",
    "make_metric_cell_fn",
    "ATTACK_TABLE_METRICS",
    "available_attack_metrics",
    "validate_attack_metric",
    "MatrixAxisGroups",
    "assign_axis_group_key",
    "model_group_key",
    "ordered_axis_group_labels",
    "parse_canonical_model_name",
]
