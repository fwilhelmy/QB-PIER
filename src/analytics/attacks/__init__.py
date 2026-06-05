"""Attack analytics helpers."""

from .model_blacklist import (
    ATTACK_MODEL_BLACKLIST,
    ATTACK_MODEL_BLACKLIST_SUBSTRING_RULES,
    filter_attack_aggregate_df,
    model_name_is_blacklisted,
)
from .transfer_scatter import (
    ATTACK_TABLE_METRICS,
    TransferFilterMode,
    TransferMetricSpec,
    TransferScatterPlotOptions,
    DEFAULT_AXIS_CLIP,
    PAIR_METRIC_MAP,
    QUADRANT_REGION_ANNOTATIONS,
    available_transfer_metrics,
    resolve_transfer_metric,
    aggregate_architecture_profile,
    build_transfer_summary_table,
    compute_model_transfer_profile,
    detect_outliers,
    generate_interpretation_notes,
    load_pair_metrics,
    plot_architecture_transfer_scatter,
    plot_variant_transfer_scatter,
    run_transfer_scatter_figures,
    run_transfer_scatter_report,
)
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
try:
    from .attack_accuracy_df import plot_attack_accuracy_df
    from .attack_effect_on_accuracy import plot_attack_effect_on_accuracy
    from .confusion_matrix import plot_confusion_matrix
    from .loss_landscape import plot_loss_landscape
    from .perturbation_and_saliency import plot_perturbation_and_saliency
    from .results import store_attack_results
    from .transfer_attack_effects import plot_transfer_attack_effects
    from .heatmap import plot_tp_heatmaps_by_class
except ImportError:  # optional runtime deps (e.g. lightning, torch)
    plot_attack_accuracy_df = None
    plot_attack_effect_on_accuracy = None
    plot_confusion_matrix = None
    plot_loss_landscape = None
    plot_perturbation_and_saliency = None
    store_attack_results = None
    plot_transfer_attack_effects = None
    plot_tp_heatmaps_by_class = None

__all__ = [
    "ATTACK_MODEL_BLACKLIST",
    "ATTACK_MODEL_BLACKLIST_SUBSTRING_RULES",
    "filter_attack_aggregate_df",
    "model_name_is_blacklisted",
    "ATTACK_TABLE_METRICS",
    "PAIR_METRIC_MAP",
    "TransferFilterMode",
    "TransferMetricSpec",
    "TransferScatterPlotOptions",
    "DEFAULT_AXIS_CLIP",
    "QUADRANT_REGION_ANNOTATIONS",
    "available_transfer_metrics",
    "resolve_transfer_metric",
    "aggregate_architecture_profile",
    "build_transfer_summary_table",
    "compute_model_transfer_profile",
    "detect_outliers",
    "generate_interpretation_notes",
    "load_pair_metrics",
    "plot_architecture_transfer_scatter",
    "plot_variant_transfer_scatter",
    "run_transfer_scatter_figures",
    "run_transfer_scatter_report",
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
