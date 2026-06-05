"""Table extraction and CSV utilities for experiment results."""

from .extract_attack import build_pair_metrics
from .model_blacklist import (
    ATTACK_MODEL_BLACKLIST,
    ATTACK_MODEL_BLACKLIST_SUBSTRING_RULES,
    filter_attack_aggregate_df,
    model_name_is_blacklisted,
)

try:
    from .baseline_training_summary import (
        build_summary_dataframe,
        generate_baseline_training_summary,
    )
except ImportError:
    build_summary_dataframe = None
    generate_baseline_training_summary = None

from .csv_utils import (
    append_clean_accuracy_to_tables,
    extract_clean_accuracy,
    load_attack_tables,
    merge_csv_files,
    normalize_model_name,
)
try:
    from .extract_data import build_inventory_dataframe
except ImportError:
    build_inventory_dataframe = None

__all__ = [
    "ATTACK_MODEL_BLACKLIST",
    "ATTACK_MODEL_BLACKLIST_SUBSTRING_RULES",
    "filter_attack_aggregate_df",
    "model_name_is_blacklisted",
    "append_clean_accuracy_to_tables",
    "build_inventory_dataframe",
    "build_pair_metrics",
    "build_summary_dataframe",
    "extract_clean_accuracy",
    "generate_baseline_training_summary",
    "load_attack_tables",
    "merge_csv_files",
    "normalize_model_name",
]
