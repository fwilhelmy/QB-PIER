"""Shared attack analytics model blacklist (source or target)."""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

log = logging.getLogger(__name__)

# Exact model paths: drop any row where source or target equals one of these.
# Include legacy and canonical names so tables work before/after migration.
ATTACK_MODEL_BLACKLIST: tuple[str, ...] = (
    "Quantum/VQM/ansatz=no_entangling, qdepth=50",
)

# Substring rules: if ALL tokens appear in a model name (case-insensitive), blacklist it.
# Mirrors attack_matrix.ipynb: VQM variants with ansatz=no_entangling.
ATTACK_MODEL_BLACKLIST_SUBSTRING_RULES: tuple[tuple[str, ...], ...] = (
    ("no_entangling", "VQM"),
)


def _normalize_exact_blacklist(
    names: Sequence[str] | None,
) -> frozenset[str]:
    base = set(ATTACK_MODEL_BLACKLIST)
    if names:
        base.update(n.strip() for n in names if isinstance(n, str) and n.strip())
    return frozenset(base)


def model_name_is_blacklisted(
    name: str,
    *,
    exact: frozenset[str] | None = None,
    substring_rules: Sequence[tuple[str, ...]] | None = None,
) -> bool:
    """True if *name* matches an exact entry or any all-of substring rule."""
    if not isinstance(name, str) or not name.strip():
        return False

    exact_set = exact if exact is not None else _normalize_exact_blacklist(None)
    if name in exact_set:
        return True

    lower = name.lower()
    rules = (
        substring_rules
        if substring_rules is not None
        else ATTACK_MODEL_BLACKLIST_SUBSTRING_RULES
    )
    for rule in rules:
        tokens = [t.lower() for t in rule if t]
        if tokens and all(token in lower for token in tokens):
            return True
    return False


def blacklisted_model_mask(
    series: pd.Series,
    *,
    exact: frozenset[str] | None = None,
    substring_rules: Sequence[tuple[str, ...]] | None = None,
) -> pd.Series:
    """Boolean mask: True where the model name should be excluded."""
    return series.astype(str).map(
        lambda n: model_name_is_blacklisted(
            n, exact=exact, substring_rules=substring_rules
        )
    )


def filter_attack_aggregate_df(
    df: pd.DataFrame,
    *,
    source_col: str = "source_model",
    target_col: str = "target_model",
    extra_exact: Sequence[str] | None = None,
    substring_rules: Sequence[tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """
    Remove rows where *source* or *target* matches ``ATTACK_MODEL_BLACKLIST`` or
    substring rules (e.g. VQM + no_entangling).
    """
    if df.empty:
        return df.copy()

    for col in (source_col, target_col):
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not in dataframe.")

    exact = _normalize_exact_blacklist(extra_exact)
    rules = (
        substring_rules
        if substring_rules is not None
        else ATTACK_MODEL_BLACKLIST_SUBSTRING_RULES
    )

    src_bad = blacklisted_model_mask(df[source_col], exact=exact, substring_rules=rules)
    tgt_bad = blacklisted_model_mask(df[target_col], exact=exact, substring_rules=rules)
    kept = ~(src_bad | tgt_bad)
    removed = int((~kept).sum())
    if removed:
        log.info(
            "Attack blacklist removed %d rows (%d exact names, %d substring rules)",
            removed,
            len(exact),
            len(rules),
        )
    return df.loc[kept].copy()


MODEL_BLACKLIST: frozenset[str] = _normalize_exact_blacklist(None)
