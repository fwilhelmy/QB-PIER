import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _default_config(**updates):
    from analytics.attacks.attacksv2_accuracy_curves import ROBUST_METRIC
    from analytics.attacks.attacksv2_transfer_tools import AttacksV2Config

    cfg = AttacksV2Config(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        metric=ROBUST_METRIC,
    )
    return cfg.with_updates(**updates) if updates else cfg


def test_source_mode_curve_output() -> None:
    from analytics.attacks.attacksv2_accuracy_curves import (
        build_accuracy_curve_table,
        make_synthetic_attack_curve_rows,
    )

    raw = make_synthetic_attack_curve_rows()
    curve = build_accuracy_curve_table(
        raw,
        _default_config(),
        mode="source",
        selected_group="Classical/CNN",
    )

    assert set(curve["source_group"]) == {"Classical/CNN"}
    assert set(curve["curve_group"]) == set(curve["target_group"])
    assert curve["epsilon"].nunique() == 3
    assert curve["robust_accuracy"].between(0.0, 1.0).all()


def test_target_mode_curve_output() -> None:
    from analytics.attacks.attacksv2_accuracy_curves import (
        build_accuracy_curve_table,
        make_synthetic_attack_curve_rows,
    )

    raw = make_synthetic_attack_curve_rows()
    curve = build_accuracy_curve_table(
        raw,
        _default_config(),
        mode="target",
        selected_group="Hybrid/HVQL",
    )

    assert set(curve["target_group"]) == {"Hybrid/HVQL"}
    assert set(curve["curve_group"]) == set(curve["source_group"])
    assert curve["epsilon"].nunique() == 3


def test_inter_architecture_excludes_same_architecture_variants() -> None:
    from analytics.attacks.attacksv2_accuracy_curves import (
        build_accuracy_curve_table,
        make_synthetic_attack_curve_rows,
    )

    raw = make_synthetic_attack_curve_rows()
    curve = build_accuracy_curve_table(
        raw,
        _default_config(),
        mode="source",
        selected_group="Classical/CNN",
    )

    assert "Classical/CNN" not in set(curve["target_group"])


def test_self_attacks_excluded_by_default_and_included_when_configured() -> None:
    from analytics.attacks.attacksv2_accuracy_curves import (
        build_accuracy_curve_table,
        make_synthetic_attack_curve_rows,
    )

    raw = make_synthetic_attack_curve_rows()
    cfg = _default_config(
        source_grain="variant",
        target_grain="variant",
        comparison_mode="inter_variant",
        apply_global_blacklist=False,
    )

    selected = "Classical/CNN/LeNet"
    default_curve = build_accuracy_curve_table(
        raw,
        cfg,
        mode="source",
        selected_group=selected,
    )
    with_self = build_accuracy_curve_table(
        raw,
        cfg,
        mode="source",
        selected_group=selected,
        include_self_attacks=True,
    )

    assert selected not in set(default_curve["target_group"])
    assert selected in set(with_self["target_group"])


def test_exact_and_substring_blacklists() -> None:
    from analytics.attacks.attacksv2_accuracy_curves import (
        build_accuracy_curve_table,
        make_synthetic_attack_curve_rows,
    )

    raw = make_synthetic_attack_curve_rows()
    exact_curve = build_accuracy_curve_table(
        raw,
        _default_config(blacklist_exact=("Classical/MLP/Base",)),
        mode="source",
        selected_group="Classical/CNN",
    )
    contains_curve = build_accuracy_curve_table(
        raw,
        _default_config(blacklist_contains=("HVQL",)),
        mode="source",
        selected_group="Classical/CNN",
    )

    assert "Classical/MLP" not in set(exact_curve["target_group"])
    assert "Hybrid/HVQL" not in set(contains_curve["target_group"])


def test_custom_source_curve_filters_comparison_targets_in_order() -> None:
    from analytics.attacks.attacksv2_accuracy_curves import (
        build_custom_accuracy_curve_table,
        make_synthetic_attack_curve_rows,
    )

    raw = make_synthetic_attack_curve_rows()
    cfg = _default_config(
        source_grain="architecture",
        target_grain="variant",
        comparison_mode="inter_variant",
        apply_global_blacklist=False,
    )
    curve = build_custom_accuracy_curve_table(
        raw,
        cfg,
        primary_role="source",
        primary_group="Classical/CNN",
        comparison_groups=(
            "Classical/MLP/Base",
            "Classical/CNN/Wide",
            "Quantum/VQM/ansatz=basic, qdepth=50",
        ),
    )

    assert set(curve["source_group"]) == {"Classical/CNN"}
    assert list(curve.groupby("comparison_group", sort=False).ngroup().unique()) == [0, 1, 2]
    assert list(curve["comparison_order"].drop_duplicates()) == [0, 1, 2]
    assert set(curve["target_group"]) == {
        "Classical/MLP/Base",
        "Classical/CNN/Wide",
        "Quantum/VQM/ansatz=basic, qdepth=50",
    }


def test_custom_target_curve_filters_comparison_sources() -> None:
    from analytics.attacks.attacksv2_accuracy_curves import (
        build_custom_accuracy_curve_table,
        make_synthetic_attack_curve_rows,
    )

    raw = make_synthetic_attack_curve_rows()
    cfg = _default_config(
        source_grain="variant",
        target_grain="architecture",
        comparison_mode="inter_variant",
        apply_global_blacklist=False,
    )
    curve = build_custom_accuracy_curve_table(
        raw,
        cfg,
        primary_role="target",
        primary_group="Classical/CNN",
        comparison_groups=(
            "Quantum/VQM/ansatz=basic, qdepth=50",
            "Hybrid/HVQL/ansatz=strong, qdepth=50",
        ),
    )

    assert set(curve["target_group"]) == {"Classical/CNN"}
    assert set(curve["source_group"]) == {
        "Quantum/VQM/ansatz=basic, qdepth=50",
        "Hybrid/HVQL/ansatz=strong, qdepth=50",
    }


def test_curve_image_dir_uses_family_architecture_role_layout(tmp_path) -> None:
    from analytics.attacks.attacksv2_accuracy_curves import curve_file_stem, curve_image_dir

    source_dir = curve_image_dir(tmp_path, "Classical/CNN", "source")
    target_dir = curve_image_dir(tmp_path, "Quantum/VQM", "target")

    assert source_dir == tmp_path / "images" / "classical" / "as_source"
    assert target_dir == tmp_path / "images" / "quantum" / "as_target"
    assert curve_file_stem("Classical/CNN", "source") == "cnn_as_source"
    assert curve_file_stem("Quantum/VQM", "target") == "vqm_as_target"


def test_average_overview_curve_table_contains_source_and_target_panels() -> None:
    from analytics.attacks.attacksv2_accuracy_curves import (
        build_average_overview_curve_table,
        make_synthetic_attack_curve_rows,
    )

    raw = make_synthetic_attack_curve_rows()
    overview = build_average_overview_curve_table(raw, _default_config())

    assert set(overview["curve_mode"]) == {"source", "target"}
    assert overview["epsilon"].nunique() == 3
    assert set(overview.loc[overview["curve_mode"] == "source", "selected_group"]) == {"All targets"}
    assert set(overview.loc[overview["curve_mode"] == "target", "selected_group"]) == {"All sources"}
    assert overview["robust_accuracy"].between(0.0, 1.0).all()
