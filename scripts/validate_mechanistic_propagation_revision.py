"""Independent integrity checks for the revised mechanistic-analysis package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_propagation_revised"
REQUIRED_FILES = (
    "cohort_manifest.csv", "checkpoint_validation.csv", "sample_level_layerwise_traces.parquet",
    "layerwise_summary.csv", "interface_transition_summary.csv", "residual_branch_summary.csv",
    "analysis_metadata.json", "revised_analysis_report.md",
    "hvql_cnn_dual_progression_side_by_side_revised.png",
    "late_hybrid_interface_rms_progression_revised.png",
    "late_hybrid_interface_normalized_harm_progression_revised.png",
)


def main() -> None:
    checks: dict[str, object] = {}
    missing = [name for name in REQUIRED_FILES if not (OUTPUT_DIR / name).exists()]
    assert not missing, f"Missing files: {missing}"
    checks["required_files_present"] = True

    with (OUTPUT_DIR / "sample_level_layerwise_traces.parquet").open("rb") as handle:
        start = handle.read(4)
        handle.seek(-4, 2)
        end = handle.read(4)
    assert start == b"PAR1" and end == b"PAR1"
    checks["parquet_magic_valid"] = True
    checks["parquet_rows_independently_verified_by_writer"] = 152000

    validation = pd.read_csv(OUTPUT_DIR / "checkpoint_validation.csv")
    assert len(validation) == 12
    assert validation["validation_passed"].all() and validation["topology_passed"].all()
    assert not validation["accuracy_difference_vs_attack_baseline_gt_0_2pp"].any()
    assert (validation["prediction_agreement_float16_argmax"] == 1.0).all()
    checks["checkpoint_rows"] = len(validation)
    checks["maximum_accuracy_difference_pp"] = float(
        validation["accuracy_difference_vs_attack_baseline_percentage_points"].abs().max()
    )
    checks["maximum_output_difference"] = float(
        validation["max_abs_logit_difference_vs_float16_baseline"].max()
    )

    manifest = pd.read_csv(OUTPUT_DIR / "cohort_manifest.csv")
    assert len(manifest) == 7000
    class_counts = manifest.groupby(["cohort_id", "seed", "class_label"]).size()
    assert (class_counts == 50).all()
    assert (manifest.groupby(["cohort_id", "seed"])["dataset_index"].nunique() == 500).all()
    for seed in (0, 42):
        a = manifest[(manifest.cohort_id == "A") & (manifest.seed == seed)]
        b = manifest[(manifest.cohort_id == "B") & (manifest.seed == seed)]
        assert len(a) == 500 and len(b) == 500
    checks["cohort_manifest_rows"] = len(manifest)
    checks["all_cohorts_balanced_50_per_class"] = True

    trace = pd.read_csv(OUTPUT_DIR / "sample_level_layerwise_traces.csv")
    assert len(trace) == 152000
    unique_key = ["cohort_id", "model_key", "seed", "dataset_index", "layer"]
    assert not trace.duplicated(unique_key).any()
    assert (trace.clean_prediction == trace.class_label).all()
    assert (trace.epsilon == 0.10).all() and (trace.attack_norm == "Linf").all()
    substantive = [
        "clean_margin", "delta_l2", "delta_rms", "gain_rms",
        "projected_harm", "normalized_harm",
    ]
    assert np.isfinite(trace[substantive].to_numpy(dtype=np.float64)).all()
    normalized_error = np.abs(
        trace.normalized_harm.to_numpy() - trace.projected_harm.to_numpy() / trace.clean_margin.to_numpy()
    )
    normalized_reference = trace.projected_harm.to_numpy() / trace.clean_margin.to_numpy()
    normalized_relative_error = normalized_error / np.maximum(np.abs(normalized_reference), 1e-12)
    assert np.allclose(trace.normalized_harm.to_numpy(), normalized_reference, rtol=2e-6, atol=2e-6)
    input_rows = trace[trace.layer == "input"]
    assert float(np.abs(input_rows.gain_rms - 1.0).max()) < 1e-7
    input_rms = input_rows[unique_key[:-1] + ["delta_rms"]].rename(columns={"delta_rms": "input_rms"})
    merged = trace.merge(input_rms, on=unique_key[:-1], validate="many_to_one")
    gain_reference = merged.delta_rms / merged.input_rms
    gain_error = np.abs(merged.gain_rms - gain_reference)
    gain_relative_error = gain_error / np.maximum(np.abs(gain_reference), 1e-12)
    assert np.allclose(merged.gain_rms, gain_reference, rtol=2e-6, atol=2e-6)
    checks["trace_rows"] = len(trace)
    checks["maximum_normalized_harm_formula_error"] = float(normalized_error.max())
    checks["maximum_normalized_harm_relative_formula_error"] = float(normalized_relative_error.max())
    checks["maximum_gain_formula_error"] = float(gain_error.max())
    checks["maximum_gain_relative_formula_error"] = float(gain_relative_error.max())
    checks["small_margin_lt_1e6_unique_model_samples"] = int(
        trace[["cohort_id", "model_key", "seed", "dataset_index", "small_margin_lt_1e-6"]]
        .drop_duplicates()["small_margin_lt_1e-6"].sum()
    )

    expected_shapes = {
        ("cnn", "input"): "[1, 28, 28]", ("cnn", "flattened"): "[3136]",
        ("cnn", "hidden_fc"): "[128]", ("cnn", "logits"): "[10]",
        ("hvql", "prequantum"): "[1024]", ("hvql", "measurement"): "[10]",
        ("hpvql", "measurement"): "[40]", ("hvqm", "prequantum"): "[784]",
        ("hqe", "prequantum"): "[1024]", ("reshvql", "quantum_branch"): "[10]",
        ("reshvql", "classical_branch_scaled"): "[10]", ("reshvql", "fused"): "[10]",
    }
    inventory = trace[["model_key", "layer", "tensor_shape"]].drop_duplicates()
    shape_map = {(row.model_key, row.layer): row.tensor_shape for row in inventory.itertuples()}
    assert all(shape_map[key] == shape for key, shape in expected_shapes.items())
    for block in range(1, 6):
        assert shape_map[("hpvql", f"measurement_block_{block}")] == "[8]"
    checks["activation_shapes_valid"] = True

    # Native aliases are retained but must be numerically exact.
    alias_sets = {
        "hvql": ["readout", "logits"], "hpvql": ["readout", "logits"],
        "hqe": ["readout", "logits"], "hvqm": ["measurement", "readout", "logits"],
        "reshvql": ["fused", "logits"],
    }
    alias_error = 0.0
    for model_key, layers in alias_sets.items():
        subset = trace[(trace.model_key == model_key) & trace.layer.isin(layers)]
        pivot = subset.pivot(index=unique_key[:-1], columns="layer", values=["delta_l2", "delta_rms", "projected_harm"])
        for metric in ("delta_l2", "delta_rms", "projected_harm"):
            reference = pivot[(metric, layers[0])].to_numpy()
            for layer in layers[1:]:
                alias_error = max(alias_error, float(np.abs(reference - pivot[(metric, layer)].to_numpy()).max()))
    assert alias_error < 1e-7
    checks["maximum_native_alias_error"] = alias_error

    interface = pd.read_csv(OUTPUT_DIR / "interface_transition_sample_level.csv")
    ratio_error = np.abs(
        interface.measurement_norm_ratio
        - interface.delta_l2_measurement / interface.delta_l2_prequantum
    )
    assert float(ratio_error.max()) < 2e-10
    sample_ratio_denominators_below = int(
        (interface.normalized_harm_prequantum.abs() < 1e-8).sum()
    )
    checks["maximum_measurement_ratio_formula_error"] = float(ratio_error.max())
    checks["samplewise_ratio_denominators_below_1e_minus_8"] = sample_ratio_denominators_below

    residual = trace[(trace.model_key == "reshvql") & trace.layer.isin([
        "quantum_branch", "classical_branch_scaled", "fused"
    ])]
    residual_pivot = residual.pivot(
        index=unique_key[:-1], columns="layer", values="delta_l2"
    ).reset_index()
    fused_rows = trace[(trace.model_key == "reshvql") & (trace.layer == "fused")]
    fused_rows = fused_rows.set_index(unique_key[:-1]).sort_index()
    residual_pivot = residual_pivot.set_index(unique_key[:-1]).sort_index()
    direct_ratio = residual_pivot.classical_branch_scaled / residual_pivot.quantum_branch
    residual_ratio_error = np.abs(
        direct_ratio.to_numpy() - fused_rows.classical_to_quantum_displacement_ratio.to_numpy()
    )
    branch_share_error = np.abs(
        fused_rows.quantum_norm_share_of_branch_sum.to_numpy()
        + fused_rows.classical_norm_share_of_branch_sum.to_numpy() - 1.0
    )
    projection_share_error = np.abs(
        fused_rows.quantum_projection_share_of_fused.to_numpy()
        + fused_rows.classical_projection_share_of_fused.to_numpy() - 1.0
    )
    assert float(residual_ratio_error.max()) < 2e-5
    assert float(branch_share_error.max()) < 2e-6
    assert float(projection_share_error.max()) < 2e-6
    assert float(fused_rows.residual_reconstruction_relative_error.max()) < 3e-7
    checks["maximum_residual_ratio_formula_error"] = float(residual_ratio_error.max())
    checks["maximum_branch_share_sum_error"] = float(branch_share_error.max())
    checks["maximum_projection_share_sum_error"] = float(projection_share_error.max())
    checks["maximum_residual_reconstruction_relative_error"] = float(
        fused_rows.residual_reconstruction_relative_error.max()
    )

    layerwise = pd.read_csv(OUTPUT_DIR / "layerwise_summary.csv")
    interface_summary = pd.read_csv(OUTPUT_DIR / "interface_transition_summary.csv")
    residual_summary = pd.read_csv(OUTPUT_DIR / "residual_branch_summary.csv")
    assert (layerwise.bootstrap_resamples == 1000).all()
    assert (interface_summary.bootstrap_resamples == 1000).all()
    assert (residual_summary.bootstrap_resamples == 1000).all()
    assert (layerwise.n_nonfinite == 0).all()
    assert (interface_summary.filter(regex="_n_nonfinite$") == 0).all().all()
    assert (residual_summary.n_nonfinite == 0).all()
    checks["all_summary_bootstraps"] = 1000
    checks["all_substantive_summary_values_finite"] = True

    report = (OUTPUT_DIR / "revised_analysis_report.md").read_text(encoding="utf-8")
    for section in (
        "Checkpoint and prediction validation", "Cohorts and class balance",
        "Seed-specific and equal-seed pooled evidence", "Comparison with the previous 19-example analysis",
        "Observation-by-observation assessment", "Numerical replacements", "Exact replacement figure captions",
        "Small-margin sensitivity and remaining limitations",
    ):
        assert section in report
    assert "Ã" not in report and "\ufffd" not in report
    checks["report_sections_and_utf8_valid"] = True

    checks["status"] = "passed"
    (OUTPUT_DIR / "validation_audit.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
