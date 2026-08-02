"""Validate the final trace-derived thesis figure package.

This utility intentionally uses fixed project paths: it is a reproducibility
companion for one analytical export, not a general command-line interface.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figure_redesign_from_trace"
FIGURE_DIR = OUTPUT_ROOT / "followup_figures"
TRACE_PATH = PROJECT_ROOT / "outputs" / "mechanistic_visualizations" / "trace_layerwise_samples.csv"
ATTACK_ROOT = PROJECT_ROOT / "logs" / "ATTACKS" / "attack_experiment_canonical"
HVQL_BASELINE = "Hybrid__HVQL__ansatz=strong,_qdepth=50"
MARGIN_TAU = 1e-3
MARGIN_THRESHOLD = 1e-3


def _clean_margin(seed: int, indices: np.ndarray) -> np.ndarray:
    labels = np.load(ATTACK_ROOT / f"seed_{seed}" / "labels.npy")
    logits = np.load(
        ATTACK_ROOT / f"seed_{seed}" / "baselines" / HVQL_BASELINE / "logits_clean.npy"
    ).astype(np.float64)
    y = labels[indices].astype(int)
    selected = logits[indices]
    competitors = selected.copy()
    competitors[np.arange(len(indices)), y] = -np.inf
    return selected[np.arange(len(indices)), y] - competitors.max(axis=1)


def _independent_hvql_checks() -> dict[str, float]:
    trace = pd.read_csv(TRACE_PATH)
    trace = trace[
        (trace["model_key"] == "hvql")
        & np.isclose(trace["epsilon"], 0.10)
        & (trace["condition"] == "auto-FGSM")
    ].copy()

    conv_seed_medians = trace[trace["layer"] == "conv1"].groupby("seed")["gain_rms"].median()
    gain_expected = float(conv_seed_medians.median())

    measure = trace[trace["layer"] == "measurement"].copy()
    normalized_seed_medians = []
    for seed, group in measure.groupby("seed"):
        margins = _clean_margin(int(seed), group["dataset_index"].to_numpy(int))
        covered = margins >= MARGIN_THRESHOLD
        values = group["projected_margin_harm"].to_numpy(float)[covered] / (margins[covered] + MARGIN_TAU)
        normalized_seed_medians.append(float(np.median(values)))
    harm_expected = float(np.median(normalized_seed_medians))

    plotted = pd.read_csv(OUTPUT_ROOT / "followup_figure_data" / "hvql_strong_d50_dual_progression.csv")
    gain_actual = float(plotted[(plotted["metric"] == "gain_rms") & (plotted["layer"] == "conv1")]["estimate"].iloc[0])
    harm_actual = float(
        plotted[
            (plotted["metric"] == "margin_normalized_harm") & (plotted["layer"] == "measurement")
        ]["estimate"].iloc[0]
    )
    if not math.isclose(gain_actual, gain_expected, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f"Gain HVQL Conv. 1 divergent: {gain_actual} != {gain_expected}")
    if not math.isclose(harm_actual, harm_expected, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f"Dommage HVQL mesure divergent: {harm_actual} != {harm_expected}")
    return {
        "hvql_conv1_gain_expected": gain_expected,
        "hvql_conv1_gain_plotted": gain_actual,
        "hvql_measurement_harm_expected": harm_expected,
        "hvql_measurement_harm_plotted": harm_actual,
    }


def _validate_figure_files(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in manifest.to_dict("records"):
        figure_id = str(record["figure_id"])
        paths = {
            "png": OUTPUT_ROOT / str(record["png_filename"]),
            "pdf": OUTPUT_ROOT / str(record["pdf_filename"]),
            "svg": OUTPUT_ROOT / str(record["svg_filename"]),
            "csv": OUTPUT_ROOT / str(record["source_csv"]),
            "script": OUTPUT_ROOT / str(record["plotting_script"]),
            "method": OUTPUT_ROOT / str(record["method_file"]),
            "caption": OUTPUT_ROOT / str(record["caption_file"]),
        }
        missing = [kind for kind, path in paths.items() if not path.exists() or path.stat().st_size == 0]
        if missing:
            raise AssertionError(f"{figure_id}: artefacts absents ou vides: {missing}")
        with Image.open(paths["png"]) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            min_dpi = float(min(dpi)) if dpi else 0.0
            width, height = image.size
        if min_dpi < 295.0:
            raise AssertionError(f"{figure_id}: PNG sous 300 dpi ({dpi})")
        if width < 1000 or height < 700:
            raise AssertionError(f"{figure_id}: dimensions PNG insuffisantes ({width}x{height})")
        if not paths["pdf"].read_bytes().startswith(b"%PDF"):
            raise AssertionError(f"{figure_id}: signature PDF invalide")
        svg_text = paths["svg"].read_text(encoding="utf-8")
        if "<svg" not in svg_text:
            raise AssertionError(f"{figure_id}: contenu SVG invalide")
        source = pd.read_csv(paths["csv"])
        if source.empty:
            raise AssertionError(f"{figure_id}: CSV tracé vide")
        script_text = paths["script"].read_text(encoding="utf-8")
        if figure_id not in script_text or "followup_figure_data" not in script_text or "replot_figure" not in script_text:
            raise AssertionError(f"{figure_id}: script de retraçage non relié au CSV/nom exact")
        method_text = paths["method"].read_text(encoding="utf-8")
        required_terms = ["Mesure directe", "Lecture visible", "Limites", "Cohorte"]
        missing_terms = [term for term in required_terms if term not in method_text]
        if missing_terms:
            raise AssertionError(f"{figure_id}: méthode incomplète: {missing_terms}")
        rows.append(
            {
                "figure_id": figure_id,
                "png_width": width,
                "png_height": height,
                "png_dpi_x": float(dpi[0]),
                "png_dpi_y": float(dpi[1]),
                "csv_rows": int(len(source)),
                "pdf_valid": True,
                "svg_valid": True,
                "companions_complete": True,
            }
        )
    return pd.DataFrame(rows)


def _validate_report(manifest: pd.DataFrame) -> dict[str, object]:
    report_path = OUTPUT_ROOT / "report_followup.html"
    receipt_path = OUTPUT_ROOT / "html_delivery_receipt_followup.json"
    html = report_path.read_text(encoding="utf-8")
    required_sections = [
        "Synthèse technique des corrections", "L’audit métrique", "HVQL rend visible",
        "Le CNN concentre", "Les trajectoires secondaires", "Les interfaces tardives",
        "L’efficacité décisionnelle", "La variation récupérable", "La vue toutes architectures",
        "L’alignement explique", "Six recommandations", "Candidats d’annexe",
        "Figures rejetées ou trompeuses", "Galerie complète", "Tables complètes", "Inventaire de sortie",
    ]
    absent = [section for section in required_sections if section not in html]
    if absent:
        raise AssertionError(f"Sections HTML absentes: {absent}")
    # Exact SVG gallery items are sandboxed inside escaped iframe ``srcdoc``.
    gallery_count = html.count("&lt;figure&gt;")
    if gallery_count != len(manifest):
        raise AssertionError(f"Galerie HTML: {gallery_count} figures au lieu de {len(manifest)}")
    external_links = re.findall(r'(?:src|href)=["\']https?://', html, flags=re.IGNORECASE)
    if external_links:
        raise AssertionError("Le rapport portable contient des dépendances HTTP externes")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not receipt.get("ok"):
        raise AssertionError("Le reçu HTML n'est pas valide")
    stages = receipt.get("stages", {})
    if stages.get("validation") != "passed" or stages.get("package") != "passed":
        raise AssertionError(f"Contrat HTML incomplet: {stages}")
    return {
        "gallery_figure_count": gallery_count,
        "external_http_dependencies": len(external_links),
        "portable_validation": stages.get("validation"),
        "portable_package": stages.get("package"),
        "portable_verification": stages.get("verification"),
        "browser_warning": receipt.get("browserWarning", {}).get("code"),
    }


def _validate_tables(manifest: pd.DataFrame) -> dict[str, object]:
    audit = pd.read_csv(OUTPUT_ROOT / "metric_definition_audit.csv")
    expected_metrics = {
        "Déplacement RMS brut", "Gain RMS normalisé à l’entrée", "Gain RMS bloc-à-bloc",
        "Dommage de marge brut", "Dommage normalisé par la marge propre", "Déplacement relatif",
        "Sensibilité du gradient", "Alignement nuisible", "Efficacité du dommage",
    }
    if set(audit["metric_name"]) != expected_metrics:
        raise AssertionError("L’audit métrique ne couvre pas exactement les neuf définitions attendues")
    normalized = audit[audit["metric_name"] == "Dommage normalisé par la marge propre"].iloc[0]
    if "projected_margin_harm / (clean_margin + 1e-3)" not in normalized["source_column"]:
        raise AssertionError("La normalisation du dommage n’est pas explicitement dérivée au niveau échantillon")
    coverage = pd.read_csv(OUTPUT_ROOT / "margin_normalization_coverage.csv")
    if float(coverage["coverage"].min()) < 0.998:
        raise AssertionError("Couverture de normalisation sous le seuil documenté de 99,8 %")
    recommendations = pd.read_csv(OUTPUT_ROOT / "followup_figure_recommendations.csv")
    if len(recommendations) != 6 or recommendations["rank"].tolist() != [1, 2, 3, 4, 5, 6]:
        raise AssertionError("La sélection finale doit contenir exactement six recommandations ordonnées")
    unavailable = pd.read_csv(OUTPUT_ROOT / "followup_unavailable_figures.csv")
    if len(unavailable) != 3:
        raise AssertionError("Les trois groupes indisponibles doivent être documentés")
    generated_ids = set(manifest["figure_id"])
    forbidden_prefixes = ("hvql_ansatz_", "hvql_depth_")
    if any(item.startswith(forbidden_prefixes) for item in generated_ids):
        raise AssertionError("Des figures ansatz/profondeur non étayées ont été générées")
    notebook = json.loads(
        (PROJECT_ROOT / "src" / "analytics" / "attacks" / "trace_figure_followup.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    if len(code_cells) != 4:
        raise AssertionError("Le notebook analytique doit conserver exactement quatre blocs pratiques")

    decision = pd.read_csv(
        OUTPUT_ROOT / "followup_figure_data" / "decision_routing_harm_efficiency_comparison.csv"
    )
    if not decision["comparison_label"].equals(decision["model"]):
        raise AssertionError("Les étiquettes du routage décisionnel doivent contenir uniquement les architectures")

    decision_harm = pd.read_csv(
        OUTPUT_ROOT / "followup_figure_data" / "decision_routing_normalized_harm_comparison.csv"
    )
    expected_harm_labels = decision_harm["model"].copy()
    expected_harm_labels.loc[decision_harm["model_key"] == "hpvql5"] = "HPVQL"
    expected_harm_labels.loc[decision_harm["model_key"] == "reshvql"] = "Résiduel"
    if not decision_harm["comparison_label"].equals(expected_harm_labels):
        raise AssertionError("Le dommage décisionnel doit afficher les architectures seules et l’étiquette HPVQL courte")

    contracts = json.loads((OUTPUT_ROOT / "followup_plot_contracts.json").read_text(encoding="utf-8"))
    late_ids = (
        "late_hybrid_interface_rms_progression",
        "late_hybrid_interface_normalized_harm_progression",
    )
    for figure_id in late_ids:
        contract = contracts[figure_id]
        if contract.get("xticks") != [0.0, 1.0, 2.0] or contract.get("xtick_labels") != [
            "Pré-interface", "Mesures", "Logits"
        ]:
            raise AssertionError(f"{figure_id}: les trois graduations demandées ne sont pas définies")
        if contract.get("xlabel") != "":
            raise AssertionError(f"{figure_id}: le titre de l’axe horizontal doit être supprimé")
        late = pd.read_csv(OUTPUT_ROOT / "followup_figure_data" / f"{figure_id}.csv")
        short_labels = dict(zip(late["model_key"], late["model"]))
        if short_labels.get("hpvql5") != "HPVQL" or short_labels.get("reshvql") != "Résiduel":
            raise AssertionError(f"{figure_id}: les légendes HPVQL/Résiduel ne sont pas raccourcies")
        merge = late[(late["model_key"] == "reshvql") & (late["layer"] == "residual merge")]
        if len(merge) != 1 or not math.isclose(float(merge["stage_order"].iloc[0]), 1.5):
            raise AssertionError(f"{figure_id}: la fusion résiduelle n’est pas au point intermédiaire 1,5")
    if contracts["late_hybrid_interface_normalized_harm_progression"].get("yaxis_side") != "left":
        raise AssertionError("L’axe vertical du dommage normalisé doit être explicitement placé à gauche")

    pair_id = "hvql_cnn_dual_progression_side_by_side"
    pair = pd.read_csv(OUTPUT_ROOT / "followup_figure_data" / f"{pair_id}.csv")
    if set(pair["model_key"]) != {"hvql", "cnn"} or set(pair["metric"]) != {
        "gain_rms", "margin_normalized_harm"
    }:
        raise AssertionError("La figure juxtaposée doit contenir exactement HVQL/CNN et les deux métriques")
    pair_contract = contracts.get(pair_id, {})
    if pair_contract.get("panel_order") != ["hvql", "cnn"]:
        raise AssertionError("La figure juxtaposée doit placer HVQL à gauche et CNN à droite")
    if pair_contract.get("panel_labels") != ["HVQL", "CNN"]:
        raise AssertionError("Les deux panneaux doivent être explicitement étiquetés HVQL et CNN")
    if pair_contract.get("xlabel") != "" or pair_contract.get("outer_y_axes_only") is not True:
        raise AssertionError("La figure juxtaposée doit supprimer les titres x et conserver seulement les axes y externes")
    expected_pair_stages = {
        "hvql": [
            "Entrée", "Conv. 1", "Pool 1", "Conv. 2", "Pool 2", "Dense 1",
            "Interface Quantique", "Mesures Quantiques", "Dense 2", "Logits",
        ],
        "cnn": ["Entrée", "Conv. 1", "Pool 1", "Conv. 2", "Pool 2", "Dense", "Logits"],
    }
    for model_key, expected in expected_pair_stages.items():
        observed = (
            pair[(pair["model_key"] == model_key) & (pair["metric"] == "gain_rms")]
            .sort_values("stage_order")["stage_label"].tolist()
        )
        if observed != expected:
            raise AssertionError(f"Progression {model_key} divergente: {observed}")
    return {
        "metric_definitions": int(len(audit)),
        "normalization_coverage_min": float(coverage["coverage"].min()),
        "recommendation_count": int(len(recommendations)),
        "unavailable_group_count": int(len(unavailable)),
        "notebook_code_cells": int(len(code_cells)),
        "decision_labels_architecture_only": True,
        "decision_harm_hpvql_label": "HPVQL",
        "decision_harm_residual_label": "Résiduel",
        "late_hybrid_ticks": ["Pré-interface", "Mesures", "Logits"],
        "late_hybrid_xlabel": "",
        "late_hybrid_short_labels": ["HPVQL", "Résiduel"],
        "late_hybrid_harm_yaxis_side": "left",
        "residual_merge_position": 1.5,
        "paired_dual_panel_order": ["hvql", "cnn"],
        "paired_dual_panel_labels": ["HVQL", "CNN"],
        "paired_dual_outer_y_axes_only": True,
        "paired_dual_stage_counts": {"hvql": 10, "cnn": 7},
    }


def _residual_ratio_check() -> float:
    branch = pd.read_csv(OUTPUT_ROOT / "followup_figure_data" / "residual_hvql_branch_rms_comparison.csv")
    quantum = float(branch[branch["branch_fr"] == "Branche quantique"]["estimate"].iloc[0])
    bypass = float(branch[branch["branch_fr"] == "Contournement classique"]["estimate"].iloc[0])
    ratio = bypass / quantum
    if not 17.0 < ratio < 17.6:
        raise AssertionError(f"Rapport contournement/quantique inattendu: {ratio}")
    return ratio


def _write_inventory() -> None:
    rows = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(OUTPUT_ROOT).as_posix(),
                    "extension": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "followup_output_inventory.csv", index=False)


def main() -> None:
    manifest = pd.read_csv(OUTPUT_ROOT / "followup_figure_manifest.csv")
    if len(manifest) != 51:
        raise AssertionError(f"51 figures attendues, {len(manifest)} trouvées")
    file_qa = _validate_figure_files(manifest)
    formula_qa = _independent_hvql_checks()
    table_qa = _validate_tables(manifest)
    report_qa = _validate_report(manifest)
    residual_ratio = _residual_ratio_check()

    file_qa.to_csv(OUTPUT_ROOT / "followup_figure_file_qa.csv", index=False)
    result = {
        "ok": True,
        "figure_count": int(len(manifest)),
        "triplet_count": int(len(file_qa)),
        "companions_complete": bool(file_qa["companions_complete"].all()),
        "minimum_png_dpi": float(file_qa[["png_dpi_x", "png_dpi_y"]].min().min()),
        "formula_checks": formula_qa,
        "table_checks": table_qa,
        "report_checks": report_qa,
        "residual_bypass_to_quantum_rms_ratio": residual_ratio,
        "limitations": [
            "La vérification HTML est structurelle: aucun exécutable Chromium n’était disponible.",
            "Les comparaisons d’interface et de routage héritées reposent sur 19 exemples communs-corrects.",
            "Les variations HVQL d’ansatz et de profondeur ne sont pas récupérables depuis les traces couche-par-couche présentes.",
        ],
    }
    (OUTPUT_ROOT / "followup_package_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    destination = OUTPUT_ROOT / "followup_scripts" / Path(__file__).name
    if destination.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, destination)
    _write_inventory()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
