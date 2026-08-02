"""Assemble the canonical portable HTML report for the figure redesign.

The Data Analytics packaged builder consumes the resulting ``artifact.json``.
This script does not hand-author a second report runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sqlite3

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figure_redesign_from_trace"
ARTIFACT_PATH = OUTPUT_DIR / "artifact.json"
DATABASE_PATH = OUTPUT_DIR / "report_source.sqlite"
GENERATED_AT = "2026-07-27T04:30:00-04:00"


SOURCE_SQL = {
    "src_hvql_rms": "SELECT * FROM hvql_rms ORDER BY stage_order",
    "src_grouped_harm": "SELECT * FROM grouped_harm ORDER BY model_key, stage_order",
    "src_interface": "SELECT * FROM interface ORDER BY model_order",
    "src_decision": "SELECT * FROM decision_stage ORDER BY estimate DESC",
    "src_segmentation": "SELECT * FROM segmentation ORDER BY model_key, stage_order",
    "src_residual": "SELECT * FROM residual ORDER BY \"order\"",
    "src_stage_mapping": "SELECT * FROM stage_mapping ORDER BY model",
    "src_progression_summary": "SELECT * FROM progression_summary ORDER BY model",
    "src_grouped_summary": "SELECT * FROM grouped_summary ORDER BY comparison_figure, model_key, stage_order",
    "src_recommendations": "SELECT * FROM recommendations ORDER BY figure_filename",
    "src_unavailable": "SELECT * FROM unavailable ORDER BY requested_group",
    "src_manifest": "SELECT * FROM figure_manifest ORDER BY figure_id",
    "src_inventory": "SELECT * FROM output_inventory ORDER BY path",
    "src_source_inventory": "SELECT * FROM source_inventory ORDER BY source_key",
    "src_checkpoint": "SELECT * FROM checkpoint_validation ORDER BY model_key, seed",
}


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.replace({np.nan: None})
    return json.loads(clean.to_json(orient="records", force_ascii=False))


def _manifest_source(source_id: str, label: str, path: str) -> dict[str, str]:
    return {"id": source_id, "label": label, "path": path}


def _query_source(
    source_id: str,
    description: str,
    tables_used: list[str],
    filters: dict[str, str],
) -> dict[str, object]:
    return {
        "id": source_id,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": SOURCE_SQL[source_id],
            "description": description,
            "tables_used": tables_used,
            "filters": filters,
            "executed_at": GENERATED_AT,
        },
    }


def _prefix_svg_ids(svg: str, prefix: str) -> str:
    identifiers = set(re.findall(r'id="([^"]+)"', svg))
    for identifier in sorted(identifiers, key=len, reverse=True):
        replacement = f"{prefix}_{identifier}"
        svg = svg.replace(f'id="{identifier}"', f'id="{replacement}"')
        svg = svg.replace(f"url(#{identifier})", f"url(#{replacement})")
        svg = svg.replace(f'href="#{identifier}"', f'href="#{replacement}"')
        svg = svg.replace(f'xlink:href="#{identifier}"', f'xlink:href="#{replacement}"')
    return svg


def _inline_svg(path: Path, prefix: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(<svg\b.*?</svg>)", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No SVG root in {path}")
    svg = re.sub(r"<metadata>.*?</metadata>", "", match.group(1), flags=re.DOTALL)
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.DOTALL)
    svg = _prefix_svg_ids(svg, prefix)
    return svg.replace("<svg ", '<svg style="width:100%;height:auto;display:block" ' , 1)


def _gallery_blocks(figure_manifest: pd.DataFrame) -> list[dict[str, object]]:
    css = """
<style>
.trace-gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;font-family:system-ui,sans-serif}
.trace-gallery figure{margin:0;padding:12px;border:1px solid #d7dde5;border-radius:10px;background:#fff;color:#18202a}
.trace-gallery figcaption{margin-top:8px;font:600 12px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}
@media(max-width:760px){.trace-gallery{grid-template-columns:1fr}}
@media(prefers-color-scheme:dark){.trace-gallery figure{background:#171a1f;color:#eef1f5;border-color:#3b424d}}
</style>
""".strip()
    blocks: list[dict[str, object]] = []
    ordered = figure_manifest.reset_index(drop=True)
    chunk_size = 10
    for chunk_index in range(0, len(ordered), chunk_size):
        chunk = ordered.iloc[chunk_index : chunk_index + chunk_size]
        parts = [css, '<div class="trace-gallery">']
        for row_index, row in chunk.iterrows():
            svg_path = OUTPUT_DIR / str(row["svg_filename"])
            svg = _inline_svg(svg_path, f"g{chunk_index}_{row_index}")
            label = str(row["filename"])
            parts.append(f"<figure>{svg}<figcaption>{label}</figcaption></figure>")
        parts.append("</div>")
        blocks.append({
            "id": f"gallery_{chunk_index // chunk_size + 1}",
            "type": "html",
            "body": "\n".join(parts),
        })
    return blocks


def _load_tables() -> dict[str, pd.DataFrame]:
    data_dir = OUTPUT_DIR / "figure_data"
    tables = {
        "hvql_rms": pd.read_csv(data_dir / "hvql_d50_rms_progression.csv"),
        "grouped_harm": pd.read_csv(data_dir / "grouped_late_hybrids_margin_harm_progression.csv"),
        "interface": pd.read_csv(data_dir / "interface_selectivity_transition_comparison.csv"),
        "decision_stage": pd.read_csv(data_dir / "decision_stage_comparison_rms.csv"),
        "segmentation": pd.read_csv(data_dir / "hpvql_blocks_rms_progression.csv"),
        "residual": pd.read_csv(data_dir / "residual_branch_rms_comparison.csv"),
        "stage_mapping": pd.read_csv(OUTPUT_DIR / "per_model_stage_mapping.csv"),
        "progression_summary": pd.read_csv(OUTPUT_DIR / "per_model_progression_summary.csv"),
        "grouped_summary": pd.read_csv(OUTPUT_DIR / "grouped_comparison_summary.csv"),
        "recommendations": pd.read_csv(OUTPUT_DIR / "figure_recommendation_summary.csv"),
        "unavailable": pd.read_csv(OUTPUT_DIR / "unavailable_requested_figures.csv"),
        "figure_manifest": pd.read_csv(OUTPUT_DIR / "figure_manifest.csv"),
        "source_inventory": pd.read_csv(OUTPUT_DIR / "source_inventory.csv"),
        "checkpoint_validation": pd.read_csv(
            PROJECT_ROOT / "outputs" / "mechanistic_visualizations" / "checkpoint_alignment.csv"
        ),
    }
    inventory_path = OUTPUT_DIR / "output_inventory.csv"
    inventory = pd.read_csv(inventory_path) if inventory_path.exists() else pd.DataFrame(columns=["path", "suffix", "bytes"])
    expected = pd.DataFrame([
        {"path": "artifact.json", "suffix": ".json", "bytes": np.nan},
        {"path": "report.html", "suffix": ".html", "bytes": np.nan},
        {"path": "report_source.sqlite", "suffix": ".sqlite", "bytes": np.nan},
        {"path": "html_delivery_receipt.json", "suffix": ".json", "bytes": np.nan},
        {"path": "figure_file_qa.csv", "suffix": ".csv", "bytes": np.nan},
        {"path": "package_validation.json", "suffix": ".json", "bytes": np.nan},
        {"path": "scripts/build_trace_figure_redesign_report.py", "suffix": ".py", "bytes": np.nan},
        {"path": "scripts/validate_trace_figure_redesign.py", "suffix": ".py", "bytes": np.nan},
    ])
    tables["output_inventory"] = (
        pd.concat([inventory, expected], ignore_index=True)
        .drop_duplicates("path", keep="last")
        .sort_values("path")
        .reset_index(drop=True)
    )
    return tables


def _chart_specs() -> list[dict[str, object]]:
    return [
        {
            "id": "chart_hvql_rms",
            "title": "Progression RMS dans HVQL",
            "subtitle": "Médiane de deux graines; intervalle interquartile dans le CSV associé; FGSM blanche, ε = 0,10.",
            "type": "line",
            "dataset": "hvql_rms",
            "sourceId": "src_hvql_rms",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "stage_label", "type": "ordinal", "label": "Bloc architectural"},
                "y": {"field": "estimate", "type": "quantitative", "label": "Déplacement RMS"},
                "tooltip": [
                    {"field": "estimate", "type": "quantitative", "label": "Médiane", "format": "number"},
                    {"field": "q25", "type": "quantitative", "label": "Q1", "format": "number"},
                    {"field": "q75", "type": "quantitative", "label": "Q3", "format": "number"},
                ],
            },
        },
        {
            "id": "chart_grouped_harm",
            "title": "Dommage normalisé des hybrides tardifs",
            "subtitle": "Étapes alignées par rôle; le vide HVQM indique l’absence d’un encodeur profond distinct.",
            "type": "line",
            "dataset": "grouped_harm",
            "sourceId": "src_grouped_harm",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "stage_label", "type": "ordinal", "label": "Rôle architectural"},
                "y": {"field": "estimate", "type": "quantitative", "label": "Dommage normalisé"},
                "color": {"field": "model", "type": "nominal", "label": "Architecture"},
                "tooltip": [
                    {"field": "model", "type": "nominal", "label": "Modèle"},
                    {"field": "estimate", "type": "quantitative", "label": "Médiane", "format": "number"},
                ],
            },
        },
        {
            "id": "chart_interface",
            "title": "Rétention RMS et variation du dommage à l’interface",
            "subtitle": "Quatre architectures intentionnellement étiquetées; HQE présente une forte variabilité de graine.",
            "type": "scatter",
            "dataset": "interface",
            "sourceId": "src_interface",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "retention_rms", "type": "quantitative", "label": "Rétention RMS"},
                "y": {"field": "interface_harm_change", "type": "quantitative", "label": "Variation du dommage normalisé"},
                "color": {"field": "model", "type": "nominal", "label": "Architecture"},
                "tooltip": [
                    {"field": "model", "type": "nominal", "label": "Modèle"},
                    {"field": "logits_harm_concentration", "type": "quantitative", "label": "Dommage aux logits", "format": "number"},
                    {"field": "alignment_change", "type": "quantitative", "label": "Δ alignement", "format": "number"},
                    {"field": "n_samples", "type": "quantitative", "label": "Observations"},
                ],
            },
        },
        {
            "id": "chart_decision_stage",
            "title": "RMS au bloc décisionnel",
            "subtitle": "Comparaison descriptive sur 19 exemples communs; les dimensions et les gradients aval diffèrent.",
            "type": "bar",
            "dataset": "decision_stage",
            "sourceId": "src_decision",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Bloc décisionnel"},
                "y": {"field": "estimate", "type": "quantitative", "label": "Déplacement RMS"},
                "tooltip": [
                    {"field": "layer", "type": "nominal", "label": "Couche tracée"},
                    {"field": "q25", "type": "quantitative", "label": "Q1", "format": "number"},
                    {"field": "q75", "type": "quantitative", "label": "Q3", "format": "number"},
                ],
            },
        },
        {
            "id": "chart_segmentation",
            "title": "Progression RMS selon le nombre de blocs HPVQL",
            "subtitle": "Checkpoints entraînés séparément; le graphe ne constitue pas une ablation causale de la segmentation.",
            "type": "line",
            "dataset": "segmentation",
            "sourceId": "src_segmentation",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "stage_label", "type": "ordinal", "label": "Bloc architectural"},
                "y": {"field": "estimate", "type": "quantitative", "label": "Déplacement RMS"},
                "color": {"field": "model", "type": "nominal", "label": "Architecture"},
                "tooltip": [
                    {"field": "model", "type": "nominal", "label": "Modèle"},
                    {"field": "estimate", "type": "quantitative", "label": "Médiane", "format": "number"},
                ],
            },
        },
        {
            "id": "chart_residual",
            "title": "Contributions RMS des branches du HVQL résiduel",
            "subtitle": "Contribution de la branche quantique, du contournement classique mis à l’échelle et de la fusion.",
            "type": "bar",
            "dataset": "residual",
            "sourceId": "src_residual",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "branch_fr", "type": "nominal", "label": "Branche"},
                "y": {"field": "estimate", "type": "quantitative", "label": "Contribution RMS"},
                "tooltip": [
                    {"field": "q25", "type": "quantitative", "label": "Q1", "format": "number"},
                    {"field": "q75", "type": "quantitative", "label": "Q3", "format": "number"},
                ],
            },
        },
    ]


def _table_specs() -> list[dict[str, object]]:
    return [
        {
            "id": "table_stage_mapping",
            "title": "Correspondance entre couches tracées et étapes finales",
            "subtitle": "Une ligne par modèle; les étiquettes conservent les blocs réellement exécutés.",
            "dataset": "stage_mapping",
            "sourceId": "src_stage_mapping",
            "defaultSort": {"field": "model", "direction": "asc"},
            "columns": [
                {"field": "model", "label": "Modèle", "type": "text"},
                {"field": "raw_traced_layers", "label": "Couches tracées", "type": "text"},
                {"field": "grouped_stages_used_in_final_figures", "label": "Étapes affichées", "type": "text"},
                {"field": "evidence_scope", "label": "Portée", "type": "text"},
            ],
        },
        {
            "id": "table_progression_summary",
            "title": "Synthèse de progression par modèle",
            "subtitle": "Les changements RMS sont relatifs à l’entrée; les valeurs d’interface ne sont renseignées que lorsqu’elles existent.",
            "dataset": "progression_summary",
            "sourceId": "src_progression_summary",
            "defaultSort": {"field": "model", "direction": "asc"},
            "columns": [
                {"field": "model", "label": "Modèle", "type": "text"},
                {"field": "first_block_rms_ratio", "label": "Premier bloc / entrée", "format": "number"},
                {"field": "maximum_rms_ratio_to_input", "label": "Maximum / entrée", "format": "number"},
                {"field": "pre_interface_rms", "label": "RMS pré-interface", "format": "number"},
                {"field": "measured_rms", "label": "RMS mesurée", "format": "number"},
                {"field": "final_margin_harm", "label": "Dommage final", "format": "number"},
                {"field": "recommended_interpretation", "label": "Interprétation prudente", "type": "text"},
            ],
        },
        {
            "id": "table_grouped_summary",
            "title": "Valeurs utilisées dans les comparaisons groupées",
            "subtitle": "Table longue complète; chaque ligne correspond à une étape et à une métrique effectivement tracées.",
            "dataset": "grouped_summary",
            "sourceId": "src_grouped_summary",
            "defaultSort": {"field": "comparison_figure", "direction": "asc"},
            "columns": [
                {"field": "comparison_figure", "label": "Comparaison", "type": "text"},
                {"field": "model", "label": "Modèle", "type": "text"},
                {"field": "stage_label", "label": "Étape", "type": "text"},
                {"field": "metric", "label": "Métrique", "type": "text"},
                {"field": "estimate", "label": "Médiane", "format": "number"},
                {"field": "q25", "label": "Q1", "format": "number"},
                {"field": "q75", "label": "Q3", "format": "number"},
                {"field": "n_samples", "label": "Observations", "format": "number"},
                {"field": "n_seeds", "label": "Graines", "format": "number"},
            ],
        },
        {
            "id": "table_recommendations",
            "title": "Sélection des trois figures principales",
            "subtitle": "Une figure recommandée pour la progression, la sélectivité et la comparaison architecturale.",
            "dataset": "recommendations",
            "sourceId": "src_recommendations",
            "defaultSort": {"field": "figure_filename", "direction": "asc"},
            "columns": [
                {"field": "figure_filename", "label": "Figure", "type": "text"},
                {"field": "concept", "label": "Concept", "type": "text"},
                {"field": "strengths", "label": "Forces", "type": "text"},
                {"field": "weaknesses", "label": "Faiblesses", "type": "text"},
                {"field": "recommended_use", "label": "Usage recommandé", "type": "text"},
            ],
        },
        {
            "id": "table_unavailable",
            "title": "Figures demandées non récupérables",
            "subtitle": "Aucune valeur n’a été imputée lorsque le traçage nécessaire était absent.",
            "dataset": "unavailable",
            "sourceId": "src_unavailable",
            "defaultSort": {"field": "requested_group", "direction": "asc"},
            "columns": [
                {"field": "requested_group", "label": "Groupe", "type": "text"},
                {"field": "status", "label": "Statut", "type": "text"},
                {"field": "reason", "label": "Raison", "type": "text"},
                {"field": "needed_evidence", "label": "Évidence requise", "type": "text"},
            ],
        },
        {
            "id": "table_manifest",
            "title": "Manifest des figures",
            "subtitle": "Une ligne par figure indépendante; chaque entrée possède PDF, SVG, PNG, CSV et méthode.",
            "dataset": "figure_manifest",
            "sourceId": "src_manifest",
            "defaultSort": {"field": "figure_id", "direction": "asc"},
            "columns": [
                {"field": "figure_id", "label": "Figure", "type": "text"},
                {"field": "concept", "label": "Concept", "type": "text"},
                {"field": "models", "label": "Modèles", "type": "text"},
                {"field": "metric", "label": "Métrique", "type": "text"},
                {"field": "recommended_use", "label": "Usage", "type": "text"},
                {"field": "filename", "label": "PNG", "type": "text"},
                {"field": "pdf_filename", "label": "PDF", "type": "text"},
                {"field": "svg_filename", "label": "SVG", "type": "text"},
                {"field": "data_csv", "label": "CSV", "type": "text"},
                {"field": "method_file", "label": "Méthode", "type": "text"},
            ],
        },
        {
            "id": "table_inventory",
            "title": "Inventaire complet de sortie",
            "subtitle": "Fichiers exportés ou attendus par le contrat de livraison portable.",
            "dataset": "output_inventory",
            "sourceId": "src_inventory",
            "defaultSort": {"field": "path", "direction": "asc"},
            "columns": [
                {"field": "path", "label": "Chemin", "type": "text"},
                {"field": "suffix", "label": "Format", "type": "text"},
                {"field": "bytes", "label": "Octets", "format": "number"},
            ],
        },
    ]


def main() -> None:
    tables = _load_tables()
    # The grouped table intentionally preserves all rows used by every grouped
    # figure, including explicit NA gaps such as the missing HVQM deep encoder.
    with sqlite3.connect(DATABASE_PATH) as connection:
        for table_name, frame in tables.items():
            frame.to_sql(table_name, connection, if_exists="replace", index=False)
        for sql in SOURCE_SQL.values():
            pd.read_sql_query(sql, connection)

    manifest_sources = [
        _manifest_source("src_hvql_rms", "Trajectoire RMS HVQL", "outputs/figure_redesign_from_trace/figure_data/hvql_d50_rms_progression.csv"),
        _manifest_source("src_grouped_harm", "Comparaison groupée du dommage", "outputs/figure_redesign_from_trace/figure_data/grouped_late_hybrids_margin_harm_progression.csv"),
        _manifest_source("src_interface", "Synthèse de l’interface", "outputs/figure_redesign_from_trace/figure_data/interface_selectivity_transition_comparison.csv"),
        _manifest_source("src_decision", "Comparaison des blocs décisionnels", "outputs/figure_redesign_from_trace/figure_data/decision_stage_comparison_rms.csv"),
        _manifest_source("src_segmentation", "Trajectoires HPVQL", "outputs/figure_redesign_from_trace/figure_data/hpvql_blocks_rms_progression.csv"),
        _manifest_source("src_residual", "Branches du HVQL résiduel", "outputs/figure_redesign_from_trace/figure_data/residual_branch_rms_comparison.csv"),
        _manifest_source("src_stage_mapping", "Cartographie des étapes", "outputs/figure_redesign_from_trace/per_model_stage_mapping.csv"),
        _manifest_source("src_progression_summary", "Synthèse par modèle", "outputs/figure_redesign_from_trace/per_model_progression_summary.csv"),
        _manifest_source("src_grouped_summary", "Valeurs groupées", "outputs/figure_redesign_from_trace/grouped_comparison_summary.csv"),
        _manifest_source("src_recommendations", "Sélection des figures", "outputs/figure_redesign_from_trace/figure_recommendation_summary.csv"),
        _manifest_source("src_unavailable", "Demandes sans traçage", "outputs/figure_redesign_from_trace/unavailable_requested_figures.csv"),
        _manifest_source("src_manifest", "Manifest des figures", "outputs/figure_redesign_from_trace/figure_manifest.csv"),
        _manifest_source("src_inventory", "Inventaire des sorties", "outputs/figure_redesign_from_trace/output_inventory.csv"),
        _manifest_source("src_source_inventory", "Inventaire des sources", "outputs/figure_redesign_from_trace/source_inventory.csv"),
        _manifest_source("src_checkpoint", "Validation des checkpoints", "outputs/mechanistic_visualizations/checkpoint_alignment.csv"),
    ]

    figure_manifest = tables["figure_manifest"]
    blocks: list[dict[str, object]] = [
        {"id": "title", "type": "markdown", "body": "# Redesign des figures de propagation adversariale"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Synthèse technique\n\n"
                "**La progression est la plus lisible en RMS par dimension, complétée par le dommage de marge.** "
                "Le RMS fournit une trajectoire stable et comparable entre largeurs de représentation; le dommage indique quand le déplacement devient décisionnel. "
                "Le déplacement relatif est utile en annexe, mais son traçage sauvegardé repose sur seulement 19 exemples et manque pour le grand MLP.\n\n"
                "**Le résultat visuel le plus robuste n’est pas une atténuation uniforme.** CNN, ResNet18, MLPMixer et HVQL atténuent d’abord le RMS, puis des blocs plus tardifs inversent cette tendance. "
                "À l’interface tardive, HVQL, HPVQL et HVQM réduisent le RMS mesuré tout en augmentant l’alignement nuisible; HQE montre le cas extrême, mais avec une variabilité de graine trop forte pour être la preuve principale.\n\n"
                "**Trois figures doivent porter le récit principal.** La trajectoire RMS de HVQL explique la progression, la relation rétention–dommage explique la sélectivité, et la trajectoire groupée du dommage compare HVQL, HPVQL et HVQM par rôle architectural."
            ),
        },
        {
            "id": "previous_weakness",
            "type": "markdown",
            "body": (
                "## Pourquoi les figures précédentes étaient difficiles à utiliser\n\n"
                "Les anciens overlays imposaient un axe commun à des micro-couches non homologues, mélangeaient trop de familles et laissaient certaines séries disparaître. "
                "Le redesign sépare désormais trois tâches : une trajectoire propre à chaque modèle, un alignement explicite par rôle pour les comparaisons tardives, et des barres ciblées pour les blocs décisionnels. "
                "Aucun titre ni paragraphe explicatif n’est intégré aux fichiers de figure; l’interprétation reste dans le manuscrit et dans ce rapport."
            ),
        },
        {
            "id": "data_scope",
            "type": "markdown",
            "body": (
                "## Données, cohortes et définitions\n\n"
                "Le redesign ne recharge aucun checkpoint et ne recalcule aucune attaque. Les figures principales lisent `trace_layerwise_samples.csv`, `trace_interface_samples.csv` et `trace_segmentation_samples.csv`, avec 500 exemples correctement classés par graine pour les graines 0 et 42. "
                "Les variantes résiduelle et dropout, ainsi que le déplacement relatif, proviennent du diagnostic antérieur de 19 exemples à la graine 0; elles sont donc reléguées aux figures de soutien.\n\n"
                "Le RMS est `sqrt(mean_j Δa_j²)`. Le dommage de marge est `H = −∇m·Δa`; sa version normalisée divise par `|H_entrée| + 10⁻³`. "
                "L’alignement nuisible est `−cos(∇m, Δa)`. Les intervalles affichés sont des intervalles interquartiles d’observations appariées; ils ne constituent pas une significativité inter-entraînements."
            ),
        },
        {
            "id": "stage_strategy",
            "type": "markdown",
            "body": (
                "## Stratégie de regroupement des étapes\n\n"
                "Chaque figure individuelle conserve les vrais blocs du code. Les figures groupées n’alignent que des rôles défendables : entrée, encodeur initial, encodeur profond, pré-interface, mesures et logits. "
                "HVQM n’a pas d’encodeur profond distinct : l’espace correspondant reste vide, au lieu d’interpoler une étape inexistante. Pour le HVQL résiduel, la branche quantique et le contournement classique sont dessinés comme deux chemins parallèles avant la fusion."
            ),
        },
        {"id": "stage_mapping_table", "type": "table", "tableId": "table_stage_mapping"},
        {
            "id": "per_model_gallery",
            "type": "markdown",
            "sourceId": "src_hvql_rms",
            "body": (
                "## Une figure par modèle révèle les renversements internes\n\n"
                "La trajectoire HVQL montre une baisse au premier bloc, une remontée dans l’encodeur et l’interface dense, une réduction à la mesure, puis une forte croissance aux logits. "
                "Ce profil illustre pourquoi le premier bloc ne détermine pas seul la vulnérabilité finale et pourquoi la norme doit être accompagnée du dommage de marge. Les onze autres modèles disposent du même triplet de vues lorsque la métrique est récupérable."
            ),
        },
        {"id": "hvql_rms_chart", "type": "chart", "chartId": "chart_hvql_rms"},
        {
            "id": "grouped_late",
            "type": "markdown",
            "sourceId": "src_grouped_harm",
            "body": (
                "## Les hybrides tardifs divergent surtout à la mesure\n\n"
                "HVQL et HPVQL réduisent d’abord le dommage normalisé dans l’encodeur, puis le concentrent fortement à la mesure. HVQM reste plus bas dans cette métrique, malgré une responsabilité quantique plus directe. "
                "Le motif est compatible avec des interfaces et readouts différents; il ne démontre pas qu’une responsabilité quantique accrue protège ou fragilise."
            ),
        },
        {"id": "grouped_harm_chart", "type": "chart", "chartId": "chart_grouped_harm"},
        {
            "id": "decision_routing",
            "type": "markdown",
            "sourceId": "src_decision",
            "body": (
                "## Le routage décisionnel distingue quantum-dominant, quantum-assisted et bypass\n\n"
                "Sur la petite cohorte commune, les sorties quantiques HVQM et HVQL portent un RMS inférieur à l’activation dense du CNN. La fusion du HVQL résiduel est au contraire dominée par un déplacement beaucoup plus grand. "
                "Le dommage par unité de déplacement est maximal pour HVQL dans ce diagnostic, ce qui suggère une direction plus compatible avec la marge sans isoler la cause quantique."
            ),
        },
        {"id": "decision_chart", "type": "chart", "chartId": "chart_decision_stage"},
        {
            "id": "internal_variation",
            "type": "markdown",
            "sourceId": "src_segmentation",
            "body": (
                "## Les variations internes récupérables concernent la segmentation et le routage\n\n"
                "HPVQL à 5 et 10 blocs suit des trajectoires RMS proches jusqu’à l’interface, puis diverge légèrement aux mesures et aux logits. Cela visualise que le couplage quantique global complet n’est pas nécessaire au profil observé; la comparaison n’établit pas que la segmentation est protectrice. "
                "Les progressions strong/basic/random et profondeur 50/100/200 ne sont pas produites, car les traces couche par couche correspondantes n’existent pas."
            ),
        },
        {"id": "segmentation_chart", "type": "chart", "chartId": "chart_segmentation"},
        {
            "id": "interface_selectivity",
            "type": "markdown",
            "sourceId": "src_interface",
            "body": (
                "## Une interface mesurée plus petite peut rester plus nuisible\n\n"
                "La rétention RMS médiane vaut environ 0,53 pour HVQL, 0,80 pour HPVQL, 0,41 pour HVQM et 0,04 pour HQE. Dans le même passage, le dommage normalisé augmente pour les quatre modèles et l’alignement nuisible progresse. "
                "Cette dissociation est compatible avec une transmission directionnelle par l’encodage, l’évolution unitaire, les observables restreintes et le readout. Elle ne permet pas de dire que la couche unitaire débruite ou efface la perturbation."
            ),
        },
        {"id": "interface_chart", "type": "chart", "chartId": "chart_interface"},
        {
            "id": "residual_bypass",
            "type": "markdown",
            "sourceId": "src_residual",
            "body": (
                "## Le contournement classique domine la fusion résiduelle\n\n"
                "La contribution RMS médiane est proche de 0,19 pour la branche quantique, de 3,33 pour le bypass classique mis à l’échelle et de 3,48 après fusion. "
                "Le résultat soutient une contribution fonctionnelle du passage sélectif dans HVQL standard, mais ne sépare pas un effet quantique d’un effet de goulot, de mesure ou de capacité classique."
            ),
        },
        {"id": "residual_chart", "type": "chart", "chartId": "chart_residual"},
        {
            "id": "best_figures",
            "type": "markdown",
            "body": (
                "## Les trois figures les plus convaincantes ont des rôles différents\n\n"
                "1. **Progression :** `hvql_d50_rms_progression.pdf` montre clairement les renversements de norme sur les vrais blocs.\n"
                "2. **Sélectivité :** `interface_retention_vs_harm.pdf` sépare rétention et dommage; HQE doit rester explicitement qualifié comme instable.\n"
                "3. **Architecture :** `grouped_late_hybrids_margin_harm_progression.pdf` compare HVQL, HPVQL et HVQM sur le même chemin conceptuel.\n\n"
                "Le RMS est donc la meilleure vue primaire de la progression; le dommage de marge est son complément nécessaire, non un substitut."
            ),
        },
        {"id": "recommendation_table", "type": "table", "tableId": "table_recommendations"},
        {
            "id": "complete_tables",
            "type": "markdown",
            "body": (
                "## Les tableaux complets rendent chaque figure auditable\n\n"
                "Les quatre tableaux demandés sont fournis ci-dessous sans sous-échantillonnage : correspondance des étapes, synthèse de progression, valeurs des comparaisons groupées et recommandation des figures. "
                "Le manifest ajoute les chemins PDF, SVG, PNG, CSV, méthode et script pour chaque export."
            ),
        },
        {"id": "progression_summary_table", "type": "table", "tableId": "table_progression_summary"},
        {"id": "grouped_summary_table", "type": "table", "tableId": "table_grouped_summary"},
        {"id": "unavailable_table", "type": "table", "tableId": "table_unavailable"},
        {"id": "manifest_table", "type": "table", "tableId": "table_manifest"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limites, incertitudes et figures à écarter du corps principal\n\n"
                "- **À garder en annexe :** déplacement relatif, comparaisons décisionnelles, standard/résiduel/dropout et balayage HPVQL, car leurs cohortes ou contrôles sont plus faibles.\n"
                "- **À ne pas utiliser comme preuve autonome :** le point HQE normalisé, dont l’intervalle interquartile et les deux graines sont très dispersés.\n"
                "- **À ne pas produire depuis ces fichiers :** ansatz HVQL strong/basic/random et profondeurs HVQL 50/100/200. Les résultats d’attaque existent, mais pas la propagation par bloc.\n"
                "- **Confondeurs persistants :** précision propre, marge, largeur latente, dimension mesurée, capacité, optimisation et force de l’attaque propre au modèle."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Prochaines analyses à rendement élevé\n\n"
                "1. Tracer strong/basic/random et profondeur 50/100/200 sur une cohorte commune, sans modifier l’encodeur ni la tête.\n"
                "2. Refaire le diagnostic résiduel sur 500 exemples et deux graines pour la trajectoire complète, pas seulement les contributions de branche.\n"
                "3. Ajouter la norme des activations propres du grand MLP afin de récupérer son déplacement relatif.\n"
                "4. Pour HQE, publier dommage brut, ratio de médianes et couverture du dénominateur avant toute formulation principale."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Questions encore ouvertes\n\n"
                "- La concentration du dommage persiste-t-elle avec une tête classique dimensionnellement appariée?\n"
                "- La segmentation change-t-elle réellement la transmission lorsque paramètres, dimension mesurée et marge propre sont contrôlés?\n"
                "- Le bypass résiduel explique-t-il la baisse du profil faible-régime après réentraînement à échelle fixée?\n\n"
                "**Conclusion mécanistique sûre :** les blocs ne font pas qu’atténuer ou amplifier la perturbation; ils en restructurent la direction. Dans les hybrides tardifs étudiés, la mesure peut réduire le RMS observable tout en conservant une composante plus alignée avec la réduction de marge, puis la lecture finale peut la réamplifier."
            ),
        },
        {
            "id": "output_inventory_intro",
            "type": "markdown",
            "body": (
                "## Inventaire de sortie\n\n"
                "Le paquet contient 59 figures indépendantes en PDF, SVG et PNG 360 dpi, 59 CSV de valeurs tracées, 59 méthodes Markdown, les scripts, le notebook, les légendes LaTeX et ce rapport portable."
            ),
        },
        {"id": "inventory_table", "type": "table", "tableId": "table_inventory"},
        {
            "id": "gallery_intro",
            "type": "markdown",
            "body": (
                "## Galerie complète des figures exportées\n\n"
                "Chaque vignette ci-dessous est l’SVG exact du fichier exporté correspondant. La galerie clôt le rapport; les versions PDF, SVG et PNG restent disponibles dans `figures/`."
            ),
        },
    ]
    blocks.extend(_gallery_blocks(figure_manifest))

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Redesign des figures de propagation adversariale",
            "description": "Rapport technique, figures par modèle, comparaisons par rôle et sélection des figures de thèse.",
            "generatedAt": GENERATED_AT,
            "filters": [],
            "cards": [],
            "charts": _chart_specs(),
            "tables": _table_specs(),
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {name: _records(frame) for name, frame in tables.items()},
            "accessIssues": [],
        },
        "sources": [
            _query_source("src_hvql_rms", "Lit les valeurs exactes de la trajectoire RMS HVQL.", ["main.hvql_rms"], {"epsilon": "0.10", "attack": "auto-FGSM"}),
            _query_source("src_grouped_harm", "Lit la trajectoire groupée de dommage normalisé.", ["main.grouped_harm"], {"models": "HVQL, HPVQL5, HVQM", "tau": "1e-3"}),
            _query_source("src_interface", "Lit les métriques agrégées de transition d’interface.", ["main.interface"], {"epsilon": "0.10", "seeds": "0, 42"}),
            _query_source("src_decision", "Lit les comparaisons des représentations décisionnelles.", ["main.decision_stage"], {"epsilon": "0.10", "cohort": "19 common-correct"}),
            _query_source("src_segmentation", "Lit les trajectoires HPVQL à 5 et 10 blocs.", ["main.segmentation"], {"epsilon": "0.10", "seeds": "0, 42"}),
            _query_source("src_residual", "Lit les contributions RMS du modèle résiduel.", ["main.residual"], {"epsilon": "0.10", "seeds": "0, 42"}),
            _query_source("src_stage_mapping", "Lit la correspondance des couches et étapes.", ["main.stage_mapping"], {"models": "12 prioritized"}),
            _query_source("src_progression_summary", "Lit la synthèse de progression par modèle.", ["main.progression_summary"], {"epsilon": "0.10"}),
            _query_source("src_grouped_summary", "Lit toutes les valeurs des comparaisons groupées.", ["main.grouped_summary"], {"epsilon": "0.10"}),
            _query_source("src_recommendations", "Lit la sélection des figures principales.", ["main.recommendations"], {"shortlist": "3 figures"}),
            _query_source("src_unavailable", "Lit les demandes non récupérables depuis les traces.", ["main.unavailable"], {"policy": "no imputation"}),
            _query_source("src_manifest", "Lit le manifest des 59 figures.", ["main.figure_manifest"], {"formats": "PDF, SVG, PNG"}),
            _query_source("src_inventory", "Lit l’inventaire du paquet de sortie.", ["main.output_inventory"], {"output": "figure_redesign_from_trace"}),
            _query_source("src_source_inventory", "Lit l’inventaire des artefacts sources.", ["main.source_inventory"], {"source": "saved trace CSVs"}),
            _query_source("src_checkpoint", "Lit les validations de concordance de checkpoints.", ["main.checkpoint_validation"], {"seeds": "0, 42"}),
        ],
        "package_info": {
            "originUrl": "artifact://trace-figure-redesign",
            "controls": {"edit": False, "refresh": False},
        },
    }

    ARTIFACT_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, OUTPUT_DIR / "scripts" / Path(__file__).name)
    print(ARTIFACT_PATH)


if __name__ == "__main__":
    main()
