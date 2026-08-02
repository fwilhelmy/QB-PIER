"""Build and package the portable HTML follow-up report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re
import shutil
import sqlite3
import subprocess

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figure_redesign_from_trace"
ARTIFACT_PATH = OUTPUT_ROOT / "artifact_followup.json"
REPORT_PATH = OUTPUT_ROOT / "report_followup.html"
RECEIPT_PATH = OUTPUT_ROOT / "html_delivery_receipt_followup.json"
DATABASE_PATH = OUTPUT_ROOT / "report_followup_source.sqlite"
PLUGIN_ROOT = Path(r"C:\Users\fwilh\.codex\plugins\cache\openai-curated-remote\data-analytics\0.2.8-13ceeea1f599")
DELIVERY_SCRIPT = PLUGIN_ROOT / "skills" / "build-report" / "scripts" / "deliver_portable_artifact.mjs"
NODE = Path(r"C:\Users\fwilh\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
GENERATED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


TABLE_FILES = {
    "metric_audit": "metric_definition_audit.csv",
    "normalization_coverage": "margin_normalization_coverage.csv",
    "stage_mapping": "all_architectures_stage_mapping.csv",
    "recommendations": "followup_figure_recommendations.csv",
    "unavailable": "followup_unavailable_figures.csv",
    "figure_manifest": "followup_figure_manifest.csv",
    "source_inventory": "followup_source_inventory.csv",
    "output_inventory": "followup_output_inventory.csv",
}

KEY_FIGURES = {
    "src_hvql": "hvql_strong_d50_dual_progression",
    "src_cnn": "cnn_32_64_fc128_dual_progression",
    "src_mlp": "mlp_1024_512_64_64_dual_progression",
    "src_mixer": "mlpmixer_dual_progression",
    "src_residual_dual": "hvql_residual_dual_progression",
    "src_late": "late_hybrid_interface_normalized_harm_progression",
    "src_decision": "decision_routing_harm_efficiency_comparison",
    "src_residual_branch": "residual_hvql_branch_rms_comparison",
    "src_segmentation": "hvql_hpvql_interface_retention",
    "src_routing_variants": "hvql_routing_variants_normalized_harm_progression",
    "src_exploratory": "all_architectures_grouped_rms_reduced",
    "src_alignment": "mlpmixer_harmful_alignment_progression",
}


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.replace({np.nan: None})
    return json.loads(clean.to_json(orient="records", force_ascii=False))


def _source_manifest(source_id: str, label: str, path: str) -> dict[str, str]:
    return {"id": source_id, "label": label, "path": path}


def _source_query(source_id: str, table: str, description: str, filters: dict[str, str]) -> dict[str, object]:
    return {
        "id": source_id,
        "query": {
            "engine": "sqlite", "language": "sql", "sql": f"SELECT * FROM {table}",
            "description": description, "tables_used": [f"main.{table}"],
            "filters": filters, "executed_at": GENERATED_AT,
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
        raise ValueError(f"Racine SVG absente : {path}")
    svg = re.sub(r"<metadata>.*?</metadata>", "", match.group(1), flags=re.DOTALL)
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.DOTALL)
    svg = _prefix_svg_ids(svg, prefix)
    return svg.replace("<svg ", '<svg style="width:100%;height:auto;display:block" ', 1)


def _figure_html(figure_id: str, prefix: str) -> str:
    svg = _inline_svg(OUTPUT_ROOT / "followup_figures" / f"{figure_id}.svg", prefix)
    return (
        '<div style="max-width:1100px;margin:0 auto 18px">'
        f"{svg}"
        f'<p style="margin:7px 0 0;font:600 12px/1.35 ui-monospace,monospace;color:var(--text-muted,#596273)">{figure_id}.pdf</p>'
        "</div>"
    )


def _key_figure_block(block_id: str, figure_id: str) -> dict[str, object]:
    return {"id": block_id, "type": "html", "body": _figure_html(figure_id, block_id)}


def _gallery_blocks(manifest: pd.DataFrame) -> list[dict[str, object]]:
    css = """
<style>
.followup-gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;font-family:system-ui,sans-serif}
.followup-gallery figure{margin:0;padding:12px;border:1px solid #d7dde5;border-radius:10px;background:#fff;color:#18202a}
.followup-gallery figcaption{margin-top:8px;font:600 12px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}
@media(max-width:760px){.followup-gallery{grid-template-columns:1fr}}
@media(prefers-color-scheme:dark){.followup-gallery figure{background:#171a1f;color:#eef1f5;border-color:#3b424d}}
</style>
""".strip()
    blocks = []
    ordered = manifest.sort_values("figure_id").reset_index(drop=True)
    for chunk_start in range(0, len(ordered), 8):
        chunk = ordered.iloc[chunk_start:chunk_start + 8]
        parts = [css, '<div class="followup-gallery">']
        for row_index, row in chunk.iterrows():
            figure_id = str(row["figure_id"])
            svg = _inline_svg(OUTPUT_ROOT / str(row["svg_filename"]), f"gallery_{chunk_start}_{row_index}")
            parts.append(f"<figure>{svg}<figcaption>{figure_id}</figcaption></figure>")
        parts.append("</div>")
        blocks.append({"id": f"gallery_{chunk_start // 8 + 1}", "type": "html", "body": "\n".join(parts)})
    return blocks


def _table_specs() -> list[dict[str, object]]:
    return [
        {
            "id": "table_metric_audit", "title": "Audit exact des définitions métriques",
            "subtitle": "Le dommage normalisé du suivi est dérivé de la marge propre et non du dommage d’entrée sauvegardé.",
            "dataset": "metric_audit", "sourceId": "src_metric_audit",
            "defaultSort": {"field": "metric_name", "direction": "asc"},
            "columns": [
                {"field": "metric_name", "label": "Métrique", "type": "text"},
                {"field": "source_column", "label": "Colonne source", "type": "text"},
                {"field": "formula", "label": "Formule", "type": "text"},
                {"field": "normalization", "label": "Normalisation", "type": "text"},
                {"field": "aggregation", "label": "Agrégation", "type": "text"},
                {"field": "interpretation", "label": "Interprétation", "type": "text"},
                {"field": "caveat", "label": "Réserve", "type": "text"},
            ],
        },
        {
            "id": "table_coverage", "title": "Couverture de la normalisation par marge propre",
            "subtitle": "Une ligne par source, modèle et graine; seuil de marge propre fixé à 10⁻³.",
            "dataset": "normalization_coverage", "sourceId": "src_coverage",
            "defaultSort": {"field": "model", "direction": "asc"},
            "columns": [
                {"field": "source", "label": "Source", "type": "text"},
                {"field": "model", "label": "Modèle", "type": "text"},
                {"field": "seed", "label": "Graine", "format": "number"},
                {"field": "n_samples", "label": "Exemples", "format": "number"},
                {"field": "n_covered", "label": "Couverts", "format": "number"},
                {"field": "coverage", "label": "Couverture", "format": "percent"},
                {"field": "min_clean_margin", "label": "Marge min.", "format": "number"},
                {"field": "median_clean_margin", "label": "Marge médiane", "format": "number"},
            ],
        },
        {
            "id": "table_stage_mapping", "title": "Mapping explicite des architectures vers six rôles",
            "subtitle": "Aucune couche n’est interpolée lorsqu’un rôle n’a pas d’équivalent défendable.",
            "dataset": "stage_mapping", "sourceId": "src_stage_mapping",
            "defaultSort": {"field": "model", "direction": "asc"},
            "columns": [
                {"field": "model", "label": "Modèle", "type": "text"},
                {"field": "conceptual_stage_order", "label": "Ordre", "format": "number"},
                {"field": "conceptual_stage", "label": "Rôle conceptuel", "type": "text"},
                {"field": "raw_traced_layer", "label": "Couche réelle", "type": "text"},
                {"field": "status", "label": "Statut", "type": "text"},
            ],
        },
        {
            "id": "table_recommendations", "title": "Recommandation finale de six figures",
            "subtitle": "Une figure pour chaque rôle demandé, avec le texte d’accompagnement et la surinterprétation à éviter.",
            "dataset": "recommendations", "sourceId": "src_recommendations",
            "defaultSort": {"field": "rank", "direction": "asc"},
            "columns": [
                {"field": "rank", "label": "Rang", "format": "number"},
                {"field": "role", "label": "Rôle", "type": "text"},
                {"field": "figure", "label": "Figure", "type": "text"},
                {"field": "demonstrates", "label": "Démontre", "type": "text"},
                {"field": "placement", "label": "Placement", "type": "text"},
                {"field": "required_text", "label": "Texte requis", "type": "text"},
                {"field": "too_strong", "label": "Claim trop fort", "type": "text"},
            ],
        },
        {
            "id": "table_unavailable", "title": "Figures rejetées ou non récupérables",
            "subtitle": "Aucune valeur interne n’est imputée à partir de logits lorsque les traces par bloc manquent.",
            "dataset": "unavailable", "sourceId": "src_unavailable",
            "defaultSort": {"field": "requested_figures", "direction": "asc"},
            "columns": [
                {"field": "requested_figures", "label": "Demande", "type": "text"},
                {"field": "status", "label": "Statut", "type": "text"},
                {"field": "reason", "label": "Raison", "type": "text"},
                {"field": "needed", "label": "Données nécessaires", "type": "text"},
            ],
        },
        {
            "id": "table_manifest", "title": "Manifest complet des figures de suivi",
            "subtitle": "Chaque figure possède PDF, SVG, PNG, CSV, méthode, légende et script exact.",
            "dataset": "figure_manifest", "sourceId": "src_manifest",
            "defaultSort": {"field": "figure_id", "direction": "asc"},
            "columns": [
                {"field": "figure_id", "label": "Figure", "type": "text"},
                {"field": "concept", "label": "Concept", "type": "text"},
                {"field": "models", "label": "Modèles", "type": "text"},
                {"field": "metric", "label": "Métrique", "type": "text"},
                {"field": "recommended_use", "label": "Usage", "type": "text"},
                {"field": "pdf_filename", "label": "PDF", "type": "text"},
                {"field": "source_csv", "label": "CSV", "type": "text"},
                {"field": "plotting_script", "label": "Script", "type": "text"},
                {"field": "method_file", "label": "Méthode", "type": "text"},
            ],
        },
        {
            "id": "table_sources", "title": "Inventaire des artefacts sources",
            "subtitle": "Toutes les sources sont des traces, logits ou audits déjà sauvegardés.",
            "dataset": "source_inventory", "sourceId": "src_source_inventory",
            "defaultSort": {"field": "source_key", "direction": "asc"},
            "columns": [
                {"field": "source_key", "label": "Source", "type": "text"},
                {"field": "path", "label": "Chemin", "type": "text"},
                {"field": "exists", "label": "Présente", "type": "text"},
                {"field": "bytes", "label": "Octets", "format": "number"},
            ],
        },
        {
            "id": "table_inventory", "title": "Inventaire de sortie du suivi",
            "subtitle": "Chemins du paquet après génération; les tailles exactes restent disponibles dans le CSV livré.",
            "dataset": "output_inventory", "sourceId": "src_inventory",
            "defaultSort": {"field": "path", "direction": "asc"},
            "columns": [
                {"field": "path", "label": "Chemin", "type": "text"},
                {"field": "extension", "label": "Format", "type": "text"},
            ],
        },
    ]


def _load_tables() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(OUTPUT_ROOT / filename) for name, filename in TABLE_FILES.items()}


def _estimate(figure_id: str, model: str, stage: str, metric: str | None = None) -> float:
    frame = pd.read_csv(OUTPUT_ROOT / "followup_figure_data" / f"{figure_id}.csv")
    selected = frame[(frame["model"] == model) & (frame["stage_label"] == stage)]
    if metric is not None:
        selected = selected[selected["metric"] == metric]
    return float(selected["estimate"].iloc[0])


def _inventory_now() -> pd.DataFrame:
    rows = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file():
            rows.append({
                "path": path.relative_to(OUTPUT_ROOT).as_posix(),
                "extension": path.suffix.lower(), "size_bytes": path.stat().st_size,
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Predeclare the three portable-report paths so the report inventory is path-complete.
    for path in (ARTIFACT_PATH, REPORT_PATH, RECEIPT_PATH):
        if not path.exists():
            path.write_text("", encoding="utf-8")
    (OUTPUT_ROOT / "followup_scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, OUTPUT_ROOT / "followup_scripts" / Path(__file__).name)
    validator = PROJECT_ROOT / "scripts" / "validate_trace_figure_followup.py"
    if validator.exists():
        shutil.copy2(validator, OUTPUT_ROOT / "followup_scripts" / validator.name)
    _inventory_now().to_csv(OUTPUT_ROOT / "followup_output_inventory.csv", index=False)

    tables = _load_tables()
    for source_id, figure_id in KEY_FIGURES.items():
        tables[source_id.removeprefix("src_")] = pd.read_csv(
            OUTPUT_ROOT / "followup_figure_data" / f"{figure_id}.csv"
        )
    with sqlite3.connect(DATABASE_PATH) as connection:
        for name, frame in tables.items():
            frame.to_sql(name, connection, if_exists="replace", index=False)

    hvql_conv = _estimate("hvql_strong_d50_dual_progression", "HVQL", "Conv. 1", "gain_rms")
    hvql_measure_gain = _estimate("hvql_strong_d50_dual_progression", "HVQL", "Mesures quantiques", "gain_rms")
    hvql_measure_harm = _estimate("hvql_strong_d50_dual_progression", "HVQL", "Mesures quantiques", "margin_normalized_harm")
    hvql_logits_gain = _estimate("hvql_strong_d50_dual_progression", "HVQL", "Logits", "gain_rms")
    cnn_dense_gain = _estimate("cnn_32_64_fc128_dual_progression", "CNN", "Projection dense", "gain_rms")
    cnn_logits_harm = _estimate("cnn_32_64_fc128_dual_progression", "CNN", "Logits", "margin_normalized_harm")
    residual = tables["residual_branch"]
    bypass = float(residual.loc[residual["layer"] == "residual_branch", "estimate"].iloc[0])
    quantum = float(residual.loc[residual["layer"] == "quantum_branch", "estimate"].iloc[0])
    retention = tables["segmentation"].set_index("model")["estimate"]

    manifest_sources = []
    top_sources = []
    for source_id, figure_id in KEY_FIGURES.items():
        table = source_id.removeprefix("src_")
        relative = f"outputs/figure_redesign_from_trace/followup_figure_data/{figure_id}.csv"
        manifest_sources.append(_source_manifest(source_id, figure_id, relative))
        top_sources.append(_source_query(source_id, table, f"Valeurs exactes de {figure_id}.", {"epsilon": "0.10", "attack": "self-FGSM"}))
    table_source_ids = {
        "src_metric_audit": ("metric_audit", "Définitions métriques"),
        "src_coverage": ("normalization_coverage", "Couverture de normalisation"),
        "src_stage_mapping": ("stage_mapping", "Mapping des étapes"),
        "src_recommendations": ("recommendations", "Recommandations finales"),
        "src_unavailable": ("unavailable", "Figures indisponibles"),
        "src_manifest": ("figure_manifest", "Manifest des figures"),
        "src_source_inventory": ("source_inventory", "Inventaire des sources"),
        "src_inventory": ("output_inventory", "Inventaire des sorties"),
    }
    for source_id, (table, label) in table_source_ids.items():
        relative = f"outputs/figure_redesign_from_trace/{TABLE_FILES[table]}"
        manifest_sources.append(_source_manifest(source_id, label, relative))
        top_sources.append(_source_query(source_id, table, f"Lecture de {label.lower()}.", {"scope": "follow-up figure set"}))

    blocks: list[dict[str, object]] = [
        {"id": "title", "type": "markdown", "body": "# Jeu final de figures mécanistiques pour la thèse"},
        {
            "id": "summary", "type": "markdown", "body": (
                "## Synthèse technique des corrections\n\n"
                "**Le jeu final sépare désormais progression, dommage, interface et routage.** Les figures individuelles suivent des blocs réellement exécutés; les comparaisons transversales n’emploient que des rôles déclarés comme homologues. Le gain RMS reste la vue primaire de propagation, tandis que le dommage normalisé par la marge propre localise la pertinence décisionnelle.\n\n"
                "**La dissociation attendue est visible sans conclure à un débruitage quantique.** Dans HVQL, le premier bloc ramène le gain à "
                f"{hvql_conv:.2f}, la mesure à {hvql_measure_gain:.2f}, puis les logits à {hvql_logits_gain:.1f}; "
                f"dans le même passage, le dommage normalisé atteint {hvql_measure_harm:.2f}. Le motif visualise une réorganisation, pas une suppression de l’information adversariale.\n\n"
                "**Les limites sont devenues explicites.** Les comparaisons HVQL strong/basic/random et profondeurs 50/100/200 ne sont pas tracées, faute d’activations internes sauvegardées. Une seule marge HVQM sur 1 000 est exclue par le seuil de 10⁻³."
            ),
        },
        {
            "id": "correction_scope", "type": "markdown", "body": (
                "## Ce suivi corrige les figures, pas l’analyse scientifique\n\n"
                "Aucun checkpoint n’est rechargé et aucune attaque n’est recalculée. Le suivi réutilise les traces à 500 exemples par graine lorsque disponibles, le diagnostic historique commun-correct de 19 exemples pour le chemin résiduel complet, les logits propres sauvegardés pour la marge de normalisation et les audits d’architecture déjà validés. Les sorties antérieures sont conservées; le nouveau jeu vit dans les dossiers `followup_*`."
            ),
        },
        {
            "id": "metric_audit", "type": "markdown", "sourceId": "src_metric_audit", "body": (
                "## L’audit métrique distingue trois normalisations incompatibles\n\n"
                "`gain_rms` est déjà normalisé par la perturbation d’entrée. `delta_rms` est un RMS brut par dimension. Le champ historique `normalized_margin_harm` divise par le dommage d’entrée et n’est donc pas utilisé ici comme dommage relatif à la marge. Le suivi calcule au niveau échantillon `projected_margin_harm / (clean_margin + 10⁻³)` et exclut les marges propres inférieures à 10⁻³."
            ),
        },
        {
            "id": "metric_coverage", "type": "markdown", "sourceId": "src_coverage", "body": (
                "### La normalisation par marge propre est numériquement couverte\n\n"
                "La couverture est de 100 % pour les modèles principaux et de 99,8 % pour HVQM à la graine 0. L’efficacité du dommage est également calculée au niveau échantillon comme `H̃/(G+10⁻³)`; ses figures restent diagnostiques lorsque le gain est faible."
            ),
        },
        {
            "id": "hvql_section", "type": "markdown", "sourceId": "src_hvql", "body": (
                "## HVQL rend visible la séquence magnitude → réorganisation → dommage\n\n"
                f"Le gain RMS baisse d’abord à {hvql_conv:.2f}, remonte dans l’interface dense, retombe à {hvql_measure_gain:.2f} aux mesures puis atteint {hvql_logits_gain:.1f} aux logits. "
                f"Le dommage normalisé reste faible dans l’encodeur avant de passer à {hvql_measure_harm:.2f} à la mesure. Cette divergence est compatible avec une sélection directionnelle suivie d’une réamplification classique; elle ne prouve pas que l’unitaire débruite."
            ),
        },
        _key_figure_block("hvql_figure", KEY_FIGURES["src_hvql"]),
        {
            "id": "cnn_section", "type": "markdown", "sourceId": "src_cnn", "body": (
                "## Le CNN concentre surtout la magnitude dans sa projection dense\n\n"
                f"Le CNN atténue également le premier bloc, puis la projection dense porte le gain à {cnn_dense_gain:.1f}. Son dommage normalisé aux logits vaut {cnn_logits_harm:.2f} et varie peu entre les premiers blocs. "
                "La comparaison montre qu’une reconcentration tardive n’est pas propre aux architectures quantiques. Le CNN n’est toutefois ni dimensionnellement ni capacitairement apparié à HVQL."
            ),
        },
        _key_figure_block("cnn_figure", KEY_FIGURES["src_cnn"]),
        {
            "id": "secondary_section", "type": "markdown", "body": (
                "## Les trajectoires secondaires empêchent une explication unique par le mélange global\n\n"
                "Le grand MLP amplifie fortement le gain sans reproduire exactement la trajectoire du petit MLP. MLPMixer montre une atténuation initiale, une divergence latente et une forte hausse d’alignement à partir du quatrième bloc. Le HVQL résiduel révèle deux routes parallèles avant la fusion. Ces figures sont plus utiles en annexe que comme comparaison causale principale."
            ),
        },
        _key_figure_block("mlp_figure", KEY_FIGURES["src_mlp"]),
        _key_figure_block("mixer_figure", KEY_FIGURES["src_mixer"]),
        _key_figure_block("residual_dual_figure", KEY_FIGURES["src_residual_dual"]),
        {
            "id": "late_section", "type": "markdown", "sourceId": "src_late", "body": (
                "## Les interfaces tardives concentrent le dommage selon des chemins différents\n\n"
                "Sur la cohorte diagnostique commune, le dommage augmente de la pré-interface à la mesure pour HVQL, HPVQL, HVQM et HQE, tandis que le HVQL résiduel concentre surtout le dommage à la fusion. Le stade de fusion est affiché uniquement pour le modèle résiduel : il n’est jamais assimilé à une sortie mesurée pure. La petite cohorte interdit une généralisation populationnelle."
            ),
        },
        _key_figure_block("late_figure", KEY_FIGURES["src_late"]),
        {
            "id": "decision_section", "type": "markdown", "sourceId": "src_decision", "body": (
                "## L’efficacité décisionnelle distingue mesures, lecture classique et fusion\n\n"
                "Sur les 19 exemples appariés, HVQM présente l’efficacité médiane la plus élevée, HVQL et HPVQL sont proches, le CNN est plus bas et la fusion résiduelle est la moins efficace par unité de gain malgré son grand déplacement. La figure décrit une compatibilité avec la marge; elle ne mesure pas une ‘responsabilité quantique’ pure."
            ),
        },
        _key_figure_block("decision_figure", KEY_FIGURES["src_decision"]),
        {
            "id": "residual_branch_section", "type": "markdown", "sourceId": "src_residual_branch", "body": (
                "### Le contournement classique transporte le déplacement autour de la mesure\n\n"
                f"Le RMS du contournement est {bypass / quantum:.1f} fois celui de la branche quantique ({bypass:.2f} contre {quantum:.2f}). La fusion reste proche du bypass. Ce résultat soutient un rôle fonctionnel du routage, sans isoler un mécanisme spécifiquement quantique."
            ),
        },
        _key_figure_block("residual_branch_figure", KEY_FIGURES["src_residual_branch"]),
        {
            "id": "controlled_section", "type": "markdown", "sourceId": "src_segmentation", "body": (
                "## La variation récupérable concerne la partition et le routage, pas l’ansatz ou la profondeur\n\n"
                f"La rétention RMS médiane passe de {retention['HVQL']:.2f} pour HVQL à {retention['HPVQL (5 blocs)']:.2f} et {retention['HPVQL (10 blocs)']:.2f}. "
                "Cela suffit à montrer que le couplage quantique global complet n’est pas nécessaire au profil observé, mais ni le nombre de paramètres ni la dimension mesurée ne sont strictement contrôlés. Les vues standard/dropout/résiduel restent un diagnostic à une graine."
            ),
        },
        {"id": "segmentation_native", "type": "chart", "chartId": "chart_retention"},
        _key_figure_block("routing_variants_figure", KEY_FIGURES["src_routing_variants"]),
        {
            "id": "exploratory_section", "type": "markdown", "sourceId": "src_exploratory", "body": (
                "## La vue toutes architectures est utilisable seulement comme carte exploratoire\n\n"
                "La version réduite révèle les renversements de gain sur six rôles explicites et reste lisible grâce à une légende hors de la zone de données. Elle mélange cependant attaques propres, capacités, cohortes et architectures; elle doit rester en annexe et être accompagnée du mapping complet. La version complète sert uniquement à l’inspection."
            ),
        },
        _key_figure_block("exploratory_figure", KEY_FIGURES["src_exploratory"]),
        {
            "id": "diagnostics_section", "type": "markdown", "sourceId": "src_alignment", "body": (
                "## L’alignement explique mieux que la sensibilité plusieurs divergences RMS–dommage\n\n"
                "Pour HVQL et MLPMixer, l’alignement nuisible augmente fortement près de la mesure ou des blocs tardifs alors que le gain ne suit pas toujours la même direction. Pour CNN et le grand MLP, sensibilité et alignement doivent être lus conjointement. Pour le modèle résiduel, l’alignement est le diagnostic le plus interprétable, mais seulement sur 19 exemples. Toutes ces figures appartiennent à l’annexe."
            ),
        },
        _key_figure_block("alignment_figure", KEY_FIGURES["src_alignment"]),
        {
            "id": "recommendation_section", "type": "markdown", "sourceId": "src_recommendations", "body": (
                "## Six recommandations couvrent exactement les rôles demandés\n\n"
                "1. **Progression principale :** `hvql_strong_d50_dual_progression.pdf`.\n"
                "2. **Comparaison classique :** `cnn_32_64_fc128_dual_progression.pdf`.\n"
                "3. **Interface sélective :** `late_hybrid_interface_normalized_harm_progression.pdf`.\n"
                "4. **Routage décisionnel :** `decision_routing_harm_efficiency_comparison.pdf`.\n"
                "5. **Variation quantique contrôlée :** `hvql_hpvql_interface_retention.pdf`.\n"
                "6. **Toutes architectures :** utilisable uniquement en annexe dans sa version réduite."
            ),
        },
        {
            "id": "appendix_section", "type": "markdown", "body": (
                "## Candidats d’annexe\n\n"
                "Les versions mono-métriques des cinq modèles, toutes les sensibilités et alignements, les variantes standard/dropout/résiduel, les décompositions de branches, les versions avec IQR et la vue globale complète sont des compléments. Elles répondent à des questions précises mais alourdiraient le récit principal."
            ),
        },
        {
            "id": "rejected_section", "type": "markdown", "sourceId": "src_unavailable", "body": (
                "## Figures rejetées ou trompeuses\n\n"
                "Les figures d’ansatz et de profondeur HVQL ne sont pas produites : les logits propres ou d’attaque ne reconstruisent ni les déplacements latents ni les gradients de marge. Le déplacement relatif du grand MLP reste absent. Aucune trace VQM profondeur 100/200 n’est relabellisée en HVQL."
            ),
        },
        {
            "id": "limitations", "type": "markdown", "body": (
                "## Limites et prochaine mesure utile\n\n"
                "Les attaques propres optimisent des directions différentes; les figures ne comparent donc pas une perturbation identique entre modèles. Les capacités, marges, largeurs et readouts restent des confondeurs. La prochaine expérience la plus utile serait un traçage apparié de HVQL strong/basic/random et profondeurs 50/100/200 avec encodeur, largeur, exemples et graines fixés."
            ),
        },
        {
            "id": "further_questions", "type": "markdown", "body": (
                "## Questions ouvertes\n\n"
                "- La concentration du dommage persiste-t-elle avec une tête classique dimensionnellement appariée?\n"
                "- La partition change-t-elle la transmission lorsque paramètres et dimension mesurée sont fixés?\n"
                "- Le profil résiduel se confirme-t-il sur une trajectoire complète de 500 exemples et deux graines?\n\n"
                "**Conclusion sûre :** le chemin visuel soutenu est magnitude de perturbation → réorganisation latente → mesure sélective → dommage décisionnel → routage de l’interface. Il ne permet pas d’attribuer causalement ce chemin à la nature quantique seule."
            ),
        },
        {
            "id": "gallery_intro", "type": "markdown", "body": (
                f"## Galerie complète des {len(tables['figure_manifest'])} figures\n\n"
                "La galerie contient l’SVG exact de chaque figure exportée. Les PDF et PNG 360 dpi se trouvent dans `followup_figures/`."
            ),
        },
    ]
    blocks.extend(_gallery_blocks(tables["figure_manifest"]))
    blocks.extend([
        {
            "id": "tables_intro", "type": "markdown", "body": (
                "## Tables complètes\n\n"
                "Les tables finales regroupent définitions, couverture, mapping, recommandations, absences, manifest et provenance. Elles apparaissent après la galerie conformément au contrat du suivi."
            ),
        },
        {"id": "metric_audit_table", "type": "table", "tableId": "table_metric_audit"},
        {"id": "coverage_table", "type": "table", "tableId": "table_coverage"},
        {"id": "stage_mapping_table", "type": "table", "tableId": "table_stage_mapping"},
        {"id": "recommendations_table", "type": "table", "tableId": "table_recommendations"},
        {"id": "unavailable_table", "type": "table", "tableId": "table_unavailable"},
        {"id": "manifest_table", "type": "table", "tableId": "table_manifest"},
        {"id": "sources_table", "type": "table", "tableId": "table_sources"},
        {"id": "inventory_intro", "type": "markdown", "body": "## Inventaire de sortie\n\nLe paquet de suivi conserve les anciennes sorties et ajoute les nouveaux dossiers, scripts, méthodes, légendes, tables et le rapport portable."},
        {"id": "inventory_table", "type": "table", "tableId": "table_inventory"},
    ])

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": "Jeu final de figures mécanistiques pour la thèse",
            "description": "Progression des perturbations, dommage de marge, interface sélective et routage décisionnel.",
            "generatedAt": GENERATED_AT, "filters": [], "cards": [], "charts": [
                {
                    "id": "chart_retention", "title": "Rétention RMS à l’interface",
                    "subtitle": "Médiane de deux graines; IQR disponible dans les données exactes.",
                    "type": "bar", "dataset": "segmentation", "sourceId": "src_segmentation",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "model", "type": "nominal", "label": "Architecture"},
                        "y": {"field": "estimate", "type": "quantitative", "label": "Mesures / pré-interface"},
                        "tooltip": [
                            {"field": "model", "type": "nominal", "label": "Modèle"},
                            {"field": "estimate", "type": "quantitative", "label": "Médiane", "format": "number"},
                            {"field": "q25", "type": "quantitative", "label": "Q1", "format": "number"},
                            {"field": "q75", "type": "quantitative", "label": "Q3", "format": "number"},
                        ],
                    },
                },
            ],
            "tables": _table_specs(), "sources": manifest_sources, "blocks": blocks,
        },
        "snapshot": {
            "version": 1, "generatedAt": GENERATED_AT, "status": "ready",
            "datasets": {name: _records(frame) for name, frame in tables.items()}, "accessIssues": [],
        },
        "sources": top_sources,
        "package_info": {"originUrl": "artifact://trace-figure-followup", "controls": {"edit": False, "refresh": False}},
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    if not NODE.exists() or not DELIVERY_SCRIPT.exists():
        raise FileNotFoundError("Node ou builder portable absent")
    result = subprocess.run(
        [str(NODE), str(DELIVERY_SCRIPT), "--input", str(ARTIFACT_PATH), "--output", str(REPORT_PATH)],
        cwd=PLUGIN_ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace",
    )
    receipt_text = result.stdout.strip() or result.stderr.strip()
    try:
        receipt = json.loads(receipt_text)
    except json.JSONDecodeError:
        receipt = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(receipt_text)
    _inventory_now().to_csv(OUTPUT_ROOT / "followup_output_inventory.csv", index=False)
    print(json.dumps({"artifact": str(ARTIFACT_PATH), "report": str(REPORT_PATH), "receipt": receipt}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
