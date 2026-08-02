"""Build the canonical portable Data Analytics report payload.

The packaged HTML itself is produced by the Data Analytics build-report
delivery command.  This script only assembles the validated artifact input
from the reviewed figure CSV files and attack aggregates.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from analytics.attacks.mechanistic_visualizations import (
    OUTPUT_DIR,
    attack_outcome_data,
    load_attack_aggregates,
)


ARTIFACT_PATH = OUTPUT_DIR / "artifact.json"
DATABASE_PATH = OUTPUT_DIR / "report_source.sqlite"
GENERATED_AT = "2026-07-27T04:30:00-04:00"

SOURCE_SQL = {
    "src_layerwise": "SELECT * FROM gain ORDER BY model_key, stage_index",
    "src_interface": "SELECT * FROM interface ORDER BY model_key",
    "src_normalization": "SELECT * FROM normalization ORDER BY model_key, epsilon, tau",
    "src_direction": "SELECT * FROM direction ORDER BY model, source_fr",
    "src_residual": "SELECT * FROM residual ORDER BY branch",
    "src_attack": "SELECT * FROM robust ORDER BY value DESC",
    "src_validation": "SELECT * FROM validation ORDER BY model",
    "src_blocks": "SELECT * FROM block_effects ORDER BY modele",
    "src_hypotheses": "SELECT * FROM hypotheses ORDER BY hypothese",
    "src_ranking": "SELECT * FROM ranking ORDER BY rang",
}


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.replace({np.nan: None})
    return json.loads(clean.to_json(orient="records", force_ascii=False))


def _source(source_id: str, label: str, path: str) -> dict[str, str]:
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


def main() -> None:
    gain = pd.read_csv(OUTPUT_DIR / "fig_a1_gain_layerwise" / "fig_a1_gain_layerwise.csv")
    gain = gain[["model_key", "model", "stage_index", "stage_fr", "estimate", "lower", "upper", "n_samples", "n_seeds"]]
    gain = gain.rename(columns={"estimate": "gain_rms"}).sort_values(["model_key", "stage_index"])

    interface = pd.read_csv(OUTPUT_DIR / "fig_c1_interface_rms" / "fig_c1_interface_rms.csv")
    interface = interface[[
        "model_key", "model", "retention_rms", "retention_rms_ci_low", "retention_rms_ci_high",
        "harm_change_normalized", "harm_change_normalized_ci_low", "harm_change_normalized_ci_high",
        "alignment_change", "harm_ratio_coverage", "n_samples", "n_seeds",
    ]]

    residual = pd.read_csv(OUTPUT_DIR / "fig_g1_residual_rms" / "fig_g1_residual_rms.csv")
    residual = residual[["branch", "branch_fr", "value", "lower", "upper", "n_samples", "n_seeds"]].rename(
        columns={"value": "delta_rms", "lower": "ci_low", "upper": "ci_high"}
    )
    residual_l2 = pd.read_csv(OUTPUT_DIR / "fig_g1_residual_l2" / "fig_g1_residual_l2.csv")
    q_l2 = float(residual_l2.loc[residual_l2["branch"] == "quantum_branch", "value"].iloc[0])
    bypass_l2 = float(residual_l2.loc[residual_l2["branch"] == "residual_branch", "value"].iloc[0])
    bypass_ratio = bypass_l2 / q_l2
    residual_summary = pd.DataFrame([{"metric": "rapport_l2_contournement_quantique", "value": bypass_ratio}])

    outcomes = attack_outcome_data(load_attack_aggregates(), ("hvql", "hpvql5", "hvqm", "hqe"))
    outcomes.to_csv(OUTPUT_DIR / "late_hybrid_outcomes.csv", index=False)
    robust = outcomes[outcomes["metric"] == "robust_accuracy"].copy()

    direction_r = pd.read_csv(OUTPUT_DIR / "fig_d1_direction_retention" / "fig_d1_direction_retention.csv")
    direction_h = pd.read_csv(OUTPUT_DIR / "fig_d2_direction_harm" / "fig_d2_direction_harm.csv")
    direction_c = pd.read_csv(OUTPUT_DIR / "fig_d3_direction_alignment" / "fig_d3_direction_alignment.csv")
    direction = direction_r[["model_key", "model", "source_key", "source_fr", "value"]].rename(
        columns={"value": "retention_rms"}
    )
    direction = direction.merge(
        direction_h[["model_key", "source_key", "value"]].rename(columns={"value": "harm_change"}),
        on=["model_key", "source_key"], how="left",
    ).merge(
        direction_c[["model_key", "source_key", "value"]].rename(columns={"value": "alignment_change"}),
        on=["model_key", "source_key"], how="left",
    )

    alignment = pd.read_csv(OUTPUT_DIR / "checkpoint_alignment.csv")
    validation = alignment.groupby(["model_key", "model"], as_index=False).agg(
        graines=("seed", "nunique"),
        accord_predictions_min=("prediction_agreement", "min"),
        ecart_logit_max=("max_abs_logit_discrepancy", "max"),
        statut=("status", lambda values: "; ".join(sorted(set(values)))),
    )

    normalization = pd.read_csv(OUTPUT_DIR / "normalization_sensitivity.csv")
    normalization = normalization[
        normalization["tau"].isin(["tau=1e-4", "tau=1e-3", "tau=1e-2"])
    ].copy()

    hypotheses = pd.DataFrame([
        {"hypothese": "H1 — mélange global précoce", "verdict": "Partiellement soutenue", "pour": "Les deux MLP amplifient dès le premier bloc.", "contre": "La projection initiale du MLPMixer atténue fortement; aucun contrôle causal de portée du mélange.", "niveau": "Association"},
        {"hypothese": "H2 — traitement local ou hiérarchique", "verdict": "Partiellement soutenue", "pour": "CNN et ResNet18 atténuent au premier bloc.", "contre": "HQuanv est proche de l'unité et les blocs tardifs inversent l'atténuation.", "niveau": "Association"},
        {"hypothese": "H3 — interaction globale après encodage", "verdict": "Partiellement soutenue", "pour": "HVQL et HPVQL restent performants malgré une interaction latente globale ou partitionnée.", "contre": "Absence de remplacement classique dimensionnellement apparié.", "niveau": "Mécanisme plausible"},
        {"hypothese": "H4 — la couche quantique remodèle sans effacer", "verdict": "Soutenue", "pour": "Rétention RMS < 1, hausse d'alignement et hausse du dommage à la mesure.", "contre": "La part état complet versus observables n'est pas tracée directement.", "niveau": "Mesure directe + conséquence mathématique"},
        {"hypothese": "H5 — représentation riche et rôle quantique intermédiaire", "verdict": "Partiellement soutenue", "pour": "HVQL et HPVQL combinent 99,2 % de précision propre et 71–76 % robuste.", "contre": "Capacité, marge et optimisation diffèrent de HVQM.", "niveau": "Association"},
        {"hypothese": "H6 — couplage global complet non nécessaire", "verdict": "Soutenue", "pour": "HPVQL à 5 et 10 blocs conserve le profil faible-régime sans circuit global unique.", "contre": "La segmentation n'est pas une ablation contrôlée et n'est pas démontrée protectrice.", "niveau": "Non-nécessité empirique"},
        {"hypothese": "H7 — responsabilité quantique non monotone", "verdict": "Soutenue descriptivement", "pour": "HVQM est moins robuste que HVQL/HPVQL; HQE n'établit aucun ordre monotone.", "contre": "La responsabilité n'est pas isolée de l'encodeur et de la capacité.", "niveau": "Association"},
        {"hypothese": "H8 — compression HQE et puissance sortante", "verdict": "Non testable ici", "pour": "HQE est fortement direction-sélectif à l'interface.", "contre": "La puissance sortante n'est pas isolée sous sources appariées; dommage normalisé instable.", "niveau": "Évidence insuffisante"},
        {"hypothese": "H9 — disparition après contrôles", "verdict": "Non testable", "pour": "Les confondeurs sont quantifiés en partie.", "contre": "Aucune série entraînée appariant capacité, marge et interface.", "niveau": "Évidence insuffisante"},
    ])

    block_effects = pd.DataFrame([
        {"modele": "CNN", "premier_bloc": "Atténuation RMS (0,79×)", "encodeur": "Renversement à 14,4× avant décision", "interface_quantique": "Sans objet", "tete_finale": "23,3× aux logits", "etape_dominante": "Encodeur profond puis tête"},
        {"modele": "ResNet18", "premier_bloc": "Forte atténuation (0,13×)", "encodeur": "Renversement à 8,0×", "interface_quantique": "Sans objet", "tete_finale": "36,2× aux logits", "etape_dominante": "Pile résiduelle pour le dommage"},
        {"modele": "MLP [128,64]", "premier_bloc": "Amplification (7,05×)", "encodeur": "18,5×", "interface_quantique": "Sans objet", "tete_finale": "76,1×", "etape_dominante": "Projection dense précoce"},
        {"modele": "MLP [1024,512,64,64]", "premier_bloc": "Amplification (2,05×)", "encodeur": "35,6×", "interface_quantique": "Sans objet", "tete_finale": "67,4×", "etape_dominante": "Encodeur dense"},
        {"modele": "MLPMixer", "premier_bloc": "Atténuation (0,19×)", "encodeur": "Renversement à 6,90×", "interface_quantique": "Sans objet", "tete_finale": "43,2×", "etape_dominante": "Norme accrue mais dommage réduit dans le mixer"},
        {"modele": "HQuanv", "premier_bloc": "Quasi-conservation (1,08×)", "encodeur": "3,60× avant décision", "interface_quantique": "Traitement local précoce", "tete_finale": "75,8×", "etape_dominante": "Tête finale en norme"},
        {"modele": "HVQL", "premier_bloc": "Atténuation (0,64×)", "encodeur": "4,85× pré-interface", "interface_quantique": "Rétention 0,53×; dommage +1,58", "tete_finale": "27,3× aux logits", "etape_dominante": "Mesure puis readout"},
        {"modele": "HPVQL (5 blocs)", "premier_bloc": "Encodeur classique riche", "encodeur": "Latent partitionné", "interface_quantique": "Rétention 0,80×; dommage +1,37", "tete_finale": "Readout classique", "etape_dominante": "Interface et readout"},
        {"modele": "HVQM", "premier_bloc": "Encodeur plus faible", "encodeur": "Responsabilité quantique accrue", "interface_quantique": "Rétention 0,41×; dommage +0,19", "tete_finale": "Readout plus contraint", "etape_dominante": "Non isolée"},
        {"modele": "HQE", "premier_bloc": "Compression dense forte", "encodeur": "Latent étroit", "interface_quantique": "Rétention 0,04×; dommage normalisé instable", "tete_finale": "Réamplification possible", "etape_dominante": "Non concluant"},
        {"modele": "HVQL résiduel", "premier_bloc": "Comme HVQL", "encodeur": "Latent partagé", "interface_quantique": "Branche RMS 0,19", "tete_finale": "Fusion RMS 3,49", "etape_dominante": "Contournement classique (17,3× L2)"},
    ])

    ranking = pd.DataFrame([
        {"rang": 1, "figure_id": "fig_c1_interface_rms", "usage": "Corps principal", "force": "Mesure directement la baisse RMS et la concentration du dommage; HQE doit être qualifié.", "includegraphics": r"\\includegraphics[width=0.9\\textwidth]{fig_c1_interface_rms.pdf}"},
        {"rang": 2, "figure_id": "fig_b4_norm_harm_transition", "usage": "Corps principal", "force": "Montre les quadrants norme–dommage et le contre-exemple MLPMixer.", "includegraphics": r"\\includegraphics[width=0.9\\textwidth]{fig_b4_norm_harm_transition.pdf}"},
        {"rang": 3, "figure_id": "fig_a1_gain_layerwise", "usage": "Corps principal", "force": "Compare les renversements après le premier bloc, y compris le grand MLP.", "includegraphics": r"\\includegraphics[width=0.9\\textwidth]{fig_a1_gain_layerwise.pdf}"},
        {"rang": 4, "figure_id": "fig_d1_direction_retention", "usage": "Corps principal", "force": "Établit l'hétérogénéité source–interface avec contrôle aléatoire.", "includegraphics": r"\\includegraphics[width=0.9\\textwidth]{fig_d1_direction_retention.pdf}"},
        {"rang": 5, "figure_id": "fig_d2_direction_harm", "usage": "Corps principal", "force": "Sépare rétention et dommage selon la direction source.", "includegraphics": r"\\includegraphics[width=0.9\\textwidth]{fig_d2_direction_harm.pdf}"},
        {"rang": 6, "figure_id": "fig_g1_residual_rms", "usage": "Corps principal", "force": "Quantifie le contournement classique au lieu de l'inférer.", "includegraphics": r"\\includegraphics[width=0.9\\textwidth]{fig_g1_residual_rms.pdf}"},
        {"rang": 7, "figure_id": "fig_b1_harm_normalized", "usage": "Corps principal", "force": "Trace la pertinence décisionnelle séparément de la norme.", "includegraphics": r"\\includegraphics[width=0.9\\textwidth]{fig_b1_harm_normalized.pdf}"},
        {"rang": 8, "figure_id": "fig_c4_selective_projection", "usage": "Corps principal", "force": "Clarifie unité, observables et readout sans prétendre à une projection linéaire.", "includegraphics": r"\\includegraphics[width=0.9\\textwidth]{fig_c4_selective_projection.pdf}"},
        {"rang": 9, "figure_id": "fig_e1_segmentation_robust_accuracy", "usage": "Annexe proche", "force": "Soutient la non-nécessité du couplage global complet; comparaison confondue.", "includegraphics": r"\\includegraphics[width=0.9\\textwidth]{fig_e1_segmentation_robust_accuracy.pdf}"},
        {"rang": 10, "figure_id": "fig_f1_responsibility_robust_accuracy", "usage": "Annexe proche", "force": "Documente l'absence de monotonie avec la responsabilité quantique.", "includegraphics": r"\\includegraphics[width=0.9\\textwidth]{fig_f1_responsibility_robust_accuracy.pdf}"},
    ])

    with sqlite3.connect(DATABASE_PATH) as connection:
        for table_name, frame in {
            "gain": gain,
            "interface": interface,
            "normalization": normalization,
            "direction": direction,
            "residual": residual,
            "robust": robust,
            "validation": validation,
            "block_effects": block_effects,
            "hypotheses": hypotheses,
            "ranking": ranking,
        }.items():
            frame.to_sql(table_name, connection, if_exists="replace", index=False)
        for sql in SOURCE_SQL.values():
            pd.read_sql_query(sql, connection)

    manifest_sources = [
        _source("src_layerwise", "Traces de gain RMS", "outputs/mechanistic_visualizations/fig_a1_gain_layerwise/fig_a1_gain_layerwise.csv"),
        _source("src_interface", "Transitions d'interface", "outputs/mechanistic_visualizations/fig_c1_interface_rms/fig_c1_interface_rms.csv"),
        _source("src_normalization", "Sensibilité de normalisation", "outputs/mechanistic_visualizations/normalization_sensitivity.csv"),
        _source("src_direction", "Transmission par direction source", "src/analytics/attacks/mechanistic_visualizations.py"),
        _source("src_residual", "Branches du HVQL résiduel", "outputs/mechanistic_visualizations/fig_g1_residual_rms/fig_g1_residual_rms.csv"),
        _source("src_attack", "Agrégats d'attaque exacts", "logs/ATTACKS/attack_experiment/tables/"),
        _source("src_validation", "Validation des checkpoints", "outputs/mechanistic_visualizations/checkpoint_alignment.csv"),
        _source("src_blocks", "Synthèse des effets par bloc", "outputs/mechanistic_visualizations/report_source.sqlite"),
        _source("src_hypotheses", "Audit des hypothèses", "outputs/mechanistic_visualizations/report_source.sqlite"),
        _source("src_ranking", "Classement des figures", "outputs/mechanistic_visualizations/report_source.sqlite"),
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Propagation adversariale dans les classifieurs hybrides MNIST",
            "description": "Rapport technique mécanistique, figures reproductibles et audit de neuf hypothèses.",
            "generatedAt": GENERATED_AT,
            "filters": [],
            "cards": [],
            "charts": [
                {
                    "id": "chart_gain", "title": "Gain RMS par étape architecturale",
                    "subtitle": "FGSM blanche, ε = 0,10; médianes hiérarchiques sur 500 exemples communs correctement classés par graine.",
                    "type": "line", "dataset": "gain", "sourceId": "src_layerwise", "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "stage_fr", "type": "ordinal", "label": "Étape architecturale"},
                        "y": {"field": "gain_rms", "type": "quantitative", "label": "Gain RMS"},
                        "color": {"field": "model", "type": "nominal", "label": "Architecture"},
                        "tooltip": [
                            {"field": "model", "type": "nominal", "label": "Modèle"},
                            {"field": "gain_rms", "type": "quantitative", "label": "Gain RMS", "format": "number"},
                            {"field": "n_samples", "type": "quantitative", "label": "Observations"},
                        ],
                    },
                },
                {
                    "id": "chart_interface", "title": "Rétention RMS et concentration du dommage à l'interface",
                    "subtitle": "Chaque point est une architecture; les valeurs HQE de dommage sont instables et qualifiées dans le texte adjacent.",
                    "type": "scatter", "dataset": "interface", "sourceId": "src_interface", "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "retention_rms", "type": "quantitative", "label": "Rétention RMS"},
                        "y": {"field": "harm_change_normalized", "type": "quantitative", "label": "Concentration du dommage"},
                        "color": {"field": "model", "type": "nominal", "label": "Architecture"},
                        "tooltip": [
                            {"field": "model", "type": "nominal", "label": "Modèle"},
                            {"field": "harm_ratio_coverage", "type": "quantitative", "label": "Couverture", "format": "percent"},
                        ],
                    },
                },
                {
                    "id": "chart_residual", "title": "Déplacement RMS des branches résiduelles",
                    "subtitle": "Contributions après application de l'échelle résiduelle entraînée; FGSM blanche, ε = 0,10.",
                    "type": "bar", "dataset": "residual", "sourceId": "src_residual", "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "branch_fr", "type": "nominal", "label": "Branche"},
                        "y": {"field": "delta_rms", "type": "quantitative", "label": "Déplacement RMS"},
                        "tooltip": [
                            {"field": "ci_low", "type": "quantitative", "label": "IC bas", "format": "number"},
                            {"field": "ci_high", "type": "quantitative", "label": "IC haut", "format": "number"},
                        ],
                    },
                },
                {
                    "id": "chart_robust", "title": "Exactitude robuste des hybrides tardifs",
                    "subtitle": "FGSM blanche, ε = 0,10, ensemble de test complet de 10 000 images par graine.",
                    "type": "bar", "dataset": "robust", "sourceId": "src_attack", "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "model", "type": "nominal", "label": "Architecture"},
                        "y": {"field": "value", "type": "quantitative", "label": "Exactitude robuste"},
                        "tooltip": [
                            {"field": "lower", "type": "quantitative", "label": "Minimum des graines", "format": "percent"},
                            {"field": "upper", "type": "quantitative", "label": "Maximum des graines", "format": "percent"},
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "table_direction", "title": "Transmission selon la source et la cible",
                    "subtitle": "Médianes hiérarchiques à l'interface, ε = 0,10; auto-FGSM, transferts et aléatoire L2-apparié.",
                    "dataset": "direction", "sourceId": "src_direction", "defaultSort": {"field": "model", "direction": "asc"},
                    "columns": [
                        {"field": "model", "label": "Cible", "type": "text"},
                        {"field": "source_fr", "label": "Source", "type": "text"},
                        {"field": "retention_rms", "label": "Rétention RMS", "format": "number"},
                        {"field": "harm_change", "label": "Δ dommage", "format": "number"},
                        {"field": "alignment_change", "label": "Δ alignement", "format": "number"},
                    ],
                },
                {
                    "id": "table_blocks", "title": "Étape où le déplacement devient décisionnel",
                    "subtitle": "Synthèse des traces directes; « dominante » indique le principal changement observé, pas une cause isolée.",
                    "dataset": "block_effects", "sourceId": "src_blocks", "defaultSort": {"field": "modele", "direction": "asc"},
                    "columns": [
                        {"field": "modele", "label": "Modèle", "type": "text"},
                        {"field": "premier_bloc", "label": "Premier bloc", "type": "text"},
                        {"field": "encodeur", "label": "Encodeur", "type": "text"},
                        {"field": "interface_quantique", "label": "Interface quantique", "type": "text"},
                        {"field": "tete_finale", "label": "Tête finale", "type": "text"},
                        {"field": "etape_dominante", "label": "Étape dominante", "type": "text"},
                    ],
                },
                {
                    "id": "table_validation", "title": "Accord des checkpoints avec les prédictions sauvegardées",
                    "subtitle": "Minimum d'accord et maximum d'écart de logit sur les deux graines et 10 000 images par checkpoint attaqué.",
                    "dataset": "validation", "sourceId": "src_validation", "defaultSort": {"field": "model", "direction": "asc"},
                    "columns": [
                        {"field": "model", "label": "Modèle", "type": "text"},
                        {"field": "graines", "label": "Graines", "format": "number"},
                        {"field": "accord_predictions_min", "label": "Accord min.", "format": "percent"},
                        {"field": "ecart_logit_max", "label": "Écart logit max.", "format": "number"},
                        {"field": "statut", "label": "Statut", "type": "text"},
                    ],
                },
                {
                    "id": "table_hypotheses", "title": "Audit des hypothèses",
                    "subtitle": "Verdicts limités à l'évidence disponible; les associations ne sont pas converties en causalité.",
                    "dataset": "hypotheses", "sourceId": "src_hypotheses", "defaultSort": {"field": "hypothese", "direction": "asc"},
                    "columns": [
                        {"field": "hypothese", "label": "Hypothèse", "type": "text"},
                        {"field": "verdict", "label": "Verdict", "type": "text"},
                        {"field": "pour", "label": "Évidence favorable", "type": "text"},
                        {"field": "contre", "label": "Évidence contraire / limite", "type": "text"},
                        {"field": "niveau", "label": "Niveau d'inférence", "type": "text"},
                    ],
                },
                {
                    "id": "table_ranking", "title": "Sélection et ordre narratif des figures",
                    "subtitle": "Ordre recommandé pour le manuscrit; chaque répertoire contient PDF, SVG, PNG, CSV, méthode, légende et script.",
                    "dataset": "ranking", "sourceId": "src_ranking", "defaultSort": {"field": "rang", "direction": "asc"},
                    "columns": [
                        {"field": "rang", "label": "Rang", "format": "number"},
                        {"field": "figure_id", "label": "Figure", "type": "text"},
                        {"field": "usage", "label": "Usage", "type": "text"},
                        {"field": "force", "label": "Raison", "type": "text"},
                        {"field": "includegraphics", "label": "LaTeX", "type": "text"},
                    ],
                },
                {
                    "id": "table_normalization", "title": "Sensibilité à la constante de normalisation",
                    "subtitle": "Changement médian de dommage pour τ ∈ {10⁻⁴, 10⁻³, 10⁻²}; couverture après seuil du dommage d'entrée.",
                    "dataset": "normalization", "sourceId": "src_normalization", "defaultSort": {"field": "model", "direction": "asc"},
                    "columns": [
                        {"field": "model", "label": "Modèle", "type": "text"},
                        {"field": "epsilon", "label": "Epsilon", "format": "number"},
                        {"field": "tau", "label": "Tau", "type": "text"},
                        {"field": "harm_change", "label": "Δ dommage", "format": "number"},
                        {"field": "normalization_coverage", "label": "Couverture", "format": "percent"},
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Propagation adversariale dans les classifieurs hybrides MNIST"},
                {"id": "technical_summary", "type": "markdown", "body": "## Conclusion technique\n\n**Le résultat central est soutenu, mais sous une forme directionnelle plutôt que dissipative.** Les blocs restructurent successivement le déplacement. Dans les hybrides tardifs, le vecteur de mesures est généralement plus petit en RMS que le latent préquantique, tandis que sa composante alignée sur la réduction de marge augmente. Cela est compatible avec une interface sélective par les observables et le readout; cela ne permet pas d'attribuer causalement l'effet au caractère quantique plutôt qu'au goulot, à la mesure ou à la tête classique.\n\n**HVQL et HPVQL combinent une représentation amont riche avec un rôle quantique intermédiaire.** Leur avantage faible-régime persiste sans couplage global complet. À l'inverse, le HVQL résiduel transporte l'essentiel du déplacement par le contournement classique.\n\n**Deux réserves empêchent une conclusion plus forte.** HQE a une normalisation de dommage instable, et les comparaisons de segmentation ou de responsabilité quantique ne contrôlent pas capacité, marge et optimisation."},
                {"id": "finding_stages", "type": "markdown", "sourceId": "src_layerwise", "body": "## L'effet du premier bloc est souvent renversé plus tard\n\nLes convolutions du CNN, la tige du ResNet18, la projection du MLPMixer et l'encodeur initial de HVQL atténuent d'abord le déplacement RMS. Les blocs profonds inversent ensuite cette atténuation. Les deux MLP amplifient au contraire dès leur première projection, mais avec des trajectoires différentes; le grand MLP constitue donc un contrôle indispensable. Cette figure indique une propagation non monotone, sans établir que localité ou mélange global en sont la cause."},
                {"id": "gain_chart", "type": "chart", "chartId": "chart_gain"},
                {"id": "finding_interface", "type": "markdown", "sourceId": "src_interface", "body": "## Une mesure plus petite peut être plus nuisible\n\nHVQL, HPVQL et HVQM se déplacent vers une représentation mesurée de RMS plus faible tout en augmentant l'alignement nuisible et le dommage projeté. HQE montre la rétention la plus faible, mais son dommage normalisé est trop sensible au dénominateur et à la graine pour servir de preuve principale. Le motif est compatible avec une sélection directionnelle par encodage, évolution unitaire, observables restreintes et readout; il ne prouve pas un débruitage unitaire."},
                {"id": "interface_chart", "type": "chart", "chartId": "chart_interface"},
                {"id": "finding_direction", "type": "markdown", "sourceId": "src_direction", "body": "## La transmission dépend de la direction source\n\nLes directions auto-générées, transférées et aléatoires de norme appariée ne traversent pas les interfaces de la même façon. HQE n'est donc pas un suppresseur uniforme : sa rétention varie fortement selon la source. L'association entre dommage à la mesure et ASR entrant reste faible et incertaine; la compatibilité d'interface n'explique qu'une partie possible du transfert."},
                {"id": "direction_table", "type": "table", "tableId": "table_direction"},
                {"id": "finding_residual", "type": "markdown", "sourceId": "src_residual", "body": "## Le contournement résiduel transporte l'essentiel du déplacement\n\nDans le HVQL résiduel, la contribution L2 médiane du chemin classique est environ dix-sept fois celle de la branche quantique après application de l'échelle apprise. La fusion reste dominée par ce chemin. Cela soutient une contribution fonctionnelle de l'interface sélective dans le modèle standard, mais n'isole pas une cause spécifiquement quantique."},
                {"id": "residual_chart", "type": "chart", "chartId": "chart_residual"},
                {"id": "finding_performance", "type": "markdown", "sourceId": "src_attack", "body": "## La responsabilité quantique n'ordonne pas la robustesse\n\nHVQL et HPVQL gardent les meilleures exactitudes propres et robustes de ce groupe tardif. HVQM, où la mesure quantique porte une responsabilité plus directe, est moins performant; HQE ne restaure pas un ordre monotone. Les différences simultanées d'encodeur, de paramètres et de marge empêchent toutefois une interprétation causale."},
                {"id": "robust_chart", "type": "chart", "chartId": "chart_robust"},
                {"id": "scope_metrics", "type": "markdown", "body": "## Portée, cohortes et définitions\n\nLes métriques internes utilisent 500 exemples MNIST communs correctement classés par tous les modèles de chaque comparaison, pour chacune des graines 0 et 42. La perturbation principale est une FGSM blanche propre au modèle à ε = 0,10; ε = 0,05, les transferts et un bruit aléatoire L2-apparié servent de sensibilités. Le gain compare des RMS par dimension. Le dommage H = −∇m·Δa conserve son signe; l'alignement C = −cos(∇m,Δa) et la sensibilité S = ‖∇m‖₂ vérifient H = S‖Δa‖₂C. Les exactitudes propre et robuste utilisent les 10 000 images complètes, sans conditionnement; l'ASR blanche reste distincte et conditionnée aux images propres correctes."},
                {"id": "block_summary", "type": "table", "tableId": "table_blocks"},
                {"id": "validation_method", "type": "markdown", "sourceId": "src_validation", "body": "## Les labels historiques et checkpoints sont réconciliés\n\nLe dossier historique nommé HVQM contient bien l'architecture sémantique HVQL riche; le dossier nommé HVQL contient l'architecture sémantique HVQM plus petite. Les checkpoints attaqués reproduisent exactement les prédictions float16 sauvegardées sur 10 000 images. Le grand MLP [1024,512,64,64] est un checkpoint MLP authentique non attaqué; le faux checkpoint MLP contenant LeNet a été exclu de toute jointure d'attaque."},
                {"id": "validation_table", "type": "table", "tableId": "table_validation"},
                {"id": "hypothesis_intro", "type": "markdown", "body": "## Les hypothèses causales restent plus fragiles que les constats de propagation\n\nL'audit distingue mesures directes, conséquences mathématiques, associations empiriques et mécanismes plausibles. H4, H6 et l'absence de monotonie simple de H7 sont les conclusions les mieux soutenues; H8 et H9 restent non testables avec les contrôles disponibles."},
                {"id": "hypothesis_table", "type": "table", "tableId": "table_hypotheses"},
                {"id": "sensitivity_intro", "type": "markdown", "sourceId": "src_normalization", "body": "## Les sensibilités confirment le motif, sauf pour HQE\n\nHVQL, HPVQL et HVQM conservent le signe et l'ordre de grandeur du changement de dommage lorsque τ varie entre 10⁻⁴ et 10⁻² et lorsque ε passe de 0,10 à 0,05. HQE conserve une rétention très faible, mais sa couverture de normalisation chute et l'écart entre graines est large. Les versions HQE normalisées appartiennent donc à l'annexe ou doivent être remplacées par dommage brut et couverture."},
                {"id": "normalization_table", "type": "table", "tableId": "table_normalization"},
                {"id": "figure_selection", "type": "markdown", "body": "## Ordre narratif recommandé pour le manuscrit\n\nCommencer par les trajectoires RMS, introduire ensuite la dissociation norme–dommage, puis montrer l'interface sélective. Les matrices de direction testent la généralité de ce mécanisme. Le résidu fournit enfin une quasi-ablation fonctionnelle : une voie classique transporte le déplacement autour de l'interface. Les comparaisons de segmentation et de responsabilité servent de soutien descriptif, pas de preuve causale."},
                {"id": "ranking_table", "type": "table", "tableId": "table_ranking"},
                {"id": "limitations", "type": "markdown", "body": "## Limites, résultats instables et omissions\n\n- **À ne pas utiliser comme preuve principale :** dommage normalisé HQE, corrélation transmission–ASR (ρ de Spearman ≈ 0,27; intervalle bootstrap incluant zéro), et toute lecture causale du balayage HPVQL.\n- **Non généré faute de contrôle adéquat :** rang effectif, angles principaux, décomposition état complet / sous-espace non observable, PGD/APGD, trajectoires de marge finale par bloc et balayage d'échelle résiduelle.\n- **Confondeurs persistants :** précision propre, distance à la frontière, largeur latente, dimension mesurée, nombre de paramètres, dropout appris, optimisation et force propre de l'attaque.\n- **Portée statistique :** deux graines seulement; les bootstrap hiérarchiques décrivent l'incertitude des cohortes mais ne justifient pas une significativité inter-entraînement."},
                {"id": "next_steps", "type": "markdown", "body": "## Expériences qui réduiraient le plus vite l'incertitude\n\n1. Réentraîner un HVQL avec encodeur fixé et remplacer seulement l'interface par identité, bloc classique dimensionnellement apparié, circuit global ou circuits partitionnés.\n2. Balayer largeur latente et dimension mesurée en appariant le nombre de paramètres et la précision propre.\n3. Mesurer marge finale exacte, H brut, S et C sur les mêmes exemples, puis ajouter PGD à faible nombre d'itérations.\n4. Pour HQE, préenregistrer un seuil de dénominateur, publier la couverture et privilégier H brut ou un ratio de médianes.\n5. Réentraîner le HVQL résiduel à plusieurs échelles au lieu de conclure d'un balayage d'inférence."},
                {"id": "further_questions", "type": "markdown", "body": "## Questions encore ouvertes\n\n- La concentration du dommage persiste-t-elle lorsque la tête classique est remplacée par un readout apparié?\n- Les observables mesurées correspondent-elles à un sous-espace nuisible stable ou dépendant de l'exemple?\n- La forte robustesse faible-régime de HVQL et HPVQL survit-elle après appariement des marges propres?\n- Le comportement de HQE reflète-t-il la compression, l'encodage quantique ou la tête aval?\n\n**Conclusion mécanistique sûre :** l'interface quantique tardive ne supprime pas génériquement la perturbation. Elle transforme un déplacement latent et n'en expose qu'une partie aux mesures; dans les modèles analysés, cette partie peut être plus petite en RMS mais davantage alignée avec la réduction de marge, puis amplifiée par le readout classique."},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "gain": _records(gain),
                "interface": _records(interface),
                "residual": _records(residual),
                "residual_summary": _records(residual_summary),
                "outcomes": _records(outcomes.assign(clean_accuracy_context=outcomes.groupby("model_key")["value"].transform(lambda s: s.iloc[0]))),
                "robust": _records(robust),
                "direction": _records(direction),
                "validation": _records(validation),
                "normalization": _records(normalization),
                "hypotheses": _records(hypotheses),
                "block_effects": _records(block_effects),
                "ranking": _records(ranking),
            },
            "accessIssues": [],
        },
        "sources": [
            _query_source("src_layerwise", "Résume les traces RMS auto-FGSM par modèle, étape et graine.", ["main.gain"], {"epsilon": "0.10", "cohort": "500 exemples communs-corrects par graine"}),
            _query_source("src_interface", "Agrège les transitions latent préquantique vers mesures.", ["main.interface"], {"epsilon": "0.10", "attack": "auto-FGSM"}),
            _query_source("src_normalization", "Compare τ et epsilon pour le dommage normalisé.", ["main.normalization"], {"tau": "1e-4, 1e-3, 1e-2", "epsilon": "0.05, 0.10"}),
            _query_source("src_direction", "Joint les trois métriques de transmission source–cible.", ["main.direction"], {"epsilon": "0.10"}),
            _query_source("src_residual", "Agrège les contributions quantique, classique et fusionnée du HVQL résiduel.", ["main.residual"], {"epsilon": "0.10", "attack": "auto-FGSM"}),
            _query_source("src_attack", "Lit les métriques exactes calculées sur les logits adversariaux sauvegardés.", ["main.robust"], {"epsilon": "0.10", "n_samples": "10000"}),
            _query_source("src_validation", "Compare les prédictions exécutées aux prédictions propres sauvegardées.", ["main.validation"], {"n_samples": "10000 par checkpoint"}),
            _query_source("src_blocks", "Synthèse prudente des effets observés par bloc.", ["main.block_effects"], {"seeds": "0, 42"}),
            _query_source("src_hypotheses", "Audit prudent des neuf hypothèses.", ["main.hypotheses"], {"verdicts": "soutenu, partiel, contradit ou non testable"}),
            _query_source("src_ranking", "Classe les figures selon leur force probante et leur place recommandée.", ["main.ranking"], {"formats": "PDF, SVG, PNG"}),
        ],
        "package_info": {"originUrl": "artifact://mnist-mechanistic-propagation", "controls": {"edit": False, "refresh": False}},
    }

    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(ARTIFACT_PATH)


if __name__ == "__main__":
    main()
