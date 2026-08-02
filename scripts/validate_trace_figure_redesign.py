"""Validate the exported trace-backed figure package and portable report."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
import shutil

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figure_redesign_from_trace"


def main() -> None:
    manifest = pd.read_csv(OUTPUT_DIR / "figure_manifest.csv")
    if manifest["figure_id"].duplicated().any():
        raise ValueError("Duplicate figure_id values")
    rows = []
    for _, row in manifest.iterrows():
        paths = {
            "png": OUTPUT_DIR / str(row["filename"]),
            "pdf": OUTPUT_DIR / str(row["pdf_filename"]),
            "svg": OUTPUT_DIR / str(row["svg_filename"]),
            "csv": OUTPUT_DIR / str(row["data_csv"]),
            "method": OUTPUT_DIR / str(row["method_file"]),
        }
        exists = {name: path.exists() and path.stat().st_size > 0 for name, path in paths.items()}
        png_width = png_height = 0
        dpi_x = dpi_y = 0.0
        if exists["png"]:
            with Image.open(paths["png"]) as image:
                png_width, png_height = image.size
                dpi_x, dpi_y = (float(value) for value in image.info.get("dpi", (0.0, 0.0)))
        pdf_magic = paths["pdf"].read_bytes()[:4] == b"%PDF" if exists["pdf"] else False
        svg_root = "<svg" in paths["svg"].read_text(encoding="utf-8")[:1000] if exists["svg"] else False
        csv_rows = len(pd.read_csv(paths["csv"])) if exists["csv"] else 0
        status = (
            "passed"
            if all(exists.values()) and min(png_width, png_height) >= 1000
            and dpi_x >= 299 and dpi_y >= 299 and pdf_magic and svg_root and csv_rows > 0
            else "failed"
        )
        rows.append({
            "figure_id": row["figure_id"],
            **{f"{name}_exists": value for name, value in exists.items()},
            "png_width": png_width,
            "png_height": png_height,
            "dpi_x": dpi_x,
            "dpi_y": dpi_y,
            "pdf_magic": pdf_magic,
            "svg_root": svg_root,
            "csv_rows": csv_rows,
            "status": status,
        })
    qa = pd.DataFrame(rows)
    qa.to_csv(OUTPUT_DIR / "figure_file_qa.csv", index=False)

    report_path = OUTPUT_DIR / "report.html"
    report = report_path.read_text(encoding="utf-8")
    decoded = html.unescape(report)
    headings = [
        "Synthèse technique",
        "Pourquoi les figures précédentes",
        "Données, cohortes et définitions",
        "Stratégie de regroupement",
        "Une figure par modèle",
        "Les hybrides tardifs",
        "Le routage décisionnel",
        "Les variations internes",
        "Une interface mesurée",
        "Les trois figures",
        "Inventaire de sortie",
        "Galerie complète",
    ]
    gallery_captions = len(re.findall(r"<figcaption>figures/[^<]+\.png</figcaption>", decoded))
    external_requests = len(re.findall(r'(?:src|href)=["\']https?://', decoded, flags=re.IGNORECASE))
    receipt = json.loads((OUTPUT_DIR / "html_delivery_receipt.json").read_text(encoding="utf-8"))
    package = {
        "figure_count": int(len(manifest)),
        "figure_qa_passed": int((qa["status"] == "passed").sum()),
        "figure_qa_failed": int((qa["status"] != "passed").sum()),
        "gallery_caption_count": gallery_captions,
        "required_headings_present": {heading: heading in decoded for heading in headings},
        "external_src_or_href_count": external_requests,
        "artifact_bytes": (OUTPUT_DIR / "artifact.json").stat().st_size,
        "report_bytes": report_path.stat().st_size,
        "portable_validation": receipt["stages"]["validation"],
        "portable_packaging": receipt["stages"]["package"],
        "portable_verification": receipt["stages"]["verification"],
        "browser_qa_limitation": receipt.get("browserWarning", {}).get("code", ""),
    }
    package["status"] = "passed_with_structural_only_html_qa" if (
        package["figure_qa_failed"] == 0
        and gallery_captions == len(manifest)
        and all(package["required_headings_present"].values())
        and external_requests == 0
        and package["portable_validation"] == "passed"
        and package["portable_packaging"] == "passed"
    ) else "failed"
    (OUTPUT_DIR / "package_validation.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(__file__, OUTPUT_DIR / "scripts" / Path(__file__).name)
    if package["status"] == "failed":
        raise RuntimeError(json.dumps(package, ensure_ascii=False, indent=2))
    print(json.dumps(package, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

