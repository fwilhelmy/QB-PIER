"""Generate the retained-VQM white-box analysis and report inputs."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QAML_RESULTS_ROOT = Path(r"C:\Users\fwilh\Desktop\QAML Guide\results")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "whitebox_vqm_analysis"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analytics.attacks.whitebox_vqm_analysis import AnalysisPaths, run_analysis


def main() -> None:
    run_analysis(
        AnalysisPaths(
            project_root=PROJECT_ROOT,
            qaml_results_root=QAML_RESULTS_ROOT,
            output_dir=OUTPUT_DIR,
        )
    )


if __name__ == "__main__":
    main()

