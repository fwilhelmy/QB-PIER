"""Generate the final thesis figure set from existing trace artifacts."""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from analytics.attacks.trace_figure_followup import build_followup_package


OUTPUT_DIR = Path("outputs/figure_redesign_from_trace")


def main() -> None:
    result = build_followup_package(OUTPUT_DIR)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
