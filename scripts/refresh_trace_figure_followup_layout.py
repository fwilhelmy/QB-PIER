"""Apply the final thesis label/tick corrections to the existing figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figure_redesign_from_trace"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analytics.attacks.trace_figure_followup import refresh_requested_layout_corrections


def main() -> None:
    result = refresh_requested_layout_corrections(OUTPUT_ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
