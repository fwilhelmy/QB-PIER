"""Generate the trace-backed mechanistic figure redesign package."""

from __future__ import annotations

from pathlib import Path
import json

from analytics.attacks.trace_figure_redesign import (
    OUTPUT_DIR,
    build_redesigned_figure_package,
)


OUTPUT_PATH = Path(OUTPUT_DIR)


def main() -> None:
    result = build_redesigned_figure_package(OUTPUT_PATH)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
