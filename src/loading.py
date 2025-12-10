from __future__ import annotations

from typing import Any, Dict, List

from repositories import ModelRepository


def crawl_experiments(path: str) -> List[str]:
    return [str(p) for p in ModelRepository(path).crawl()]


def load_experiment(path: str, *, load: str = "final", map_location: str = "cpu") -> Dict[str, Dict[Any, Any]]:
    return ModelRepository(path).load(load=load, map_location=map_location)
