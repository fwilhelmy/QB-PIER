from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseRepository(ABC):
    """Base class for repository objects that persist experiment artifacts."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def profile(self, *args, **kwargs):
        """Return a profiling helper/context manager for the repository."""

    @abstractmethod
    def save(self, *args, **kwargs):
        """Persist artifacts handled by the repository."""

    @abstractmethod
    def load(self, *args, **kwargs):
        """Load artifacts handled by the repository."""