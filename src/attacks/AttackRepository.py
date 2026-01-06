from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor

from BaseRepository import BaseRepository
from attacks.AttackResult import AttackResult
from models.ModelRepository import ModelProfiler


class AttackRepository(BaseRepository):
    def __init__(self, base_dir: str | Path):
        super().__init__(base_dir)
        # Crawl the repository to initialize any needed structures

    def save(self, results: pd.DataFrame | Iterable[AttackResult]):
        df = self._to_dataframe(results)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.base_dir / "results.csv", index=False)
        df.to_json(self.base_dir / "results.json", orient="records", indent=2)
        return df

    def load(self) -> pd.DataFrame:
        path = self.base_dir / "results.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def _to_dataframe(self, results: pd.DataFrame | Iterable[AttackResult]) -> pd.DataFrame:
        if isinstance(results, pd.DataFrame):
            return results
        return pd.DataFrame([r.to_record() for r in results])

    def existing_pairs(self) -> Tuple[List[AttackResult], set[Tuple[int, str, str]]]:
        data = self.load()
        if data.empty:
            return [], set()
        pairs = set(zip(data["seed"], data["source_model"], data["target_model"]))
        results = [AttackResult.from_record(record) for record in data.to_dict("records")]
        return results, pairs

    def save_predictions(
        self,
        *,
        target_model: str,
        seed: int,
        source_model: str,
        epsilon: float,
        predictions: Tensor,
        targets: Sequence[int],
    ) -> str:
        model_dir = self.base_dir / target_model
        model_dir.mkdir(parents=True, exist_ok=True)
        file_path = model_dir / "predictions.csv"
        df = pd.DataFrame(
            {
                "seed": seed,
                "source_model": source_model,
                "target_model": target_model,
                "epsilon": epsilon,
                "sample_index": range(len(predictions)),
                "target": targets,
                "prediction": predictions.cpu().tolist(),
            }
        )
        df.to_csv(file_path, mode="a", header=not file_path.exists(), index=False)
        return str(file_path)

    def save_attack(
        self,
        *,
        seed: int,
        source_model: str,
        target_model: str,
        epsilon: float,
        accuracy: float,
        predictions: Tensor,
        targets: Sequence[int],
        num_samples: int,
        gradients_path: str | None = None,
        gradients: List[Tuple[Tensor, Tensor]] | None = None,
    ) -> AttackResult:
        if gradients_path is None and gradients is not None:
            gradients_path = self.save_gradients(
                attacker_model=source_model, seed=seed, gradients=gradients
            )

        predictions_path = self.save_predictions(
            target_model=target_model,
            seed=seed,
            source_model=source_model,
            epsilon=epsilon,
            predictions=predictions,
            targets=targets,
        )

        return AttackResult(
            seed=seed,
            source_model=source_model,
            target_model=target_model,
            epsilon=epsilon,
            accuracy=accuracy,
            num_samples=num_samples,
            predictions_path=predictions_path,
            gradients_path=gradients_path,
        )

    def save_results(self, results: Iterable[AttackResult]) -> pd.DataFrame:
        return self.save(results)

    def save_gradients(self, *, attacker_model: str, seed: int, gradients: List[Tuple[Tensor, Tensor]]) -> str:
        model_dir = self.base_dir / attacker_model
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"gradients_seed{seed}.pt"
        torch.save({"seed": seed, "model": attacker_model, "gradients": gradients}, path)
        return str(path)

    def save_logits(self, *, logits: Tensor, result: AttackResult) -> str:
        model_dir = self.base_dir / result.target_model
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / (
            f"logits_seed{result.seed}_src{result.source_model}_eps{result.epsilon}.pt"
        )
        torch.save(
            {
                "seed": result.seed,
                "source_model": result.source_model,
                "target_model": result.target_model,
                "epsilon": result.epsilon,
                "logits": logits,
            },
            path,
        )
        return str(path)
