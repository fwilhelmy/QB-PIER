class AttackRepository(BaseRepository):
    def profile(self, input_shape, epochs):
        return ExperimentProfiler(str(self.base_dir), input_shape, epochs)

    def save(self, results: pd.DataFrame):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.base_dir / "results.csv", index=False)
        results.to_json(self.base_dir / "results.json", orient="records", indent=2)

    def load(self) -> pd.DataFrame:
        path = self.base_dir / "results.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def existing_pairs(self) -> Tuple[List[Dict[str, object]], set[Tuple[int, str, str]]]:
        data = self.load()
        if data.empty:
            return [], set()
        pairs = set(zip(data["seed"], data["source_model"], data["target_model"]))
        return data.to_dict("records"), pairs

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

    def save_gradients(self, *, attacker_model: str, seed: int, gradients: List[Tuple[Tensor, Tensor]]) -> str:
        model_dir = self.base_dir / attacker_model
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"gradients_seed{seed}.pt"
        torch.save({"seed": seed, "model": attacker_model, "gradients": gradients}, path)
        return str(path)

    def save_logits(self, *, model: str, seed: int, logits: Tensor) -> str:
        model_dir = self.base_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"logits_seed{seed}.pt"
        torch.save({"seed": seed, "model": model, "logits": logits}, path)
        return str(path)
