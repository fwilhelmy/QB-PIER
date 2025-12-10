class ModelRepository(BaseRepository):
    def profile(self, input_shape, epochs):
        return self.profiler(input_shape, epochs)

    def save(self, model_name: str, seed: int, model: Any, *, best: bool = False):
        model_root = self.base_dir / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        (model_root / ".model").write_text(model_name)
        seed_dir = model_root / str(seed)
        seed_dir.mkdir(exist_ok=True)
        ckpt_dir = seed_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / ("best.ckpt" if best else "final.ckpt")
        torch.save(getattr(model, "state_dict", lambda: model)(), ckpt_path)
        return ckpt_path

    def save_metrics(self, model_name: str, seed: int, container: MetricsContainer):
        (self.base_dir / model_name / str(seed)).mkdir(parents=True, exist_ok=True)
        container.save_metrics()

    def crawl(self) -> List[Path]:
        experiments: List[Path] = []
        for root, _, files in os.walk(self.base_dir):
            if ".model" in files:
                experiments.append(Path(root))
        experiments.sort()
        return experiments

    def _best_checkpoint(self, ckpt_dir: Path) -> Path:
        ckpts = sorted([f for f in ckpt_dir.iterdir() if f.suffix == ".ckpt"], key=self._score)
        return ckpts[0]

    @staticmethod
    def _score(path: Path) -> float:
        match = re.search(r"loss=(\d+.\d+)", path.name)
        return float(match.group(1)) if match else float("inf")

    def load_models(self, path: Path, *, best: bool, map_location: str) -> Dict[int, Any]:
        model_name = (path / ".model").read_text().strip()
        models: Dict[int, Any] = {}
        for seed_dir in path.iterdir():
            if not seed_dir.is_dir() or seed_dir.name == ".hydra" or not seed_dir.name.isdigit():
                continue
            ckpt = seed_dir / "final.ckpt"
            if best:
                ckpt = self._best_checkpoint(seed_dir / "checkpoints")
            model = ModelFactory.load(model_name, checkpoint=str(ckpt), map_location=map_location, log_dir=str(seed_dir))
            models[int(seed_dir.name)] = model
        return models

    def load_metrics(self, path: Path) -> Dict[int, MetricsContainer]:
        metrics: Dict[int, MetricsContainer] = {}
        for seed_dir in sorted(path.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            container = MetricsContainer(str(seed_dir))
            container.load_metrics(str(seed_dir))
            metrics[int(seed_dir.name)] = container
        return metrics

    def load(self, *, load: str = "final", map_location: str = "cpu") -> Dict[str, Dict[Any, Any]]:
        experiment: Dict[str, Dict[Any, Any]] = {}
        LoggingService(os.path.join(self.base_dir, datetime.now().strftime("%Y-%m-%d@%H-%M-%S")))
        for model_path in tqdm(self.crawl(), desc="Loading experiment"):
            name = os.path.relpath(model_path, self.base_dir)
            if load == "metrics":
                experiment[name] = self.load_metrics(model_path)
            elif load == "best":
                experiment[name] = self.load_models(model_path, best=True, map_location=map_location)
            else:
                experiment[name] = self.load_models(model_path, best=False, map_location=map_location)
        return experiment