    def log_resources(self, step: Optional[int] = None) -> None:
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_used_gb = (mem.total - mem.available) / (1024 ** 3)
        metrics = {
            "cpu_percent": cpu_pct,
            "mem_used_gb": mem_used_gb
        }
        if torch.cuda.is_available():
            metrics["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        self.log_dict("performance", metrics, step=step)