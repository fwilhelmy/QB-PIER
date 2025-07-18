from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig):
    overrides = ["training/optimiser=momentum"]
    cfg = hydra.compose(
        config_name="config",
        overrides=overrides
    )
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()