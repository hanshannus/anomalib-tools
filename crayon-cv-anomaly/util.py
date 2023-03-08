from pathlib import Path
from omegaconf import DictConfig
from anomalib.config import get_configurable_parameters


def _resolve_path(path: str) -> str:
    """Interpret '~/' to '$HOME' directory."""
    if path is None:
        path = "."
    elif len(path) == 1 and path == "~":
        path = str(Path.home())
    elif len(path) == 2 and path == "~/":
        path = str(Path.home())
    elif len(path) > 2 and path[:2] == "~/":
        path = str(Path.home() / path[2:])
    return path


def load_config(config_path: str) -> DictConfig:
    """Load configuration file and preprocess parameters."""
    cfg = get_configurable_parameters(config_path=config_path)
    # ensure that mandatory groups are present in configuration
    if "dataset" not in cfg:
        cfg.dataset = {}
    if "project" not in cfg:
        cfg.project = {}
    if "inferencer" not in cfg:
        cfg.inferencer = {}
    if "visualization" not in cfg:
        cfg.visualization = {}
    # replace '~' with '$HOME' and 'None' with '.' directory in paths
    cfg.dataset.path = _resolve_path(cfg.dataset.get("path"))
    cfg.project.path = _resolve_path(cfg.project.get("path"))
    cfg.inferencer.path = _resolve_path(cfg.inferencer.get("path"))
    # anomalib mask_dir path is absolute; make it relative to dataset path
    if cfg.dataset.get("format") == "folder":
        cfg.dataset.mask_dir = f"{cfg.dataset.path}/{cfg.dataset.mask_dir}"
    return cfg
