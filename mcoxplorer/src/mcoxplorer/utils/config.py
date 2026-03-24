from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError("Config root must be a mapping")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = ["inputs", "output_dir", "sequence", "structure", "fusion", "modes", "external_tools"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ConfigError(f"Missing required keys: {missing}")
    for key in ["positive_fasta", "candidate_fasta", "positive_metadata_csv"]:
        if key not in cfg["inputs"]:
            raise ConfigError(f"Missing inputs.{key}")

    if "weights" not in cfg["fusion"]:
        raise ConfigError("Missing fusion.weights")
    for wk in ["sequence_score", "structure_score", "novelty_bonus", "false_positive_risk_penalty"]:
        if wk not in cfg["fusion"]["weights"]:
            raise ConfigError(f"Missing fusion.weights.{wk}")
