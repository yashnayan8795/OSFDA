"""
Configuration & Path Resolution
================================
Central config loader that resolves all paths relative to the project root,
ensuring notebooks, scripts, and tests all use the same config seamlessly.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def get_project_root() -> Path:
    """
    Walk upward from this file's location to find the project root.
    The root is identified by the presence of `pyproject.toml`.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not locate project root (no pyproject.toml found in parent chain)."
    )


PROJECT_ROOT = get_project_root()


def load_yaml(rel_path: str) -> Dict[str, Any]:
    """Load a YAML file relative to the project root."""
    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_main_config() -> Dict[str, Any]:
    """Load the main project config."""
    return load_yaml("configs/main_config.yaml")


def load_severity_rubric(version: str = "v1") -> Dict[str, Any]:
    """Load the severity rubric specification."""
    return load_yaml(f"configs/severity_rubric_{version}.yaml")


def load_category_taxonomy(version: str = "v1") -> Dict[str, Any]:
    """Load the category taxonomy specification."""
    return load_yaml(f"configs/category_taxonomy_{version}.yaml")


def load_cost_matrix() -> Dict[str, Any]:
    """Load the asymmetric cost matrix."""
    return load_yaml("configs/cost_matrix.yaml")


def load_feature_whitelist() -> Dict[str, Any]:
    """Load the feature whitelist for leakage gating."""
    return load_yaml("configs/feature_whitelist.yaml")


def resolve_path(rel_path: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / rel_path


def set_seeds(seed: Optional[int] = None):
    """Set random seeds for reproducibility across numpy, random, and torch."""
    if seed is None:
        config = load_main_config()
        seed = config.get("seeds", {}).get("global_seed", 42)

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
