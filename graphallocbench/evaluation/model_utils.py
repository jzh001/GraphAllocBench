from ..constants import RL_Model, INFERENCE_DEVICE
from pathlib import Path
import sys
import torch


def _resolve_model_path(model_path: str | None) -> str | None:
    if model_path is None:
        return None
    p = Path(model_path)
    # if absolute or exists as given
    if p.exists():
        return str(p)

    # try common extensions
    for ext in (".zip", ".pt", ".pth"):
        if not str(p).endswith(ext):
            cand = p.with_suffix(ext)
            if cand.exists():
                return str(cand)

    # If path is relative, try resolving from repo root upwards
    start = Path.cwd()
    for parent in [start] + list(start.parents):
        cand = parent / model_path
        if cand.exists():
            return str(cand)
        for ext in (".zip", ".pt", ".pth"):
            cand2 = parent / (str(model_path) + ext)
            if cand2.exists():
                return str(cand2)

    # Try package-relative models directory
    try:
        repo_root = Path(__file__).resolve().parents[2]
        cand = repo_root / 'models' / model_path
        if cand.exists():
            return str(cand)
        for ext in (".zip", ".pt", ".pth"):
            cand2 = repo_root / 'models' / (str(model_path) + ext)
            if cand2.exists():
                return str(cand2)
    except Exception:
        pass

    return None


def load_model(model_path, env=None):
    """Load an RL model with helpful path resolution and errors.

    Returns a loaded model. Raises FileNotFoundError if path cannot be found.
    """
    resolved = _resolve_model_path(model_path)
    if resolved is None:
        raise FileNotFoundError(f"Model path not found: {model_path}. Tried cwd, parent dirs, and package models/.")

    # Ensure device is valid for SB3 (fallback to cpu if cuda not available)
    device_to_use = INFERENCE_DEVICE
    try:
        if isinstance(device_to_use, str) and device_to_use.startswith("cuda") and not torch.cuda.is_available():
            device_to_use = "cpu"
    except Exception:
        device_to_use = "cpu"

    # If resolved is a directory, pick the first .zip (or latest) inside
    res_path = Path(resolved)
    if res_path.is_dir():
        zips = sorted([p for p in res_path.glob('**/*.zip')], key=lambda p: p.stat().st_mtime, reverse=True)
        if len(zips) == 0:
            raise FileNotFoundError(f"No .zip model files found inside directory: {resolved}")
        resolved = str(zips[0])

    try:
        # Provide compatibility aliases for modules used when models were
        # originally pickled outside this package (cloudpickle will import
        # them by name). Map common legacy top-level module names to the
        # current package locations so deserialization succeeds.
        import importlib

        legacy_aliases = {
            "city_env": "graphallocbench.city_env",
            "city_env.env_model": "graphallocbench.city_env.env_model",
            "architectures": "graphallocbench.city_env.architectures",
            "components": "graphallocbench.city_env.components",
        }
        for old_name, new_name in legacy_aliases.items():
            if old_name not in sys.modules:
                try:
                    mod = importlib.import_module(new_name)
                    sys.modules[old_name] = mod
                except Exception:
                    # ignore failures; SB3 load may not need all aliases
                    pass

        model = RL_Model.load(resolved, env=env, device=device_to_use)
        return model
    except Exception as e:
        # Re-raise with context to help debug
        raise RuntimeError(f"Failed to load model from '{resolved}': {e}") from e
