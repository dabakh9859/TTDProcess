"""Top-level handlers exposed over JSON-RPC.

Each handler takes (params: dict, emit: callable) and returns a JSON-serialisable
result. ``emit("event_name", data)`` may be called to stream progress to the
Rust parent during long-running operations (typically training).

Heavy imports (torch, pypots) are deferred until the first method that needs
them so that ``health()`` always answers quickly even if torch is missing.
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Callable


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

def health(_params: dict[str, Any], _emit: Callable[[str, dict], None]) -> dict[str, Any]:
    """Quick sanity check: Python version, torch presence, GPU availability."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_available": False,
        "cuda_available": False,
        "device_name": None,
        "vram_gb": None,
    }
    try:
        import torch  # noqa: WPS433
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / 1e9, 1)
    except ImportError:
        pass

    try:
        import pypots  # noqa: WPS433
        info["pypots_version"] = pypots.__version__ if hasattr(pypots, "__version__") else "unknown"
    except ImportError:
        info["pypots_version"] = None

    return info


# ---------------------------------------------------------------------------
# train / predict / model_* — implemented in next iterations
# ---------------------------------------------------------------------------
# Stubs are intentionally explicit so the Rust side can already wire to them
# and we can fill in the implementation without changing the public surface.

def train(params: dict[str, Any], emit: Callable[[str, dict], None]) -> dict[str, Any]:
    from .training import run_training
    return run_training(params, emit)


def predict(params: dict[str, Any], emit: Callable[[str, dict], None]) -> dict[str, Any]:
    from .inference import run_predict
    return run_predict(params, emit)


def model_save(params: dict[str, Any], _emit: Callable[[str, dict], None]) -> dict[str, Any]:
    from .store import save_to_path
    return save_to_path(params)


def model_load(params: dict[str, Any], _emit: Callable[[str, dict], None]) -> dict[str, Any]:
    from .store import load_from_path
    return load_from_path(params)


def inspect_files(params: dict[str, Any], emit: Callable[[str, dict], None]) -> dict[str, Any]:
    """Load files, parse the sensor tree and per-sensor coverage stats.

    Used by the AiTrainingPage wizard to auto-detect the sensor structure and
    suggest defaults (which sensors, which periods)."""
    from .inspect import run_inspect
    return run_inspect(params, emit)


def list_env_columns(params: dict[str, Any], _emit: Callable[[str, dict], None]) -> dict[str, Any]:
    """Return the numeric column names of an env xlsx so the UI can populate
    a multi-select. Cheap: just reads headers + dtypes, no resampling.
    """
    from .data import load_env_file
    path = params["path"]
    df = load_env_file(path)
    numeric_cols = [c for c in df.columns if str(df[c].dtype).startswith(("float", "int"))]
    return {
        "path": str(path),
        "n_rows": int(len(df)),
        "all_columns": [str(c) for c in df.columns],
        "numeric_columns": numeric_cols,
        "first_ts": df.index.min().isoformat() if len(df) else None,
        "last_ts":  df.index.max().isoformat() if len(df) else None,
    }


def list_models(_params: dict[str, Any], _emit: Callable[[str, dict], None]) -> dict[str, Any]:
    """Return all models currently in the in-process registry."""
    from . import registry
    out = []
    for mid in registry.list_ids():
        e = registry.get(mid)
        out.append({
            "model_id": mid,
            "feat_cols": e.feat_cols,
            "window": e.window,
            "sampling_step": e.sampling_step,
            "meta": e.meta,
        })
    return {"models": out}
