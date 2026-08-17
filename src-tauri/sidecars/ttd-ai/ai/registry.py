"""In-process registry of trained/loaded SAITS models.

Models live here for the lifetime of the sidecar process. Saving/loading to
disk is handled separately by `store.py` (.ttdmodel zip format).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelEntry:
    """A loaded model + everything needed to use it again at inference time."""
    model_id: str
    saits: Any                  # the pypots SAITS instance
    feat_cols: list[str]
    mu: list[float]             # per-feature mean (computed on train)
    sig: list[float]            # per-feature stdev (computed on train)
    window: int
    sampling_step: str          # e.g. "5min"
    config: dict[str, Any]      # model hyperparams (n_layers, d_model, ...)
    meta: dict[str, Any] = field(default_factory=dict)  # train period, metrics, etc.


_REGISTRY: dict[str, ModelEntry] = {}


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def register(entry: ModelEntry) -> None:
    _REGISTRY[entry.model_id] = entry


def get(model_id: str) -> ModelEntry:
    if model_id not in _REGISTRY:
        raise KeyError(f"Unknown model_id: {model_id}")
    return _REGISTRY[model_id]


def list_ids() -> list[str]:
    return list(_REGISTRY.keys())


def remove(model_id: str) -> bool:
    return _REGISTRY.pop(model_id, None) is not None
