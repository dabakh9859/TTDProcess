"""Save / load SAITS model entries as `.ttdmodel` zip files.

A `.ttdmodel` is a plain zip with three entries:
  - weights.pt    SAITS state_dict, torch.save'd
  - config.json   feat_cols, mu, sig, window, sampling_step, hyperparams
  - meta.json     train period, source files, metrics, sidecar version

The format is intentionally simple so a researcher can inspect or even
hand-edit the meta. The weights blob is the only opaque piece.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import torch

from . import registry


SIDECAR_VERSION = 1


def save_to_path(params: dict[str, Any]) -> dict[str, Any]:
    """params: {model_id, path}. Writes <path> as a .ttdmodel zip."""
    model_id = params["model_id"]
    out_path = Path(params["path"])

    entry = registry.get(model_id)

    config = {
        "feat_cols": entry.feat_cols,
        "mu": entry.mu,
        "sig": entry.sig,
        "window": entry.window,
        "sampling_step": entry.sampling_step,
        "hyperparams": entry.config,
        "sidecar_version": SIDECAR_VERSION,
    }
    meta = dict(entry.meta)

    weights_buf = io.BytesIO()
    # ``model.state_dict()`` is the canonical way to serialise pypots SAITS.
    torch.save(entry.saits.model.state_dict(), weights_buf)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", json.dumps(config, indent=2))
        zf.writestr("meta.json", json.dumps(meta, indent=2))
        zf.writestr("weights.pt", weights_buf.getvalue())

    return {"path": str(out_path), "size_bytes": out_path.stat().st_size}


def load_from_path(params: dict[str, Any]) -> dict[str, Any]:
    """params: {path, device?}. Loads the .ttdmodel and registers a new
    in-memory model. Returns the new model_id."""
    in_path = Path(params["path"])
    device_pref = params.get("device", "auto")

    if device_pref == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_pref

    with zipfile.ZipFile(in_path, "r") as zf:
        config = json.loads(zf.read("config.json"))
        meta = json.loads(zf.read("meta.json"))
        weights_bytes = zf.read("weights.pt")

    if config.get("sidecar_version", 0) > SIDECAR_VERSION:
        raise ValueError(
            f"Model file format v{config['sidecar_version']} is newer than this "
            f"sidecar (v{SIDECAR_VERSION}). Update the app."
        )

    feat_cols = config["feat_cols"]
    hp = config["hyperparams"]

    from pypots.imputation import SAITS
    from pypots.optim import Adam

    saits = SAITS(
        n_steps=config["window"],
        n_features=len(feat_cols),
        n_layers=hp["n_layers"],
        d_model=hp["d_model"],
        n_heads=hp["n_heads"],
        d_k=hp["d_k"],
        d_v=hp["d_v"],
        d_ffn=hp["d_ffn"],
        dropout=hp.get("dropout", 0.1),
        attn_dropout=hp.get("attn_dropout", 0.1),
        ORT_weight=hp.get("ORT_weight", 1.0),
        MIT_weight=hp.get("MIT_weight", 1.0),
        batch_size=hp.get("batch_size", 32),
        epochs=1,                  # fit() won't be called again on this instance
        patience=0,                # must be < epochs; irrelevant for inference
        optimizer=Adam(lr=hp.get("learning_rate", 1e-3)),
        num_workers=0,
        device=device,
        saving_path=None,
        model_saving_strategy=None,
    )

    # pypots 1.5 builds the underlying torch model in __init__, so saits.model
    # already exists — load the saved weights straight in (no private builder).
    state = torch.load(io.BytesIO(weights_bytes), map_location=device, weights_only=False)
    saits.model.load_state_dict(state)
    saits.model.to(device)
    saits.model.eval()

    model_id = registry.new_id()
    entry = registry.ModelEntry(
        model_id=model_id,
        saits=saits,
        feat_cols=feat_cols,
        mu=config["mu"],
        sig=config["sig"],
        window=config["window"],
        sampling_step=config["sampling_step"],
        config=hp,
        meta=meta,
    )
    registry.register(entry)

    return {
        "model_id": model_id,
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "device": device,
        "meta": meta,
    }
