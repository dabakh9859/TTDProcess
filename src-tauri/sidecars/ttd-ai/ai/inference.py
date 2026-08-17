"""Predict / impute a gap with a trained SAITS model."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from . import registry
from .data import (
    load_and_concat, resample_to_grid,
    load_env_file, align_env_to_grid, build_time_features,
    CLIP_Z,
)


def run_predict(params: dict[str, Any], _emit: Callable[[str, dict], None]) -> dict[str, Any]:
    """Reconstruct a target column over a gap by running SAITS over a
    sliding window and averaging overlapping predictions.

    Params:
      model_id        str         the trained model in the registry
      files           list[str]   xlsx paths covering the period to impute
      target_column   str         e.g. "SF_1a-TA-1"
      gap_start       str         "YYYY-MM-DD..." inclusive
      gap_end         str         "YYYY-MM-DD..." exclusive of last day
      env_file        str         override env xlsx (defaults to model's training env)
    """
    model_id = params["model_id"]
    files = params["files"]
    target_col = params["target_column"]
    gap_start = pd.Timestamp(params["gap_start"])
    gap_end   = pd.Timestamp(params["gap_end"])

    entry = registry.get(model_id)
    feat_cols = entry.feat_cols
    mu = pd.Series(entry.mu, index=feat_cols)
    sig = pd.Series(entry.sig, index=feat_cols)
    window = entry.window
    saits = entry.saits

    # Split feat_cols back into sensor / env / time groups so we can rebuild
    # each piece from the right source. Order MUST match training:
    # sensors ++ env (prefix env_) ++ time (prefix time_).
    sensor_cols = [c for c in feat_cols if not c.startswith("env_") and not c.startswith("time_")]
    env_feat_cols = [c for c in feat_cols if c.startswith("env_")]
    time_feat_cols = [c for c in feat_cols if c.startswith("time_")]

    # Load sensor data and align.
    df_raw = load_and_concat(files)

    # Column substitution: if target_col is NOT in the model's features,
    # find the best correlated model column and substitute it. This lets the
    # user clean/complete columns that weren't in training.
    proxy_col = None
    target_mu = None
    target_sig = None
    if target_col not in feat_cols:
        if target_col not in df_raw.columns:
            raise ValueError(f"target_column '{target_col}' not found in data files")
        # Find the most correlated sensor column in the model
        best_corr = -1.0
        best_proxy = None
        for sc in sensor_cols:
            if sc not in df_raw.columns:
                continue
            valid = df_raw[[sc, target_col]].dropna()
            if len(valid) < 50:
                continue
            c = valid[sc].corr(valid[target_col])
            if abs(c) > best_corr:
                best_corr = abs(c)
                best_proxy = sc
        if best_proxy is None:
            raise ValueError(
                f"target_column '{target_col}' not in model features and no correlated proxy found"
            )
        proxy_col = best_proxy
        # Compute target's own statistics for de-normalization
        vals = df_raw[target_col].dropna()
        target_mu = float(vals.mean())
        target_sig = float(vals.std())
        if target_sig < 1e-9:
            target_sig = 1.0

    # Features the model knows but that this file does not carry. Typical case:
    # the model was trained with EXTRA station files, so its feature set spans
    # several stations, while prediction runs on a single station's export.
    #
    # This used to abort. It should not: SAITS represents missingness natively
    # through its mask, so an absent column is simply "never observed" — the
    # tensor keeps the shape the network expects, and the attention falls back
    # on the columns that ARE present (plus meteo and calendar inputs). The
    # reconstruction is naturally less informed than with every station on hand,
    # but it is exactly the situation the model is built for.
    missing = [c for c in sensor_cols if c not in df_raw.columns]
    if missing:
        present = [c for c in sensor_cols if c in df_raw.columns]
        if not present:
            raise ValueError(
                "None of the model's sensors are present in the input files "
                f"(model expects {len(sensor_cols)} columns such as {sensor_cols[:3]}). "
                "This model was trained on a different dataset."
            )
        for c in missing:
            df_raw[c] = np.nan
        _emit("warning", {
            "step": "missing_features_filled",
            "n_missing": len(missing),
            "n_present": len(present),
            "missing": missing[:20],
        })
    df = resample_to_grid(df_raw[sensor_cols], step=entry.sampling_step)

    # If substituting, replace the proxy column's data with the target column's
    # data (normalized to the proxy's z-score scale). The model reconstructs the
    # proxy slot, and we de-normalize with the target's statistics.
    if proxy_col is not None:
        target_resampled = resample_to_grid(df_raw[[target_col]], step=entry.sampling_step)
        df[proxy_col] = target_resampled[target_col]

    # Rebuild env features if the model was trained with them.
    if env_feat_cols:
        env_path = params.get("env_file") or entry.meta.get("env_file")
        if not env_path:
            raise ValueError(
                "Model was trained with env features but no env_file is set and meta.env_file is missing. "
                "Pass env_file in the predict params."
            )
        # Original column names = env_feat_cols stripped of the "env_" prefix.
        env_orig_cols = [c[len("env_"):] for c in env_feat_cols]
        env_raw = load_env_file(env_path)
        env_aligned = align_env_to_grid(env_raw, df.index, columns=env_orig_cols, step=entry.sampling_step)
        env_aligned = env_aligned.add_prefix("env_")
        df = pd.concat([df, env_aligned], axis=1)

    # Rebuild time features if any.
    if time_feat_cols:
        time_orig_names = [c[len("time_"):] for c in time_feat_cols]
        tf = build_time_features(df.index, time_orig_names)
        tf = tf.add_prefix("time_")
        df = pd.concat([df, tf], axis=1)

    # Defensive: ensure column order matches training. SAITS is sensitive to
    # feature order since it uses indexed mu/sig.
    df = df[feat_cols]

    # Two modes:
    #   • detection (default): blank the WHOLE target over [gap_start, gap_end]
    #     so SAITS reconstructs it independently of its own values — the
    #     prediction we compare against to find outliers.
    #   • gap filling (impute_existing=True): keep the observed target values
    #     and only let SAITS impute the cells that are ALREADY missing (NaN).
    #     This gives the model the sensor's own neighbouring values as context,
    #     so fills track the real dynamics instead of a cross-sensor guess.
    impute_existing = bool(params.get("impute_existing"))
    mask_col = proxy_col if proxy_col is not None else target_col
    df_corr = df.copy()
    if not impute_existing:
        df_corr.loc[gap_start:gap_end, mask_col] = np.nan

    # Slide window over the entire timeline, average overlaps. Clip the
    # standardized inputs to the same band used in training so an observed
    # dropout artifact (|z| ~ 30+) in the context can't drag the reconstruction.
    # For column substitution, normalize the proxy slot with the TARGET's stats
    # so z-scores are comparable.
    norm_mu = mu.copy()
    norm_sig = sig.copy()
    if proxy_col is not None:
        norm_mu[proxy_col] = target_mu
        norm_sig[proxy_col] = target_sig
    X = np.clip(((df_corr - norm_mu) / norm_sig).values, -CLIP_Z, CLIP_Z)  # (T, F)
    T, F = X.shape
    if T < window:
        raise ValueError(f"Not enough data points ({T}) for window={window}.")
    stride = window // 2
    starts = list(range(0, T - window + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != T - window:
        starts.append(T - window)
    W = np.stack([X[s:s + window] for s in starts])

    Y = np.asarray(saits.impute({"X": W}))  # (n_windows, window, F)

    # Stitch overlapping windows into one timeline. Two refinements over a plain
    # average, both aimed at the "reconstruction inexacte" symptom:
    #   1. Skip windows that are almost entirely missing across EVERY feature —
    #      SAITS has nothing to condition on and returns near-mean garbage that
    #      would pollute the overlap average. The threshold is on ALL features,
    #      so a window keeps contributing as long as it still has the (always
    #      present) time/env features — which is exactly how a system-wide
    #      sensor gap still gets a sensible diurnal reconstruction.
    #   2. Weight each window with a triangular taper so a timestep is dominated
    #      by the window that holds it near its CENTRE (max temporal context on
    #      both sides) rather than one that only catches it at the very edge.
    #      Kills the boundary seams / flat spikes at the edges of long gaps.
    taper = np.bartlett(window + 2)[1:-1]      # length=window, ~0 at the ends
    taper = np.clip(taper, 0.05, None)         # floor so isolated coverage still counts
    wgt = taper[:, None]
    stitched = np.zeros((T, F))
    counts = np.zeros((T, F))
    for k, s in enumerate(starts):
        if np.isnan(W[k]).mean() > 0.95:
            continue
        stitched[s:s + window] += Y[k] * wgt
        counts[s:s + window] += wgt
    stitched = stitched / np.maximum(counts, 1e-9)

    # When using column substitution, the proxy slot holds the predictions.
    impute_col = proxy_col if proxy_col is not None else target_col
    tgt_idx = feat_cols.index(impute_col)
    if proxy_col is not None:
        # De-normalize with the TARGET's statistics, not the proxy's.
        pred = stitched[:, tgt_idx] * target_sig + target_mu
    else:
        pred = stitched[:, tgt_idx] * entry.sig[tgt_idx] + entry.mu[tgt_idx]

    # Guard against aberrant fills: bound the reconstruction to a robust band of
    # the target's OWN observed values (median ± K·MAD). On near-constant
    # channels with rare dropout artifacts (e.g. T600) this clips the occasional
    # dropout-magnitude fill back to the plausible range; on high-variance
    # signals the MAD band is wide, so legitimate dynamics pass untouched.
    obs_col = target_col if proxy_col is not None else target_col
    obs = (df_raw[obs_col] if proxy_col is not None else df[target_col]).reindex(df.index).to_numpy(dtype=float)
    obs = obs[np.isfinite(obs)]
    if obs.size >= 20:
        med = float(np.median(obs))
        mad = float(np.median(np.abs(obs - med))) * 1.4826
        if mad > 0:
            k = 10.0
            pred = np.clip(pred, med - k * mad, med + k * mad)

    pred_series = pd.Series(pred, index=df.index, name=target_col)
    gap_series = pred_series.loc[gap_start:gap_end]
    if proxy_col is not None:
        actual_resampled = resample_to_grid(df_raw[[target_col]], step=entry.sampling_step)
        actual_series = actual_resampled.loc[gap_start:gap_end, target_col]
    else:
        actual_series = df.loc[gap_start:gap_end, target_col]

    def _f(v):
        return float(v) if v is not None and np.isfinite(v) else None

    # The model's TYPICAL reconstruction error for this column (z units, learned
    # on held-out good data), converted to physical units. The detector uses it
    # as its threshold scale: flag a point only when its error far exceeds the
    # model's normal error — instead of thresholding the residual distribution,
    # which collapses on near-constant channels and over-flags.
    if proxy_col is not None:
        recon_z = (entry.meta.get("recon_error_z") or {}).get(proxy_col)
        recon_error = float(recon_z) * target_sig if recon_z is not None else None
    else:
        recon_z = (entry.meta.get("recon_error_z") or {}).get(target_col)
        recon_error = float(recon_z) * float(sig[target_col]) if recon_z is not None else None

    return {
        "target_column": target_col,
        "gap_start": str(gap_start),
        "gap_end":   str(gap_end),
        "n_points":  int(len(gap_series)),
        "recon_error": _f(recon_error),
        "imputed": [
            {"timestamp": ts.isoformat(), "value": _f(v)}
            for ts, v in gap_series.items()
        ],
        "actual": [
            {"timestamp": ts.isoformat(), "value": _f(v)}
            for ts, v in actual_series.items()
        ],
    }
