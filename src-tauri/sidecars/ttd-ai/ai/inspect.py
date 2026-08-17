"""File inspection — parse the sensor tree + per-sensor coverage stats.

Powers the Phase 2.2 wizard step "Capteurs détectés" in the frontend.
Lightweight on purpose: no model loaded, just pandas + the same loader the
training pipeline uses, so what the wizard shows is exactly what training
will see.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .data import (
    find_continuous_segments,
    load_and_concat,
    parse_sensors,
    resample_to_grid,
)


def run_inspect(params: dict[str, Any], emit: Callable[[str, dict], None]) -> dict[str, Any]:
    """Params: {files: [str], step: str = "5min", max_gap: str = "6h",
                preview_buckets: int = 500}.

    Returns:
        {
          "time_range": {"start": iso, "end": iso, "n_rows": int, "step": str},
          "segments": [{"start": iso, "end": iso, "n_rows": int}],
          "sensors": [...],
          "trees":  [str],
          "branches": [str],
          "preview": {                       # NEW: per-sensor data for the brush chart
            "timestamps":  [iso, ...],       # one per bucket
            "sensor_names":[str, ...],       # matches feat order
            "coverage_pct":[[float,...],...],# (n_sensors, n_buckets), 0..100
            "mean_signal": [[float|null,...],...], # same shape, z-scored
          },
        }
    """
    files = params["files"]
    step = params.get("step", "5min")
    max_gap = params.get("max_gap", "6h")

    emit("inspect_started", {"files": [str(f) for f in files]})

    df_raw = load_and_concat(files)
    sensors = parse_sensors(list(df_raw.columns))
    if not sensors:
        raise ValueError("No TC_/SF_ sensor columns recognised in input files.")

    feat_cols = [s.name for s in sensors]
    df = resample_to_grid(df_raw[feat_cols], step=step)

    segments = find_continuous_segments(df, max_gap=max_gap)

    out_sensors = []
    for s in sensors:
        col = df[s.name]
        notna = col.notna()
        n_points = int(notna.sum())
        n_missing = int(len(col) - n_points)
        if n_points > 0:
            valid_ts = col.index[notna]
            first_seen = valid_ts.min().isoformat()
            last_seen  = valid_ts.max().isoformat()
        else:
            first_seen = last_seen = None
        out_sensors.append({
            "name":         s.name,
            "type":         s.type,
            "tree":         s.tree,
            "branch":       s.branch,
            "num":          s.num,
            "side":         s.side,
            "n_points":     n_points,
            "n_missing":    n_missing,
            "coverage_pct": float(round(100.0 * n_points / max(len(col), 1), 1)),
            "first_seen":   first_seen,
            "last_seen":    last_seen,
        })

    trees = sorted({s.tree for s in sensors})
    branches = sorted({s.branch for s in sensors})

    # Build the brush-chart preview: bucket the timeline into ~N buckets
    # and compute, per bucket, the % of sensors with any data plus the
    # z-score'd mean signal. This is what the frontend renders so the
    # user can visually select training periods.
    n_buckets = int(params.get("preview_buckets", 500))
    preview = _build_preview(df, n_buckets=n_buckets)

    return {
        "time_range": {
            "start":  df.index.min().isoformat() if len(df) else None,
            "end":    df.index.max().isoformat() if len(df) else None,
            "n_rows": int(len(df)),
            "step":   step,
        },
        "segments": [
            {
                "start":  s.isoformat(),
                "end":    e.isoformat(),
                "n_rows": int(len(df.loc[s:e])),
            }
            for s, e in segments
        ],
        "sensors": out_sensors,
        "trees":   trees,
        "branches": branches,
        "preview": preview,
    }


def _build_preview(df: pd.DataFrame, n_buckets: int = 500) -> dict[str, Any]:
    """Bucket the timeline into ``n_buckets`` and report two curves PER SENSOR:
    coverage (% non-null per bucket) and z-score'd mean. The frontend
    aggregates across the user's selected subset on the fly, so toggling
    sensors in the wizard updates the chart instantly without a roundtrip.
    """
    if len(df) == 0 or len(df.columns) == 0:
        return {"timestamps": [], "sensor_names": [], "coverage_pct": [], "mean_signal": []}

    # Per-sensor z-score so the mean across heterogeneous sensors makes sense.
    mu  = df.mean(axis=0)
    sig = df.std(axis=0).replace(0, 1.0)
    z = (df - mu) / sig

    span_seconds = (df.index.max() - df.index.min()).total_seconds()
    bucket_seconds = max(int(span_seconds / max(n_buckets, 1)), 60)
    rule = f"{bucket_seconds}s"

    # Per-sensor coverage % per bucket: notna fraction within each bucket.
    cov_per_sensor = df.notna().astype("float64").resample(rule).mean() * 100.0
    # Per-sensor mean (z-scored) per bucket. NaN if all-null inside.
    z_per_sensor = z.resample(rule).mean()

    sensor_names = list(df.columns)
    timestamps = [t.isoformat() for t in cov_per_sensor.index]

    # JSON shape: parallel arrays (sensor_names i ↔ coverage_pct[i] ↔ mean_signal[i]).
    cov_arr = [[float(round(v, 1)) if pd.notna(v) else 0.0 for v in cov_per_sensor[name].values]
               for name in sensor_names]
    z_arr   = [[float(round(v, 3)) if pd.notna(v) else None for v in z_per_sensor[name].values]
               for name in sensor_names]

    return {
        "timestamps":   timestamps,
        "sensor_names": sensor_names,
        "coverage_pct": cov_arr,
        "mean_signal":  z_arr,
    }
